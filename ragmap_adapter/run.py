"""``ragmap-run``: build an original HOV-SG graph from a RAGMAP RGB-D scene.

    ragmap-run --input /input --output /output [hydra overrides ...]

Follows ``application/create_graph.py`` step for step (feature map, masked
point clouds, full point cloud, ``Graph.build_graph``), then names rooms with
HOV-SG's CLIP view-embedding classifier and exports the RAGMAP contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

logger = logging.getLogger("ragmap_adapter")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="ragmap-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. pipeline.skip_frames=5")
    return parser.parse_args(argv)


def _compose(input_dir: Path, output_dir: Path, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    base = [f"main.dataset_path={input_dir}", f"main.save_path={output_dir / 'hovsg'}"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="ragmap", overrides=base + list(overrides))


def _resolve_checkpoints(cfg) -> dict:
    from ragmap_adapter.weights import CLIP_FILES, SAM_FILES, ensure_checkpoint

    weights = Path(cfg.ragmap.weights_dir)
    if cfg.models.clip.checkpoint == "auto":
        if cfg.models.clip.type not in CLIP_FILES:
            raise ValueError(f"no default checkpoint for CLIP type {cfg.models.clip.type}; set models.clip.checkpoint")
        cfg.models.clip.checkpoint = str(weights / CLIP_FILES[cfg.models.clip.type])
    if cfg.models.sam.checkpoint == "auto":
        cfg.models.sam.checkpoint = str(weights / SAM_FILES[cfg.models.sam.type])
    started = time.monotonic()
    ensure_checkpoint(cfg.models.clip.checkpoint)
    ensure_checkpoint(cfg.models.sam.checkpoint)
    return {"clip": cfg.models.clip.checkpoint, "sam": cfg.models.sam.checkpoint, "ensure_seconds": time.monotonic() - started}


def _git_sha() -> str | None:
    sha = os.environ.get("HOVSG_GIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def run(input_dir: Path, output_dir: Path, overrides: list[str]) -> dict:
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from ragmap_adapter.cpu_shim import install_if_no_cuda

    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    report: dict = {
        "status": "running",
        "input": str(input_dir),
        "output": str(output_dir),
        "overrides": list(overrides),
        "hovsg_git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    def tick(name, since):
        timings[name] = round(time.monotonic() - since, 3)

    t_all = time.monotonic()
    cfg = _compose(input_dir, output_dir, overrides)
    OmegaConf.set_struct(cfg, False)
    report["weights"] = _resolve_checkpoints(cfg)
    report["cpu_shim"] = install_if_no_cuda()

    seed = int(cfg.ragmap.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Upstream resolves some assets relative to the repository root
    # (hovsg/utils/label_feats.py: "hovsg/labels").
    os.chdir(REPO_ROOT)

    from hovsg.graph.graph import Graph

    from ragmap_adapter.dataset import RagmapDataset
    from ragmap_adapter.export import VisibilityConfig, export

    class RagmapGraph(Graph):
        """Upstream Graph; only the navigation graph failure mode is softened."""

        nav_graph_status = "not run"

        def create_nav_graph(self):
            if not cfg.ragmap.nav_graph:
                RagmapGraph.nav_graph_status = "disabled"
                return
            try:
                super().create_nav_graph()
                RagmapGraph.nav_graph_status = "ok"
            except Exception as exc:  # the graph itself is still saved by build_graph
                RagmapGraph.nav_graph_status = f"failed: {type(exc).__name__}: {exc}"
                logger.warning("create_nav_graph failed; continuing without it\n%s", traceback.format_exc())

    save_dir = str(cfg.main.save_path)
    os.makedirs(save_dir, exist_ok=True)

    t = time.monotonic()
    hovsg = RagmapGraph(cfg)
    tick("load_models", t)
    hovsg.dataset = RagmapDataset({"root_dir": str(input_dir), "transforms": None})
    ds = hovsg.dataset
    report["input_meta"] = ds.meta
    report["frames"] = {
        "total": len(ds),
        "skipped_bad_pose": len(ds.skipped_frames),
        "processed": len(range(0, len(ds), int(cfg.pipeline.skip_frames))),
        "skip_frames": int(cfg.pipeline.skip_frames),
        "depth_resolution": [ds.depth_W, ds.depth_H],
    }
    report["up_axis"] = {"declared": ds.up_axis, "world_to_hovsg": ds.world_to_hovsg.tolist()}

    # --- application/create_graph.py, in order ---------------------------------
    t = time.monotonic()
    hovsg.create_feature_map()
    tick("create_feature_map", t)
    t = time.monotonic()
    hovsg.save_masked_pcds(path=save_dir, state="both")  # also drops small/empty masks, as upstream
    hovsg.save_full_pcd(path=save_dir)
    if cfg.ragmap.save_feature_map:
        hovsg.save_full_pcd_feats(path=save_dir)
    tick("save_feature_map", t)
    t = time.monotonic()
    if cfg.pipeline.create_graph:
        hovsg.build_graph(save_path=save_dir)
    tick("build_graph", t)
    report["nav_graph"] = RagmapGraph.nav_graph_status

    # --- room naming: CLIP view-embedding classification (no LLM) --------------
    t = time.monotonic()
    room_types = list(cfg.ragmap.room_types)
    if hovsg.rooms:
        hovsg.generate_room_names(generate_method="view_embedding", default_room_types=room_types)
        # Re-save rooms so the upstream graph/ directory carries the names too.
        rooms_dir = os.path.join(save_dir, "graph", "rooms")
        if os.path.isdir(rooms_dir):
            for room in hovsg.rooms:
                room.save(rooms_dir)
    tick("room_naming", t)

    t = time.monotonic()
    vis = VisibilityConfig(**OmegaConf.to_container(cfg.ragmap.visibility, resolve=True))
    counts = export(hovsg, ds, output_dir, vis, seed=seed)
    tick("export", t)

    timings["total"] = round(time.monotonic() - t_all, 3)
    report.update(
        status="ok",
        counts=counts,
        timings=timings,
        config=OmegaConf.to_container(cfg, resolve=True),
        outputs={
            "objects": "objects.jsonl",
            "rooms": "rooms.jsonl",
            "floors": "floors.jsonl",
            "object_clip_feats": "object_clip_feats.npy",
            "hovsg_native": "hovsg/ (upstream layout; coordinates in HOV-SG's Y-up frame = world_to_hovsg @ world)",
        },
    )
    return report


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    input_dir, output_dir = args.input.resolve(), args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = run(input_dir, output_dir, args.overrides)
        code = 0
    except Exception as exc:
        logger.error("HOV-SG run failed\n%s", traceback.format_exc())
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                  "input": str(input_dir), "output": str(output_dir), "overrides": args.overrides}
        code = 1
    (output_dir / "run.json").write_text(json.dumps(report, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
