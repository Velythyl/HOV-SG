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

    from ragmap_adapter.headless import install as install_headless

    install_headless()
    from hovsg.graph.graph import Graph

    from ragmap_adapter.dataset import RagmapDataset
    from ragmap_adapter.export import VisibilityConfig, export

    class RagmapGraph(Graph):
        """Upstream Graph with three failure modes handled; the algorithm is unchanged.

        A failed navigation graph is recorded instead of losing the whole graph;
        masks emptied by denoising are dropped instead of crashing; and, only when
        ``ragmap.single_storey_fallback`` is set, a map with no floor/ceiling
        peak pair becomes one storey. Each is reported in run.json.
        """

        nav_graph_status = "not run"
        floor_segmentation = "upstream"
        empty_masks_dropped = 0

        def segment_objects(self, save_dir=None):
            # Upstream denoises each mask cloud (DBSCAN, eps=0.05, min_points=10)
            # and then takes np.min of its points, which raises on a mask the
            # denoise emptied. Same denoise, done here once, with those masks
            # dropped from mask_pcds and mask_feats together (they are indexed
            # in step); upstream's own call is then an identity. A run with no
            # emptied mask is unchanged.
            import hovsg.graph.graph as graph_module

            denoise = graph_module.pcd_denoise_dbscan
            kept = [
                (cloud, feats)
                for cloud, feats in (
                    (denoise(pcd, eps=0.05, min_points=10), feats)
                    for pcd, feats in zip(self.mask_pcds, self.mask_feats)
                )
                if len(cloud.points)
            ]
            RagmapGraph.empty_masks_dropped = len(self.mask_pcds) - len(kept)
            if RagmapGraph.empty_masks_dropped:
                logger.warning("dropping %d masks emptied by denoising", RagmapGraph.empty_masks_dropped)
            self.mask_pcds = [cloud for cloud, _ in kept]
            self.mask_feats = [feats for _, feats in kept]
            graph_module.pcd_denoise_dbscan = lambda pcd, **_: pcd
            try:
                return super().segment_objects(save_dir)
            finally:
                graph_module.pcd_denoise_dbscan = denoise

        def segment_floors(self, path, flip_zy=False):
            try:
                return super().segment_floors(path, flip_zy=flip_zy)
            except IndexError:
                # Only upstream's "no floor/ceiling pair" failure, which leaves
                # self.floors empty; anything else is re-raised untouched.
                if self.floors or not cfg.ragmap.single_storey_fallback:
                    raise
            import numpy as np
            import open3d as o3d
            from hovsg.graph.floor import Floor

            points = np.asarray(self.full_pcd.points)
            low, high = float(points[:, 1].min()), float(points[:, 1].max())
            floor_obj = Floor("0", name="floor_0")
            floor_pcd = self.full_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-np.inf, low, -np.inf), max_bound=(np.inf, high, np.inf)))
            floor_obj.vertices = np.asarray(floor_pcd.get_axis_aligned_bounding_box().get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = low
            floor_obj.floor_height = high - low
            self.floors.append(floor_obj)
            RagmapGraph.floor_segmentation = (
                f"single_storey_fallback: no floor/ceiling peak pair; one storey from {low:.3f} to {high:.3f}"
            )
            logger.warning("segment_floors found no floor/ceiling pair; %s", RagmapGraph.floor_segmentation)
            return [[low, high]]

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
    elif cfg.ragmap.save_mask_feats:
        # The mask half of upstream's save_full_pcd_feats, verbatim (graph.py
        # 1261-1265): the per-mask features aligned with objects/pcd_<i>.ply, as
        # OpenLex3D's hovsg_to_openlex_format.py reads them, without the
        # N_points x 1024 full_feats.pt. Upstream's in-place np.array conversion
        # is kept, so build_graph sees what it sees upstream.
        if len(hovsg.mask_feats) != 0:
            hovsg.mask_feats = np.array(hovsg.mask_feats)
            torch.save(torch.from_numpy(hovsg.mask_feats), os.path.join(save_dir, "mask_feats.pt"))
    tick("save_feature_map", t)
    t = time.monotonic()
    if cfg.pipeline.create_graph:
        hovsg.build_graph(save_path=save_dir)
    tick("build_graph", t)
    report["nav_graph"] = RagmapGraph.nav_graph_status
    report["floor_segmentation"] = RagmapGraph.floor_segmentation
    report["empty_masks_dropped"] = RagmapGraph.empty_masks_dropped

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
