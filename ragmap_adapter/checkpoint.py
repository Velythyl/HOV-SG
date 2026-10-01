"""Resumable ``create_feature_map``: checkpoints of the extraction and the sequential merge.

HOV-SG builds its masks in two long phases inside ``Graph.create_feature_map``:
per-frame SAM + CLIP extraction, then the sequential merge of every frame's 3-D
masks. A run cut off by its allocation's time limit would otherwise start over.
With ``HOVSG_CHECKPOINT_DIR`` set (RAGMAP mounts the step's work directory
there), this module saves

* ``extraction/`` once, when extraction is done and the merge is about to
  start: the filtered full point cloud, its fused per-point features, every
  frame's 3-D masks (points and colours, in order), and the RNG states;
* ``merge.npz`` every ``HOVSG_CHECKPOINT_EVERY_S`` seconds (default 300) of
  merging, and again when the process is sent SIGTERM / SIGUSR1: the global
  masks after the last completed frame, that frame's index, and the RNG states.

On a restart with the same input and configuration (``key.json``) the
extraction is not redone and the merge continues after the last saved frame.
The resumed run's output is byte-identical to an uninterrupted one: the merge
is a deterministic function of the global masks (stored exactly, float64
points and colours in order) and the remaining frames; RNG states are
restored, because ``save_masked_pcds`` draws colours from ``np.random``.

Every write goes to a temporary name and is renamed into place, so a run killed
mid-write leaves the previous checkpoint intact. A checkpoint from different
input or configuration is discarded, never used.

``resumable_create_feature_map`` is upstream's ``Graph.create_feature_map``
from its merge onwards (``hovsg/graph/graph.py``), copied verbatim; the
extraction before it is upstream's own method, run unchanged on a fresh start.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = 1


class Interrupted(BaseException):
    """Raised in the main thread by SIGTERM / SIGUSR1 (BaseException: not caught by ``except Exception``)."""

    def __init__(self, signum: int):
        super().__init__(f"signal {signum}")
        self.signum = signum


def install_signal_handlers() -> None:
    def handler(signum, _frame):
        raise Interrupted(signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGUSR1, handler)


def _rng_state():
    import torch

    state = {"numpy": np.random.get_state(), "python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state) -> None:
    import torch

    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _pack(clouds):
    """List of PointClouds -> (counts, points, colours) arrays, exactly."""
    pts = [np.asarray(c.points) for c in clouds]
    cols = [np.asarray(c.colors) for c in clouds]
    counts = np.array([p.shape[0] for p in pts], dtype=np.int64)
    ccounts = np.array([c.shape[0] for c in cols], dtype=np.int64)
    points = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
    colours = np.concatenate(cols, axis=0) if cols else np.zeros((0, 3))
    return counts, ccounts, points, colours


def _unpack(counts, ccounts, points, colours):
    import open3d as o3d

    out = []
    p0 = c0 = 0
    for n, m in zip(counts.tolist(), ccounts.tolist()):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[p0:p0 + n])
        if m:
            pc.colors = o3d.utility.Vector3dVector(colours[c0:c0 + m])
        out.append(pc)
        p0 += n
        c0 += m
    return out


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Checkpoint:
    def __init__(self, root: Path, key: dict, every_s: float):
        self.root = Path(root)
        self.key = key
        self.every_s = float(every_s)
        self.events: list[dict] = []
        self.root.mkdir(parents=True, exist_ok=True)
        key_path = self.root / "key.json"
        stored = None
        if key_path.is_file():
            try:
                stored = json.loads(key_path.read_text())
            except (OSError, json.JSONDecodeError):
                stored = None
        if stored != key:
            if stored is not None:
                logger.warning("checkpoint in %s is for another input/configuration; discarding it", self.root)
                self.events.append({"event": "discarded_stale", "at": time.time()})
            for child in self.root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            _atomic_write(key_path, lambda fh: fh.write(json.dumps(key, sort_keys=True).encode()))

    # --- extraction ---------------------------------------------------------
    def has_extraction(self) -> bool:
        return (self.root / "extraction" / "done").is_file()

    def save_extraction(self, graph, frames_pcd) -> None:
        t = time.monotonic()
        final = self.root / "extraction"
        tmp = self.root / f"extraction.tmp-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        full_points = np.asarray(graph.full_pcd.points)
        full_colours = np.asarray(graph.full_pcd.colors)
        np.save(tmp / "full_points.npy", full_points)
        np.save(tmp / "full_colours.npy", full_colours)
        np.save(tmp / "full_feats.npy", np.asarray(graph.full_feats_array))
        frame_sizes = np.array([len(f) for f in frames_pcd], dtype=np.int64)
        counts, ccounts, points, colours = _pack([c for f in frames_pcd for c in f])
        np.savez(tmp / "frames.npz", frame_sizes=frame_sizes, counts=counts, ccounts=ccounts, points=points, colours=colours)
        with open(tmp / "rng.pkl", "wb") as fh:
            pickle.dump(_rng_state(), fh, protocol=4)
        (tmp / "done").write_text(json.dumps({"format": FORMAT, "frames": len(frames_pcd)}))
        for child in tmp.iterdir():
            with open(child, "rb") as fh:
                os.fsync(fh.fileno())
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        self.events.append({"event": "saved_extraction", "seconds": round(time.monotonic() - t, 1), "at": time.time()})
        logger.info("checkpoint: extraction saved (%d frames) in %.1fs", len(frames_pcd), time.monotonic() - t)

    def load_extraction(self):
        import open3d as o3d

        d = self.root / "extraction"
        full = o3d.geometry.PointCloud()
        full.points = o3d.utility.Vector3dVector(np.load(d / "full_points.npy"))
        colours = np.load(d / "full_colours.npy")
        if colours.shape[0]:
            full.colors = o3d.utility.Vector3dVector(colours)
        feats = np.load(d / "full_feats.npy")
        z = np.load(d / "frames.npz")
        clouds = _unpack(z["counts"], z["ccounts"], z["points"], z["colours"])
        frames_pcd, at = [], 0
        for n in z["frame_sizes"].tolist():
            frames_pcd.append(clouds[at:at + n])
            at += n
        with open(d / "rng.pkl", "rb") as fh:
            rng = pickle.load(fh)
        self.events.append({"event": "resumed_extraction", "at": time.time()})
        return full, feats, frames_pcd, rng

    # --- merge ----------------------------------------------------------------
    def save_merge(self, index: int, global_masks) -> None:
        t = time.monotonic()
        counts, ccounts, points, colours = _pack(global_masks)
        rng = pickle.dumps(_rng_state(), protocol=4)

        def write(fh):
            np.savez(fh, index=np.int64(index), counts=counts, ccounts=ccounts, points=points, colours=colours,
                     rng=np.frombuffer(rng, dtype=np.uint8))

        _atomic_write(self.root / "merge.npz", write)
        self.events.append({"event": "saved_merge", "index": int(index), "masks": len(global_masks),
                            "seconds": round(time.monotonic() - t, 1), "at": time.time()})
        logger.info("checkpoint: merge saved after frame %d (%d masks) in %.1fs", index, len(global_masks), time.monotonic() - t)

    def load_merge(self):
        path = self.root / "merge.npz"
        if not path.is_file():
            return None
        z = np.load(path)
        masks = _unpack(z["counts"], z["ccounts"], z["points"], z["colours"])
        rng = pickle.loads(z["rng"].tobytes())
        self.events.append({"event": "resumed_merge", "index": int(z["index"]), "at": time.time()})
        return int(z["index"]), masks, rng

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def checkpoint_key(cfg, input_dir: Path, git_sha) -> dict:
    """What the saved state depends on: the input contract files, the config, the code."""
    from omegaconf import OmegaConf

    h = hashlib.sha256()
    for name in ("meta.json", "frames.jsonl"):
        h.update((Path(input_dir) / name).read_bytes())
    conf = OmegaConf.to_container(cfg, resolve=True)
    relevant = {k: conf.get(k) for k in ("pipeline", "models", "main")}
    relevant["ragmap"] = {k: v for k, v in (conf.get("ragmap") or {}).items() if k == "fast_merge"}
    return {"format": FORMAT, "input_sha256": h.hexdigest(), "config": relevant, "hovsg_git_sha": git_sha}


def resumable_seq_merge(ck: Checkpoint, merge_3d_masks, tqdm):
    """Upstream ``seq_merge`` (graph_utils.py), verbatim but for the checkpoints."""

    def seq_merge(frames_pcd, th, down_size, proxy_th):
        # `state` is (last completed frame, global masks after it), rebound in
        # one store so a signal can never see the two out of step.
        state = (0, frames_pcd[0])
        saved = ck.load_merge()
        if saved is not None:
            done, masks, rng = saved
            _set_rng_state(rng)
            state = (done, masks)
            logger.info("checkpoint: resuming the merge after frame %d of %d", done, len(frames_pcd) - 1)
        start = state[0] + 1
        last_saved, last_time = state[0], time.monotonic()
        stop_after = int(os.environ.get("HOVSG_STOP_AFTER_MERGE_ITERS", "0") or 0)  # test hook
        try:
            for i in tqdm(range(start, len(frames_pcd)), initial=start - 1, total=len(frames_pcd) - 1):
                global_masks = state[1]
                mask_list = global_masks + frames_pcd[i]
                merged_mask_list = merge_3d_masks(
                    mask_list,
                    overlap_threshold=th,
                    radius=down_size,
                    iou_thresh=proxy_th,
                )
                state = (i, merged_mask_list)
                if time.monotonic() - last_time >= ck.every_s:
                    ck.save_merge(*state)
                    last_saved, last_time = state[0], time.monotonic()
                if stop_after and i - start + 1 >= stop_after:
                    os.kill(os.getpid(), signal.SIGTERM)
        except Interrupted:
            if state[0] > last_saved:
                ck.save_merge(*state)
            raise
        if state[0] > last_saved:
            ck.save_merge(*state)
        global_masks = state[1]
        # apply one more merge
        global_masks = merge_3d_masks(
            global_masks, overlap_threshold=th, radius=down_size, iou_thresh=proxy_th
        )
        return global_masks

    return seq_merge


def resume_feature_map(graph, ck: Checkpoint, seq_merge) -> None:
    """Upstream ``Graph.create_feature_map`` after its extraction, from a checkpoint.

    Everything from ``# merging the masks`` on is upstream's code (graph.py),
    verbatim, with ``self`` -> ``graph``; only the sequential merge is supported.
    """
    import hovsg.graph.graph as gm
    from scipy.spatial import cKDTree
    from tqdm import tqdm

    full, feats, frames_pcd, rng = ck.load_extraction()
    graph.full_pcd = full
    graph.full_feats_array = feats
    _set_rng_state(rng)
    locs_in = np.array(graph.full_pcd.points)
    tree_pcd = cKDTree(locs_in)
    logger.info("checkpoint: extraction restored (%d frames); skipping SAM/CLIP", len(frames_pcd))

    # --- upstream graph.py, verbatim from here ------------------------------
    # merging the masks
    tqdm.write("Merging 3d masks sequentially")
    graph.mask_pcds = seq_merge(
        frames_pcd,
        graph.cfg.pipeline.init_overlap_thresh,
        graph.cfg.pipeline.voxel_size,
        graph.cfg.pipeline.iou_thresh
    )

    # remove any small pcds
    for i, pcd in enumerate(graph.mask_pcds):
        if pcd.is_empty() or len(pcd.points) < 100:
            graph.mask_pcds.pop(i)
    # fuse point features in every 3d mask
    masks_feats = []
    for i, mask_3d in tqdm(enumerate(graph.mask_pcds), desc="Fusing features"):
        # find the points in the mask
        mask_3d = mask_3d.voxel_down_sample(graph.cfg.pipeline.voxel_size * 2)
        points = np.asarray(mask_3d.points)
        dist, idx = tree_pcd.query(points, k=1, workers=-1)
        feats = graph.full_feats_array[idx]
        feats = np.nan_to_num(feats)
        # filter feats with dbscan
        if feats.shape[0] == 0:
            masks_feats.append(
                np.zeros((1, graph.clip_feat_dim), dtype=graph.full_feats_array.dtype)
            )
            continue
        feats = gm.feats_denoise_dbscan(feats, eps=0.01, min_points=100)
        masks_feats.append(feats)
    graph.mask_feats = masks_feats
    print("number of masks: ", len(graph.mask_feats))
    print("number of pcds in hovsg: ", len(graph.mask_pcds))
    assert len(graph.mask_pcds) == len(graph.mask_feats)
