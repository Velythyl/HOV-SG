"""Translate a built HOV-SG graph into the RAGMAP object-mapping output contract."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from ragmap_adapter.dataset import RagmapDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisibilityConfig:
    max_points_per_object: int = 2000
    depth_tolerance: float = 0.05  # metres, absolute floor of the depth test
    depth_tolerance_rel: float = 0.05  # fraction of the measured depth
    min_points: int = 10
    min_fraction: float = 0.1
    crop_margin: float = 0.1


def _to_world(points_hovsg: np.ndarray, world_to_hovsg: np.ndarray) -> np.ndarray:
    # p_h = A p_w  =>  p_w = A^T p_h  =>  row form: p_w = p_h @ A
    return points_hovsg @ world_to_hovsg[:3, :3]


def _true_colors(points: np.ndarray, full_tree: cKDTree, full_colors: np.ndarray) -> np.ndarray:
    # Graph.save_masked_pcds paints every mask point cloud one random colour.
    # Mask points are copies of full-map points, so the nearest full-map point
    # carries the observed RGB.
    _, idx = full_tree.query(points, k=1, workers=-1)
    return full_colors[idx]


def _visibility(dataset: RagmapDataset, samples: list[np.ndarray], cfg: VisibilityConfig):
    """Per object: visible-point counts per dataset frame, and the pixel box in its best frame.

    A sampled object point counts as seen in a frame when it projects inside
    the depth image, in front of the camera, and agrees with the measured
    depth there (so occluded points do not count). Returns ``counts`` of shape
    (objects, frames) and ``best_box`` (objects, 4) = umin, vmin, umax, vmax in
    depth pixels for the frame with the most visible points.
    """
    n_obj, n_frames = len(samples), len(dataset)
    counts = np.zeros((n_obj, n_frames), dtype=np.int32)
    best_box = np.zeros((n_obj, 4), dtype=np.int64)
    if n_obj == 0:
        return counts, best_box
    owner = np.concatenate([np.full(len(s), i) for i, s in enumerate(samples)])
    pts_h = np.hstack([np.concatenate(samples), np.ones((len(owner), 1))])
    K = dataset.depth_intrinsics
    H, W = dataset.depth_H, dataset.depth_W
    best_count = np.zeros(n_obj, dtype=np.int32)
    for f in range(n_frames):
        depth = np.asarray(dataset._load_depth(dataset.data_list[f]["depth"]), dtype=np.float32) / dataset.scale
        pc = (np.linalg.inv(dataset._load_pose(f)) @ pts_h.T).T[:, :3]
        z = pc[:, 2]
        front = z > 1e-3
        u = np.full(len(z), -1, dtype=np.int64)
        v = np.full(len(z), -1, dtype=np.int64)
        u[front] = np.round(K[0, 0] * pc[front, 0] / z[front] + K[0, 2]).astype(np.int64)
        v[front] = np.round(K[1, 1] * pc[front, 1] / z[front] + K[1, 2]).astype(np.int64)
        idx = np.nonzero(front & (u >= 0) & (u < W) & (v >= 0) & (v < H))[0]
        d = depth[v[idx], u[idx]]
        tol = np.maximum(cfg.depth_tolerance, cfg.depth_tolerance_rel * d)
        idx = idx[(d > 0) & (np.abs(z[idx] - d) < tol)]
        c = np.bincount(owner[idx], minlength=n_obj)
        counts[:, f] = c
        better = c > best_count
        if better.any():
            best_count[better] = c[better]
            keep = better[owner[idx]]
            o, uu, vv = owner[idx][keep], u[idx][keep], v[idx][keep]
            box = np.stack([np.full(n_obj, W), np.full(n_obj, H), np.full(n_obj, -1), np.full(n_obj, -1)], axis=1)
            np.minimum.at(box[:, 0], o, uu)
            np.minimum.at(box[:, 1], o, vv)
            np.maximum.at(box[:, 2], o, uu)
            np.maximum.at(box[:, 3], o, vv)
            best_box[better] = box[better]
    return counts, best_box


def export(graph, dataset: RagmapDataset, out_dir: Path, cfg: VisibilityConfig, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "objects").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    A = dataset.world_to_hovsg
    rng = np.random.default_rng(seed)

    full_points = np.asarray(graph.full_pcd.points)
    full_colors = np.asarray(graph.full_pcd.colors)
    full_tree = cKDTree(full_points)

    rooms_by_id = {room.room_id: room for room in graph.rooms}
    objects = list(graph.objects)

    samples = []
    for obj in objects:
        pts = np.asarray(obj.pcd.points)
        if len(pts) > cfg.max_points_per_object:
            pts = pts[rng.choice(len(pts), cfg.max_points_per_object, replace=False)]
        samples.append(pts)
    counts, best_box = _visibility(dataset, samples, cfg)

    feats = []
    records = []
    for i, obj in enumerate(objects):
        oid = str(obj.object_id)
        pts_h = np.asarray(obj.pcd.points)
        pts_w = _to_world(pts_h, A)
        colors = _true_colors(pts_h, full_tree, full_colors) if len(pts_h) else np.zeros((0, 3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_w)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out_dir / "objects" / f"{oid}.ply"), pcd)

        n_sample = max(len(samples[i]), 1)
        need = max(cfg.min_points, int(np.ceil(cfg.min_fraction * n_sample)))
        seen = np.nonzero(counts[i] >= need)[0]
        frame_indices = [dataset.frame_index(int(f)) for f in seen]

        crop_rel = None
        best = int(np.argmax(counts[i])) if counts.shape[1] else -1
        if best >= 0 and counts[i, best] >= cfg.min_points:
            if _write_crop(dataset, best, best_box[i], out_dir / "crops" / f"{oid}.png", cfg.crop_margin):
                crop_rel = f"crops/{oid}.png"

        room = rooms_by_id.get(obj.room_id)
        floor_id = room.floor_id if room is not None else str(obj.room_id).split("_")[0]
        emb = np.asarray(obj.embedding, dtype=np.float32).reshape(-1) if obj.embedding is not None else None
        extra = {
            "hovsg_object_id": oid,
            "hovsg_room_id": obj.room_id,
            "num_points": int(len(pts_h)),
            "best_frame_index": dataset.frame_index(best) if crop_rel else None,
            "frame_indices_method": "reprojection+depth-test of the object point cloud",
            "label_vocabulary": str(graph.cfg.pipeline.obj_labels),
        }
        if emb is not None:
            extra["clip_feature_row"] = len(feats)
            feats.append(emb)
        records.append(
            {
                "id": oid,
                "label": obj.name,
                "caption": None,
                "crop": crop_rel,
                "centroid": pts_w.mean(axis=0).tolist() if len(pts_w) else None,
                "bbox_min": pts_w.min(axis=0).tolist() if len(pts_w) else None,
                "bbox_max": pts_w.max(axis=0).tolist() if len(pts_w) else None,
                "pointcloud": f"objects/{oid}.ply",
                "frame_indices": frame_indices,
                "floor": str(floor_id),
                "room": room.name if room is not None else None,
                "extra": extra,
            }
        )

    with open(out_dir / "objects.jsonl", "w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    if feats:
        np.save(out_dir / "object_clip_feats.npy", np.stack(feats))

    _write_rooms_floors(graph, dataset, out_dir)
    return {"floors": len(graph.floors), "rooms": len(graph.rooms), "objects": len(records)}


def _write_crop(dataset: RagmapDataset, f: int, box_px: np.ndarray, dest: Path, margin: float) -> bool:
    umin, vmin, umax, vmax = (int(x) for x in box_px)
    if umax < umin or vmax < vmin:
        return False
    rgb = dataset._load_image(dataset.data_list[f]["rgb"])
    sx, sy = rgb.size[0] / dataset.depth_W, rgb.size[1] / dataset.depth_H
    u0, u1, v0, v1 = umin * sx, (umax + 1) * sx, vmin * sy, (vmax + 1) * sy
    mu, mv = (u1 - u0) * margin, (v1 - v0) * margin
    box = (
        int(max(0, np.floor(u0 - mu))),
        int(max(0, np.floor(v0 - mv))),
        int(min(rgb.size[0], np.ceil(u1 + mu))),
        int(min(rgb.size[1], np.ceil(v1 + mv))),
    )
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return False
    rgb.crop(box).save(dest)
    return True


def _write_rooms_floors(graph, dataset: RagmapDataset, out_dir: Path) -> None:
    A = dataset.world_to_hovsg
    up_world = A[1, :3]  # world direction that HOV-SG treats as +Y
    with open(out_dir / "floors.jsonl", "w") as handle:
        for floor in graph.floors:
            pts = _to_world(np.asarray(floor.pcd.points), A)
            handle.write(json.dumps({
                "id": str(floor.floor_id),
                "name": floor.name,
                "zero_level": float(floor.floor_zero_level),
                "height": float(floor.floor_height),
                "up_vector": up_world.tolist(),
                "bbox_min": pts.min(axis=0).tolist() if len(pts) else None,
                "bbox_max": pts.max(axis=0).tolist() if len(pts) else None,
                "rooms": [r.room_id for r in floor.rooms],
            }) + "\n")
    with open(out_dir / "rooms.jsonl", "w") as handle:
        for room in graph.rooms:
            pts = _to_world(np.asarray(room.pcd.points), A) if room.pcd is not None else np.zeros((0, 3))
            handle.write(json.dumps({
                "id": room.room_id,
                "name": room.name,
                "floor": str(room.floor_id),
                "centroid": pts.mean(axis=0).tolist() if len(pts) else None,
                "bbox_min": pts.min(axis=0).tolist() if len(pts) else None,
                "bbox_max": pts.max(axis=0).tolist() if len(pts) else None,
                "objects": [o.object_id for o in room.objects],
                "represent_frame_indices": [
                    dataset.frame_index(int(i)) for i in room.represent_images if 0 <= int(i) < len(dataset)
                ],
            }) + "\n")
