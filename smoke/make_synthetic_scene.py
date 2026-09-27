"""Render a tiny synthetic RGB-D scene in the RAGMAP object-mapping input contract.

A closed box room (floor, ceiling, four walls) with a few coloured boxes in it,
observed by a camera spinning in place. Depth is ray-cast analytically, so the
geometry is exact. The scene is built Z-up and can be emitted in any of the
contract's up-axis conventions, which is what the smoke test uses to check that
the adapter maps the vertical axis correctly.

    python smoke/make_synthetic_scene.py --out /tmp/scene [--up-axis z|y|-y] [--frames 24]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOM_MIN = np.array([0.0, 0.0, 0.0])
ROOM_MAX = np.array([6.0, 5.0, 2.7])
# (min corner, max corner, RGB) in the Z-up scene frame.
BOXES = [
    (np.array([1.0, 1.0, 0.0]), np.array([2.2, 1.8, 0.75]), (200, 40, 40)),   # "table"
    (np.array([4.3, 3.6, 0.0]), np.array([5.6, 4.6, 1.9]), (40, 60, 200)),    # "cabinet"
    (np.array([4.2, 0.6, 0.0]), np.array([4.9, 1.3, 0.5]), (40, 170, 60)),    # "stool"
    (np.array([0.8, 3.8, 0.0]), np.array([1.6, 4.4, 1.0]), (230, 200, 30)),   # "box"
]
# Wall/floor/ceiling base colours by face index: -x, +x, -y, +y, floor, ceiling.
FACE_COLORS = [(180, 170, 150), (160, 175, 185), (190, 180, 170), (170, 160, 180), (120, 90, 60), (235, 235, 235)]

#: Z-up scene coordinates -> declared world frame (proper rotations).
SCENE_TO_WORLD = {
    "z": np.eye(3),
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "-y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}


def _look_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Camera-to-scene rotation for an OpenCV camera (x right, y down, z forward), Z-up scene."""
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)


def _render(K, W, H, R, t):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    d = d_cam @ R.T  # scene-frame direction with unit camera-z component: ray param == depth
    n = len(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        # Interior of the room: exit distance through the nearest face.
        t_faces = np.where(inv > 0, (ROOM_MAX - t) * inv, (ROOM_MIN - t) * inv)
        t_faces[~np.isfinite(t_faces)] = np.inf
        best_t = t_faces.min(axis=1)
        axis = t_faces.argmin(axis=1)
        face = axis * 2 + (d[np.arange(n), axis] > 0)
        # face ids: 0:-x 1:+x 2:-y 3:+y 4:floor 5:ceiling
        color_id = face
        for bi, (bmin, bmax, _) in enumerate(BOXES):
            t0 = (bmin - t) * inv
            t1 = (bmax - t) * inv
            tn = np.nanmax(np.minimum(t0, t1), axis=1)
            tf = np.nanmin(np.maximum(t0, t1), axis=1)
            hit = (tn <= tf) & (tn > 0) & (tn < best_t)
            best_t = np.where(hit, tn, best_t)
            color_id = np.where(hit, 6 + bi, color_id)
    pts = t + d * best_t[:, None]
    palette = np.array(FACE_COLORS + [b[2] for b in BOXES], dtype=np.float64)
    rgb = palette[color_id]
    # Checker texture (25 cm) so SAM/CLIP see some structure; boxes get a faint one.
    checker = (np.floor(pts / 0.25).astype(np.int64).sum(axis=1) % 2).astype(np.float64)
    strength = np.where(color_id >= 6, 0.08, 0.18)
    rgb = np.clip(rgb * (1.0 - strength * checker)[:, None], 0, 255)
    return rgb.reshape(H, W, 3).astype(np.uint8), best_t.reshape(H, W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--up-axis", default="z", choices=sorted(SCENE_TO_WORLD))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--depth-noise", type=float, default=0.02, help="std-dev in metres")
    args = ap.parse_args()

    W, H = args.width, args.height
    f = 0.62 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    depth_scale = 1000.0
    M = SCENE_TO_WORLD[args.up_axis]
    out = args.out
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    centers = [np.array([3.0, 2.5, 1.5]), np.array([2.6, 2.2, 1.4])]
    with open(out / "frames.jsonl", "w") as handle:
        for i in range(args.frames):
            yaw = 2 * np.pi * i / (args.frames / len(centers))
            c = centers[i % len(centers)]
            # Look down, level and up: HOV-SG's floor segmentation needs both
            # the floor and the ceiling in its height histogram.
            # (irregular in i so every frame stride still sees both).
            pitch = 0.42 * np.cos(1.3 * i)
            R = _look_rotation(yaw, pitch)
            rgb, depth = _render(K, W, H, R, c)
            # Sensor-like noise; without it the floor sits exactly on the lowest
            # histogram bin, where scipy's find_peaks cannot report a peak.
            depth = depth + rng.normal(0.0, args.depth_noise, depth.shape)
            Image.fromarray(rgb).save(out / "rgb" / f"{i:06d}.jpg", quality=95)
            Image.fromarray(np.round(depth * depth_scale).astype(np.uint16)).save(out / "depth" / f"{i:06d}.png")
            pose = np.eye(4)
            pose[:3, :3] = M @ R
            pose[:3, 3] = M @ c
            handle.write(json.dumps({
                "frame_index": i,
                "observation_id": f"synthetic-{i:06d}",
                "rgb": f"rgb/{i:06d}.jpg",
                "depth": f"depth/{i:06d}.png",
                "pose": pose.reshape(-1).tolist(),
            }) + "\n")

    corners = np.array([[x, y, z] for x in (ROOM_MIN[0], ROOM_MAX[0]) for y in (ROOM_MIN[1], ROOM_MAX[1]) for z in (ROOM_MIN[2], ROOM_MAX[2])]) @ M.T
    meta = {
        "fx": f, "fy": f, "cx": W / 2, "cy": H / 2, "width": W, "height": H,
        "depth_scale": depth_scale, "up_axis": args.up_axis, "frame": "ragmap_scene",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    truth = {
        "room_bbox_min": corners.min(0).tolist(),
        "room_bbox_max": corners.max(0).tolist(),
        "boxes": [
            {"centroid": (M @ ((bmin + bmax) / 2)).tolist()} for bmin, bmax, _ in BOXES
        ],
        "up_axis": args.up_axis,
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {args.frames} frames to {out} (up_axis={args.up_axis})")


if __name__ == "__main__":
    main()
