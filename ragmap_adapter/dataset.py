"""HOV-SG dataset loader for the RAGMAP object-mapping input contract.

Input directory layout (see README, "RAGMAP adapter"):

    meta.json     {"fx","fy","cx","cy","width","height","depth_scale","up_axis","frame"}
    frames.jsonl  one {"frame_index","observation_id","rgb","depth","pose"} per line,
                  pose = 16 floats, row-major 4x4 camera-to-world, OpenCV camera
                  convention (x right, y down, z forward).

Up axis
-------
HOV-SG's graph construction hard-codes **+Y as the vertical axis** of the world
frame: ``Graph.segment_floors`` histograms ``points[:, 1]``, ``segment_rooms``
slices on ``[:, 1]`` and projects onto ``[:, [0, 2]]``, the navigation graph and
``compute_room_embeddings`` read camera height from ``pose[1, 3]``. That is the
Habitat (HM3D) world convention; the HM3DSem loader only converts the camera
from OpenGL to OpenCV (``pose @ diag(1, -1, -1, 1)``) and leaves the world Y-up.

RAGMAP worlds are floor-aligned with a declared ``up_axis``. This loader
left-multiplies every pose by a proper rotation ``A`` (``world_to_hovsg``) that
takes the declared up axis to +Y, so HOV-SG sees exactly the frame it was
written for. Everything exported back to RAGMAP is mapped with ``A^T``.

Cameras are already OpenCV, which is what HOV-SG's back-projection
(``RGBDDataset.create_pcd``) expects, so no camera-side flip is applied.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from hovsg.dataloader.generic import RGBDDataset

logger = logging.getLogger(__name__)

#: Rotation taking the declared world up axis to HOV-SG's +Y. All are proper
#: rotations (det = +1), so handedness is preserved.
_UP_AXIS_TO_HOVSG = {
    # (x, y, z) -> (x, z, -y): world +Z becomes +Y.
    "z": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    # Already HOV-SG's convention.
    "y": np.eye(3),
    # (x, y, z) -> (x, -y, -z): world -Y becomes +Y (180 deg about X).
    "-y": np.diag([1.0, -1.0, -1.0]),
}


def world_to_hovsg(up_axis: str) -> np.ndarray:
    """4x4 transform from the RAGMAP world frame to HOV-SG's Y-up frame."""
    try:
        rot = _UP_AXIS_TO_HOVSG[up_axis]
    except KeyError:
        raise ValueError(f"meta.json up_axis must be one of {sorted(_UP_AXIS_TO_HOVSG)}, got {up_axis!r}") from None
    out = np.eye(4)
    out[:3, :3] = rot
    return out


class RagmapDataset(RGBDDataset):
    """RGB-D sequence in the RAGMAP contract, presented as a HOV-SG dataset.

    ``__getitem__`` returns ``(rgb PIL, depth PIL uint16, pose 4x4, [], K)``
    exactly like ``HM3DSemDataset``; ``pose`` is camera-to-world in HOV-SG's
    Y-up frame.
    """

    def __init__(self, cfg):
        # RGBDDataset.__init__ only sets these fields and calls _get_data_list;
        # it is re-implemented here because that call needs meta.json first.
        self.root_dir = Path(cfg["root_dir"])
        self.transforms = cfg.get("transforms")
        self.meta = json.loads((self.root_dir / "meta.json").read_text())
        self.up_axis = str(self.meta.get("up_axis", "z"))
        self.world_to_hovsg = world_to_hovsg(self.up_axis)
        self.scale = float(self.meta["depth_scale"])
        self.skipped_frames: list[dict] = []
        self.data_list = self._get_data_list()
        if not self.data_list:
            raise ValueError(f"{self.root_dir / 'frames.jsonl'} contains no usable frames")

        self.rgb_W, self.rgb_H = int(self.meta["width"]), int(self.meta["height"])
        self.rgb_intrinsics = np.array(
            [[float(self.meta["fx"]), 0.0, float(self.meta["cx"])],
             [0.0, float(self.meta["fy"]), float(self.meta["cy"])],
             [0.0, 0.0, 1.0]]
        )
        # meta intrinsics describe a width x height image. If depth is stored at
        # another resolution, scale K to it: HOV-SG back-projects at depth
        # resolution (it resizes RGB to the depth size).
        depth_w, depth_h = Image.open(self.data_list[0]["depth"]).size
        self.depth_W, self.depth_H = depth_w, depth_h
        sx, sy = depth_w / self.rgb_W, depth_h / self.rgb_H
        self.depth_intrinsics = np.diag([sx, sy, 1.0]) @ self.rgb_intrinsics

    def _get_data_list(self):
        frames = []
        with open(self.root_dir / "frames.jsonl") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pose = np.asarray(rec["pose"], dtype=np.float64)
                if pose.size != 16 or not np.all(np.isfinite(pose)):
                    self.skipped_frames.append({"frame_index": rec.get("frame_index"), "reason": "non-finite or malformed pose"})
                    continue
                frames.append(
                    {
                        "frame_index": int(rec["frame_index"]),
                        "observation_id": rec.get("observation_id"),
                        "rgb": str(self.root_dir / rec["rgb"]),
                        "depth": str(self.root_dir / rec["depth"]),
                        "pose_world": pose.reshape(4, 4),
                    }
                )
        frames.sort(key=lambda f: f["frame_index"])
        if self.skipped_frames:
            logger.warning("skipped %d frames with unusable poses", len(self.skipped_frames))
        return frames

    def __getitem__(self, idx):
        rec = self.data_list[idx]
        rgb_image = self._load_image(rec["rgb"])
        depth_image = self._load_depth(rec["depth"])
        pose = self._load_pose(idx)
        return rgb_image, depth_image, pose, list(), self.depth_intrinsics

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _load_depth(self, path):
        depth = Image.open(path)
        arr = np.asarray(depth)
        if arr.dtype != np.uint16:
            # Keep HOV-SG's expectation of an integer depth PNG in `scale` units.
            depth = Image.fromarray(arr.astype(np.uint16))
        return depth

    def _load_pose(self, idx):
        return self.world_to_hovsg @ self.data_list[idx]["pose_world"]

    def _load_depth_intrinsics(self, *_args):
        return self.depth_intrinsics

    def frame_index(self, idx: int) -> int:
        """Contract ``frame_index`` of dataset position ``idx``."""
        return self.data_list[idx]["frame_index"]
