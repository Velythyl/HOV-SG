"""Validate a ragmap-run output directory against the contract (and the synthetic truth).

    python smoke/check_output.py --output /tmp/out --scene /tmp/scene
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED = {"id", "label", "caption", "crop", "centroid", "bbox_min", "bbox_max", "pointcloud", "frame_indices", "floor", "room", "extra"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scene", required=True, type=Path)
    args = ap.parse_args()
    out, scene = args.output, args.scene
    errors: list[str] = []

    run = json.loads((out / "run.json").read_text())
    if run.get("status") != "ok":
        print(json.dumps(run, indent=2)[:4000])
        return 1
    for key in ("floors", "rooms", "objects"):
        if key not in run.get("counts", {}):
            errors.append(f"run.json counts missing {key}")
    if "timings" not in run or "config" not in run:
        errors.append("run.json missing timings/config")

    frame_ids = {json.loads(l)["frame_index"] for l in (scene / "frames.jsonl").read_text().splitlines() if l.strip()}
    truth = json.loads((scene / "truth.json").read_text())
    lo = np.array(truth["room_bbox_min"]) - 0.25
    hi = np.array(truth["room_bbox_max"]) + 0.25

    objects = [json.loads(l) for l in (out / "objects.jsonl").read_text().splitlines() if l.strip()]
    if not objects:
        errors.append("objects.jsonl is empty")
    ids = set()
    for obj in objects:
        missing = REQUIRED - obj.keys()
        if missing:
            errors.append(f"{obj.get('id')}: missing keys {sorted(missing)}")
            continue
        if obj["id"] in ids:
            errors.append(f"duplicate id {obj['id']}")
        ids.add(obj["id"])
        if not isinstance(obj["label"], str) or not obj["label"]:
            errors.append(f"{obj['id']}: empty label")
        if not isinstance(obj["extra"], dict):
            errors.append(f"{obj['id']}: extra is not an object")
        if not (out / obj["pointcloud"]).is_file():
            errors.append(f"{obj['id']}: missing {obj['pointcloud']}")
        if obj["crop"] is not None and not (out / obj["crop"]).is_file():
            errors.append(f"{obj['id']}: missing {obj['crop']}")
        if not set(obj["frame_indices"]) <= frame_ids:
            errors.append(f"{obj['id']}: frame_indices not in input")
        c, bmin, bmax = (np.array(obj[k], dtype=float) for k in ("centroid", "bbox_min", "bbox_max"))
        if not (np.all(bmin <= c + 1e-6) and np.all(c <= bmax + 1e-6)):
            errors.append(f"{obj['id']}: centroid outside its bbox")
        # Coordinates must be back in the *input* world frame: a wrong up-axis
        # mapping puts objects outside the room (e.g. below the floor).
        if not (np.all(c >= lo) and np.all(c <= hi)):
            errors.append(f"{obj['id']}: centroid {c.round(2).tolist()} outside room {lo.round(2).tolist()}..{hi.round(2).tolist()}")

    # Soft signal: how many ground-truth boxes have an object centroid nearby.
    cents = np.array([o["centroid"] for o in objects]) if objects else np.zeros((0, 3))
    matched = 0
    for box in truth["boxes"]:
        if len(cents) and np.min(np.linalg.norm(cents - np.array(box["centroid"]), axis=1)) < 0.5:
            matched += 1
    print(json.dumps({
        "counts": run["counts"],
        "timings": run["timings"],
        "nav_graph": run.get("nav_graph"),
        "labels": sorted({o["label"] for o in objects}),
        "rooms": sorted({str(o["room"]) for o in objects}),
        "floors": sorted({str(o["floor"]) for o in objects}),
        "with_crop": sum(o["crop"] is not None for o in objects),
        "with_frames": sum(bool(o["frame_indices"]) for o in objects),
        "truth_boxes_matched": f"{matched}/{len(truth['boxes'])}",
    }, indent=2))
    if errors:
        print("FAILED:\n  " + "\n  ".join(errors[:50]))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
