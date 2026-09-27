"""Download-on-first-run for the HOV-SG checkpoints into a mounted cache dir.

The checkpoints are not baked into the image (SAM ViT-H is 2.4 GB, OpenCLIP
ViT-H-14 is 3.9 GB). Mount a persistent directory at ``/weights`` (or set
``HOVSG_WEIGHTS``); the first run fills it, later runs reuse it. Downloads go
to ``<name>.part`` and are renamed into place, under an exclusive ``flock`` so
concurrent containers sharing the cache do not race.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

#: file name (as referenced by HOV-SG's README / config) -> source URL.
CHECKPOINTS = {
    "laion2b_s32b_b79k.bin": f"{HF_ENDPOINT}/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin",
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

SAM_FILES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
CLIP_FILES = {"ViT-H-14": "laion2b_s32b_b79k.bin"}


def _download(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    started = time.monotonic()
    logger.warning("downloading %s -> %s", url, dest)
    request = urllib.request.Request(url, headers={"User-Agent": "hovsg-ragmap-adapter"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out, length=16 << 20)
    part.replace(dest)
    logger.warning("downloaded %s (%.1f MB) in %.0fs", dest.name, dest.stat().st_size / 1e6, time.monotonic() - started)


def ensure_checkpoint(path: str | os.PathLike) -> Path:
    """Return ``path``, downloading it first if missing and its name is known."""
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return path
    url = CHECKPOINTS.get(path.name)
    if url is None:
        raise FileNotFoundError(f"checkpoint {path} does not exist and has no known download URL")
    if os.environ.get("HOVSG_OFFLINE") == "1":
        raise FileNotFoundError(f"checkpoint {path} missing and HOVSG_OFFLINE=1")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / f".{path.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (path.is_file() and path.stat().st_size > 0):
            _download(url, path)
    return path
