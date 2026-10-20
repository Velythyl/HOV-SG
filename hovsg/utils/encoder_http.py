"""Client for RAGMAP's encoder service (``ragmap-encoder-service``), for ``models.clip.type=http``.

The service owns the visual-text encoder: it applies the model's own preprocessing
(and, for the LLM-based embedders, their fixed instructions) and returns
L2-normalised float32 embeddings. HOV-SG keeps everything that is method logic:
which images it embeds (full frames, masked and unmasked 512x512 mask crops,
room views), the masked/unmasked fusion and the text templates. Only the
``open_clip`` forward pass and its eval transform are replaced.

Wire format (mirrors ``ragmap.encoders.wire``; this image cannot import ragmap):

* ``GET /healthz`` -> ``{status, profile, model, revision, dim, normalized}``;
* ``POST /v1/embed/images`` ``{"images": [base64 PNG, RGB uint8]}`` and
  ``POST /v1/embed/texts`` ``{"texts": [...]}`` -> ``{profile, revision, dim,
  count, dtype: "float32", embeddings: base64 of little-endian float32 (n, D)}``.

Every answer is checked against the identity ``/healthz`` gave at start-up, and
any error status raises: an embedding is never replaced by zeros.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class EncoderServiceError(RuntimeError):
    pass


def _encode_png(image: np.ndarray) -> str:
    array = np.uint8(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {array.shape}")
    buffer = io.BytesIO()
    # Lossless either way; level 1 is several times faster to write than the default 6.
    Image.fromarray(array, "RGB").save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class HttpEncoder:
    """Stands in for the open_clip model object (``clip_model``) in ``hovsg.utils.clip_utils``."""

    def __init__(
        self,
        base_url: str,
        profile: Optional[str] = None,
        revision: Optional[str] = None,
        timeout: float = 900.0,
        image_batch: int = 64,
        text_batch: int = 512,
        wait_s: float = 900.0,
        label_cache_dir: Optional[str] = None,
    ):
        if not base_url:
            raise EncoderServiceError(
                "models.clip.type=http needs the encoder service URL "
                "(models.clip.url or env RAGMAP_ENCODER_BASE_URL)"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.image_batch = int(image_batch)
        self.text_batch = int(text_batch)
        self.label_cache_dir = label_cache_dir
        self.identity = self._wait_healthy(float(wait_s))
        if profile and self.identity.get("profile") != profile:
            raise EncoderServiceError(
                f"{self.base_url} serves profile {self.identity.get('profile')!r}, expected {profile!r}"
            )
        if revision and self.identity.get("revision") != revision:
            raise EncoderServiceError(
                f"{self.base_url} serves revision {self.identity.get('revision')!r}, expected {revision!r}"
            )
        if not self.identity.get("normalized", True):
            raise EncoderServiceError(f"{self.base_url} does not return L2-normalised embeddings: {self.identity}")
        self.profile = str(self.identity["profile"])
        self.revision = str(self.identity["revision"])
        self.dim = int(self.identity["dim"])
        logger.info("encoder service %s: %s", self.base_url, self.identity)

    # --- open_clip model API that HOV-SG touches ----------------------------
    def eval(self):
        return self

    # --- transport ---------------------------------------------------------------
    def _call(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            hint = " (GPU out of memory in the service)" if exc.code == 507 else ""
            raise EncoderServiceError(f"{self.base_url}{path}: HTTP {exc.code}{hint}: {detail}") from exc

    def _wait_healthy(self, wait_s: float) -> Dict[str, Any]:
        # The service loads its weights before it listens; give it time, then fail.
        deadline = time.monotonic() + wait_s
        while True:
            try:
                health = self._call("/healthz")
                if health.get("status") == "ok":
                    return health
                last = f"not ready: {health}"
            except (urllib.error.URLError, ConnectionError, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if time.monotonic() >= deadline:
                raise EncoderServiceError(f"encoder service {self.base_url} not healthy after {wait_s:.0f}s ({last})")
            time.sleep(5.0)

    def _embed(self, path: str, key: str, items: List[Any], batch: int) -> np.ndarray:
        parts = [np.zeros((0, self.dim), np.float32)]
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            answer = self._call(path, {key: chunk})
            for field in ("profile", "revision", "dim"):
                if answer.get(field) != self.identity[field]:
                    raise EncoderServiceError(
                        f"{field} changed under us: {answer.get(field)!r} != {self.identity[field]!r}"
                    )
            if answer.get("count") != len(chunk) or answer.get("dtype") != "float32":
                raise EncoderServiceError(
                    f"bad answer for {len(chunk)} {key}: count={answer.get('count')} dtype={answer.get('dtype')}"
                )
            array = np.frombuffer(base64.b64decode(answer["embeddings"]), dtype="<f4")
            if array.size != len(chunk) * self.dim:
                raise EncoderServiceError(f"embedding payload has {array.size} floats, expected {len(chunk)} x {self.dim}")
            if not np.all(np.isfinite(array)):
                raise EncoderServiceError(f"non-finite embedding values from {self.base_url}{path}")
            parts.append(array.reshape(len(chunk), self.dim).astype(np.float32))
        return np.concatenate(parts, axis=0)

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB arrays -> ``(n, D)`` float32, L2-normalised by the service."""
        return self._embed("/v1/embed/images", "images", [_encode_png(image) for image in images], self.image_batch)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("/v1/embed/texts", "texts", [str(text) for text in texts], self.text_batch)

    # --- per-encoder cache for precomputed label features ------------------------
    def label_feats_path(self, file_name: str, classes: Sequence[str]) -> Optional[Path]:
        """Where this encoder's text features for ``classes`` are cached, or None (no cache).

        Keyed by profile, revision and a hash of the class list, so the ViT-H
        file shipped with HOV-SG (``hovsg/labels/text_feats_*.npy``) is never read
        for another encoder, and a changed list is never served stale features.
        """
        if not self.label_cache_dir:
            return None
        digest = hashlib.sha256("\n".join(str(c) for c in classes).encode()).hexdigest()[:16]
        stem, ext = os.path.splitext(file_name)
        return Path(self.label_cache_dir) / self.profile / self.revision / f"{stem}-{digest}{ext}"


def is_http(clip_model: Any) -> bool:
    return isinstance(clip_model, HttpEncoder)
