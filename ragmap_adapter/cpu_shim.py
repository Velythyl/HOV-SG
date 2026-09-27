"""CPU fallback for HOV-SG's hard-coded ``.cuda()`` calls.

Upstream HOV-SG assumes a GPU: ``clip_utils`` and ``sam_clip_feats_extractor``
call ``tensor.cuda()`` and ``torch.zeros(..., device="cuda")`` unconditionally,
while ``Graph`` itself picks ``cuda`` only if available. This shim is installed
**only when no CUDA device is visible** (the CI smoke test on a CPU runner) and
redirects those calls to the CPU. With a GPU it is never installed, so
production runs execute the upstream code unchanged.
"""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)
_installed = False


def _cpu_device(device):
    if device is None:
        return None
    if isinstance(device, torch.device):
        return torch.device("cpu") if device.type == "cuda" else device
    if isinstance(device, str) and device.startswith("cuda"):
        return "cpu"
    if isinstance(device, int):
        return "cpu"
    return device


def _wrap_factory(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _cpu_device(kwargs["device"])
        return fn(*args, **kwargs)

    return wrapper


def install_if_no_cuda() -> bool:
    global _installed
    if _installed or torch.cuda.is_available():
        return _installed
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[assignment]
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self  # type: ignore[assignment]
    for name in ("zeros", "ones", "empty", "full", "arange", "tensor", "rand", "randn", "zeros_like", "ones_like"):
        setattr(torch, name, _wrap_factory(getattr(torch, name)))
    _original_to = torch.Tensor.to

    def _to(self, *args, **kwargs):
        args = tuple(_cpu_device(a) if isinstance(a, (str, torch.device)) else a for a in args)
        if "device" in kwargs:
            kwargs["device"] = _cpu_device(kwargs["device"])
        return _original_to(self, *args, **kwargs)

    torch.Tensor.to = _to  # type: ignore[assignment]
    _installed = True
    logger.warning("no CUDA device visible: redirecting HOV-SG's hard-coded .cuda() calls to CPU (smoke/CI mode)")
    return True
