"""Keep upstream's matplotlib calls working in a headless container.

``hovsg/graph/navigation_graph.py`` runs ``plt.switch_backend("TkAgg")`` at
import time. With no display (every container run) matplotlib refuses and the
import of ``hovsg.graph.graph`` fails. Upstream only uses matplotlib there to
save debug figures, so an interactive backend that cannot be loaded is replaced
by the non-interactive Agg backend. Call ``install()`` before importing hovsg.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def install() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if getattr(plt.switch_backend, "_ragmap_headless", False):
        return
    original = plt.switch_backend

    @functools.wraps(original)
    def switch_backend(newbackend):
        try:
            return original(newbackend)
        except ImportError as exc:
            logger.info("matplotlib backend %r unavailable (%s); staying on Agg", newbackend, exc)
            return original("Agg")

    switch_backend._ragmap_headless = True
    plt.switch_backend = switch_backend
