"""Per-frame driver for the MLX SAFMN upscaler.

Both SAFMN variants are single-image networks (no temporal state), so each frame
upscales independently and is emitted immediately; the feed()/flush() shape mirrors
the other upscalers so the harness wiring stays parallel.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..upscaler_base import to_rgb_batch

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class SafmnUpscaler:
    """feed()/flush() driver for the per-frame SAFMN upscaler."""

    def __init__(self, weights: Any = None, dtype: Any = mx.float16, compile: bool = True):
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self.scale = self._cfg[3]
        self._fwd = net.make_forward(self._p, self._cfg, compile=compile)

    def reset(self) -> None:
        pass

    def feed(self, rgb: Any, token: Any = None) -> list:
        sr = self._fwd(to_rgb_batch(rgb))
        mx.eval(sr)
        return [(sr[0], token)]

    def flush(self) -> list:
        return []
