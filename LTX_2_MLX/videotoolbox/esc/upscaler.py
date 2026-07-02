"""Per-frame driver for the MLX ESC-Real upscaler (single-image, no temporal state)."""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..upscaler_base import to_rgb_batch

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class EscUpscaler:
    """feed()/flush() driver for the per-frame ESC-Real upscaler."""

    def __init__(self, weights: Any = None, dtype: Any = mx.float16, compile: bool = True):
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self.scale = self._cfg[6]
        self._fwd = net.make_forward(self._p, self._cfg, compile=compile)

    def reset(self) -> None:
        pass

    def feed(self, rgb: Any, token: Any = None) -> list:
        sr = self._fwd(to_rgb_batch(rgb))
        mx.eval(sr)
        return [(sr[0], token)]

    def flush(self) -> list:
        return []
