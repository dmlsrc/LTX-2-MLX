"""Per-frame driver for the MLX RRDBNet upscaler.

Real-ESRGAN / ESRGAN are single-image networks, so unlike the BasicVSR wrappers
there is no sliding window, trim, or recurrent state -- each frame is upscaled
independently and emitted immediately. The feed()/flush() shape mirrors the other
upscalers so the harness wiring stays parallel.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..upscaler_base import to_rgb_batch

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class RealEsrganUpscaler:
    """feed()/flush() driver for the per-frame RRDBNet upscaler."""

    def __init__(self, weights: Any = None, denoise_strength: float = 1.0):
        weights = net.resolve_weights(weights)   # variant token or path -> file
        wdn = net.wdn_path_for(weights) if float(denoise_strength) < 1.0 else None
        self._p = net.load_params(weights, wdn_path=wdn,
                                  denoise_strength=denoise_strength)
        self.scale = net.scale_of(self._p)
        self.reset()

    def reset(self) -> None:
        pass

    def feed(self, rgb: Any, token: Any = None) -> list:
        sr = net.upscale([to_rgb_batch(rgb)], self._p)[0]
        mx.eval(sr)
        return [(sr[0], token)]

    def flush(self) -> list:
        return []
