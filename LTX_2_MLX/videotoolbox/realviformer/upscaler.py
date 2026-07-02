"""Streaming driver for the MLX RealViformer upscaler.

RealViformer's recurrence is CAUSAL (forward-only), so unlike the bidirectional
VSR nets there is no window/trim buffering: each frame upscales as it arrives,
carrying the propagated features (and the previous frame, for the flow) across
calls. reset() drops the temporal state -- the harness calls it at hard cuts.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..upscaler_base import to_rgb_batch
from ..vsr_blocks import compiled_spynet_flow, flow_warp

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class RealViformerUpscaler:
    """Streaming feed()/flush() driver for the causal RealViformer."""

    def __init__(self, weights: Any = None, dtype: Any = mx.float16, compile: bool = True):
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self.scale = 4
        self._first, self._next = net.make_steps(self._p, self._cfg, compile=compile)
        self.reset()

    def reset(self) -> None:
        self._prev: Any = None       # previous input frame (for the flow)
        self._feat: Any = None       # propagated features

    def close(self) -> None:
        pass

    @staticmethod
    def _pad4(x: Any) -> Any:
        """Replicate-pad bottom/right to multiples of 4 (the U-Net has two 2x downs)."""
        _, h, w, _ = x.shape
        ph, pw = (-h) % 4, (-w) % 4
        if ph:
            x = mx.concatenate(
                [x, mx.broadcast_to(x[:, h - 1:h], (x.shape[0], ph, x.shape[2], x.shape[3]))], axis=1)
        if pw:
            x = mx.concatenate(
                [x, mx.broadcast_to(x[:, :, w - 1:w], (x.shape[0], x.shape[1], pw, x.shape[3]))], axis=2)
        return x

    def feed(self, rgb: Any, token: Any = None) -> list:
        x = to_rgb_batch(rgb)
        h, w = x.shape[1], x.shape[2]
        xp = self._pad4(x)
        if self._prev is None:
            sr, feat = self._first(xp)
        else:
            dt = self._feat.dtype
            flow = compiled_spynet_flow(self._p, xp.astype(dt), self._prev.astype(dt))
            warped = flow_warp(self._feat, flow, "zeros")
            sr, feat = self._next(xp, warped)
        sr = sr[:, :h * 4, :w * 4, :]
        mx.eval(sr, feat)
        self._prev, self._feat = xp, feat
        return [(sr[0], token)]

    def flush(self) -> list:
        self.reset()
        return []
