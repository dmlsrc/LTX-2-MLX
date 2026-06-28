"""Windowed streaming wrapper for the MLX RealBasicVSR net."""
from __future__ import annotations

from typing import Any

import mlx.core as mx

try:
    from . import net
except ImportError:   # running directly as a script
    import net

SCALE = 4


class RealBasicVsrUpscaler:
    """Windowed feed()/flush() driver for RealBasicVSR.

    RealBasicVSR is bidirectional over a clip, so the harness feeds it sliding
    windows and emits the stable interior frames. This mirrors BasicVSR++'s
    wrapper while exposing the cleaning loop threshold used by RealBasicVSR.
    """

    def __init__(
        self,
        weights: Any = None,
        window: int = 14,
        trim: int = 0,
        dynamic_refine_thres: float = 5.0,
        clean_iters: int = 3,
        residual_strength: float = 1.0,
        flow_consistency: float = 0.0,
    ):
        self._p = net.load_params(weights)
        self._T = int(trim)
        self._W = int(window)
        if self._T < 0:
            raise ValueError(f"RealBasicVSR trim must be >= 0; got {self._T}")
        if self._W < 1:
            raise ValueError(f"RealBasicVSR window must be >= 1; got {self._W}")
        if self._T and self._W <= 2 * self._T:
            raise ValueError(
                "RealBasicVSR window must be greater than 2*trim so each "
                f"window can emit interior frames; got window={self._W}, trim={self._T}. "
                "Use trim=0 for reference-like non-overlapping chunks."
            )
        self._dynamic_refine_thres = float(dynamic_refine_thres)
        self._clean_iters = int(clean_iters)
        self._residual_strength = float(residual_strength)
        self._flow_consistency = float(flow_consistency)
        self.reset()

    def reset(self) -> None:
        self._frames: list = []
        self._tokens: list = []
        self._base = 0
        self._emitted = 0

    @staticmethod
    def _batch(rgb: Any) -> Any:
        a = rgb if rgb.ndim == 4 else rgb[None]
        return a[..., :3].astype(mx.float32)

    def feed(self, rgb: Any, token: Any = None) -> list:
        self._frames.append(self._batch(rgb))
        self._tokens.append(token)
        out: list = []
        while (self._base + len(self._frames)) >= max(0, self._emitted - self._T) + self._W:
            ws = max(0, self._emitted - self._T)
            out.extend(self._run(ws, ws + self._W, last=False))
        total = self._base + len(self._frames)
        keep = max(0, min(self._emitted - self._T, total - self._W)) - self._base
        if keep > 0:
            self._frames = self._frames[keep:]
            self._tokens = self._tokens[keep:]
            self._base += keep
        return out

    def flush(self) -> list:
        total = self._base + len(self._frames)
        if self._emitted >= total:
            self.reset()
            return []
        ws = max(0, min(self._emitted - self._T, total - self._W))
        out = self._run(ws, total, last=True)
        self.reset()
        return out

    def _run(self, ws: int, we: int, last: bool) -> list:
        sr = net.upscale(
            self._frames[ws - self._base:we - self._base],
            self._p,
            dynamic_refine_thres=self._dynamic_refine_thres,
            clean_iters=self._clean_iters,
            residual_strength=self._residual_strength,
            flow_consistency=self._flow_consistency,
        )
        mx.eval(*sr)
        end = we if last else we - self._T
        out = [(sr[g - ws][0], self._tokens[g - self._base]) for g in range(self._emitted, end)]
        self._emitted = end
        return out
