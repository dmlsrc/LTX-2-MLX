"""Windowed streaming wrapper for the MLX RealBasicVSR net.

RealBasicVSR is the real-world sibling of BasicVSR++: an iterative image-cleaning
front-end followed by the first-order BasicVSR recurrent core, bidirectional over
a clip. The sliding-window feed()/flush() machinery lives in ../upscaler_base;
this wrapper loads the weights, exposes the cleaning-loop knobs, and upscales each
window.
"""
from __future__ import annotations

from typing import Any

from ..upscaler_base import WindowedUpscaler

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class RealBasicVsrUpscaler(WindowedUpscaler):
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
        self._p = net.load_params(net.resolve_weights(weights))
        t, w = int(trim), int(window)
        if t < 0:
            raise ValueError(f"RealBasicVSR trim must be >= 0; got {t}")
        if w < 1:
            raise ValueError(f"RealBasicVSR window must be >= 1; got {w}")
        if t and w <= 2 * t:
            raise ValueError(
                "RealBasicVSR window must be greater than 2*trim so each "
                f"window can emit interior frames; got window={w}, trim={t}. "
                "Use trim=0 for reference-like non-overlapping chunks."
            )
        self._dynamic_refine_thres = float(dynamic_refine_thres)
        self._clean_iters = int(clean_iters)
        self._residual_strength = float(residual_strength)
        self._flow_consistency = float(flow_consistency)
        super().__init__(window=w, trim=t)

    def _upscale_window(self, frames: list) -> list:
        return net.upscale(
            frames,
            self._p,
            dynamic_refine_thres=self._dynamic_refine_thres,
            clean_iters=self._clean_iters,
            residual_strength=self._residual_strength,
            flow_consistency=self._flow_consistency,
        )
