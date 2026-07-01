"""Per-frame NAFNet restorer, RGB in / RGB out.

NAFNet (deblur / denoise / video variants) is a single-image residual net, so this is a
stateless per-frame stage -- restore each frame independently. Pair it after deblock +
denoise as a light detail/deblur pass. `strength` scales the predicted residual
(1.0 = full, <1 = light); since it is single-image, a strong strength can flicker on
video, so keep it light there. Exposes .denoise() as the harness per-frame contract
(same as FbcnnDeblocker) -- the name is the interface, not a claim about the task.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import net


class NafnetRestorer:
    """Stateless per-frame RGB NAFNet restorer (deblur / denoise / video variant)."""

    def __init__(self, weights: Any = None, strength: float = 1.0,
                 compile: bool = True, dtype: Any = mx.float32):   # fp16 overflows -- see net.load_params
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self._strength = float(strength)
        self._fwd = net.make_forward(self._p, self._strength, self._cfg, compile=compile)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def denoise(self, rgb_f32: Any) -> Any:
        """Restore one RGB frame (H,W,3) in [0,1]; returns (H,W,3)."""
        a = rgb_f32 if rgb_f32.ndim == 4 else rgb_f32[None]
        out = mx.clip(self._fwd(a[..., :3]), 0.0, 1.0)
        mx.eval(out)
        return out[0]
