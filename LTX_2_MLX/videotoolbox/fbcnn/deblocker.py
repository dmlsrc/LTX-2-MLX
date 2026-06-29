"""Per-frame FBCNN JPEG-artifact-removal deblocker, RGB in / RGB out.

FBCNN is a single-image network (no temporal window), so this is a stateless per-frame
stage -- denoise(rgb) restores each frame independently. Pair it before the scaler to
strip JPEG / intra-block artifacts. Because it is single-image, BLIND mode
(quality=None) re-estimates the quality factor per frame, which can flicker on video as
the estimate drifts shot to shot; pass a fixed `quality` (a JPEG quality factor, lower =
stronger removal) for a temporally stable result. Unlike the temporal STDF deblocker it
does no noise averaging, so it is a pure deblocker, not a partial denoiser.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import net


class FbcnnDeblocker:
    """Stateless per-frame RGB JPEG-artifact deblocker (FBCNN color)."""

    def __init__(self, weights: Any = None, quality: Any = None, dtype: Any = mx.float16):
        self._p = net.load_params(weights, dtype=dtype)
        self._in_nc, self._nb = net._config(self._p)
        if self._in_nc != 3:
            raise ValueError(
                f"FbcnnDeblocker expects the RGB (color) FBCNN checkpoint (in_nc=3), got "
                f"{self._in_nc}; the grayscale variants would need a luma path not wired here.")
        # User-facing JPEG quality (1-100, lower = more compressed = stronger removal)
        # maps to the model's inverted-quality qf_input = 1 - quality/100; None = blind.
        self._quality = None if quality is None else float(quality)
        self._qf_input = None if quality is None else max(0.0, min(1.0, 1.0 - float(quality) / 100.0))

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def denoise(self, rgb_f32: Any) -> Any:
        """Restore one RGB frame (H,W,3) in [0,1]; returns (H,W,3)."""
        a = rgb_f32 if rgb_f32.ndim == 4 else rgb_f32[None]
        out, _qf = net.fbcnn(a[..., :3], self._p, qf_input=self._qf_input, nb=self._nb)
        out = mx.clip(out, 0.0, 1.0)
        mx.eval(out)
        return out[0]
