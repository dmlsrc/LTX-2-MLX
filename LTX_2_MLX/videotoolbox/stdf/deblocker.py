"""Streaming STDF deblocker: a centered (2*radius+1)-frame luma window in, the
deblocked center frame out, as an RGB-in / RGB-out delay line.

STDF is luma-only, so each RGB frame is split into BT.601 luma + chroma; the net
deblocks the luma using its temporal window and the original chroma is carried
through unchanged (compression chroma artifacts are far less visible, and the net
was trained on Y). feed(rgb, token) buffers a frame and returns the (deblocked_rgb,
token) pairs now ready; flush() drains the tail. Clip ends use a reflected window,
exactly as the reference. Pair this before the scaler so compression blocking is
removed before any GAN SR amplifies it.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import net

# BT.601 full-range luma; the two chroma-difference scales are 2*(1-Kb), 2*(1-Kr).
_KR, _KG, _KB = 0.299, 0.587, 0.114
_CB, _CR = 1.772, 1.402


class StdfDeblocker:
    """RGB-in / RGB-out streaming compressed-video deblocker (luma-only STDF)."""

    def __init__(self, weights: Any = None, strength: float = 1.0, dtype: Any = mx.float16):
        self._p = net.load_params(weights, dtype=dtype)
        self._strength = float(strength)
        in_nc, self._ilen, _ = net._config(self._p)
        if in_nc != 1:
            raise ValueError(f"StdfDeblocker expects a Y-only STDF checkpoint (in_nc=1), got {in_nc}")
        self._radius = self._ilen // 2
        self._reset()

    def _reset(self) -> None:
        self._buf: list = []     # (luma (1,H,W,1), rgb (1,H,W,3), token) for abs idx [_base..]
        self._base = 0
        self._received = 0
        self._emitted = 0

    def reset(self) -> None:
        self._reset()

    def close(self) -> None:
        pass

    @staticmethod
    def _split(rgb: Any) -> tuple:
        a = (rgb if rgb.ndim == 4 else rgb[None])[..., :3].astype(mx.float32)
        y = _KR * a[..., 0:1] + _KG * a[..., 1:2] + _KB * a[..., 2:3]
        return y, a

    @staticmethod
    def _recombine(rgb: Any, new_y: Any) -> Any:
        """Swap the luma of `rgb` (1,H,W,3) for `new_y` (1,H,W,1), keeping its chroma."""
        r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        yo = _KR * r + _KG * g + _KB * b
        cb, cr = (b - yo) / _CB, (r - yo) / _CR     # original chroma, preserved
        ny = new_y.astype(mx.float32)
        nr = ny + _CR * cr
        nb = ny + _CB * cb
        ng = (ny - _KR * nr - _KB * nb) / _KG
        return mx.clip(mx.concatenate([nr, ng, nb], axis=-1), 0.0, 1.0)

    @staticmethod
    def _reflect(i: int, last: int) -> int:
        if i < 0:
            i = -i
        if i > last:
            i = 2 * last - i
        return max(0, min(last, i))

    def _luma(self, i: int, last: int) -> Any:
        return self._buf[self._reflect(i, last) - self._base][0]

    def _emit_one(self, last: int) -> tuple:
        t = self._emitted
        window = [self._luma(t + d, last) for d in range(-self._radius, self._radius + 1)]
        dy = net.deblock(window, self._p, self._strength)       # deblocked center luma
        _, rgb, tok = self._buf[t - self._base]
        out = self._recombine(rgb, dy)
        mx.eval(out)
        self._emitted += 1
        keep = self._emitted - self._radius                    # oldest index still needed
        while self._base < keep and self._buf:
            self._buf.pop(0)
            self._base += 1
        return out[0], tok

    def feed(self, rgb: Any, token: Any = None) -> list:
        y, full = self._split(rgb)
        self._buf.append((y, full, token))
        self._received += 1
        last = self._received - 1
        ready = []
        while last - self._emitted >= self._radius:
            ready.append(self._emit_one(last))
        return ready

    def flush(self) -> list:
        last = self._received - 1
        out = []
        while self._emitted <= last:
            out.append(self._emit_one(last))
        self._reset()
        return out
