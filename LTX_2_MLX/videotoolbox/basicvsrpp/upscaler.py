"""Streaming windowed wrapper around the recurrent BasicVSR++ net so it fits a
frame-at-a-time pipeline. feed()/flush() mirror the FastDVDnet delay-line, but
every emitted frame is 4x upscaled.

BasicVSR++ is bidirectional + second-order recurrent over the whole clip, so a
single frame can't be upscaled in isolation. We slide a window of `window`
frames, emit its interior, and trim `trim` warm-up frames at each window join
(the propagation's transient edge). Interior frames match the full-clip result
to ~50 dB; only the clip ends use one-sided context, exactly as the reference.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

try:
    from . import net
except ImportError:   # running directly as a script
    import net

SCALE = 4


class BasicVsrUpscaler:
    """Windowed feed()/flush() driver for net.upscale. Each call to feed(rgb)
    buffers a frame and returns a list of (upscaled_rgb_4x, token) tuples once
    enough lookahead has arrived; flush() drains the tail. Memory is bounded to
    ~`window` buffered LR frames regardless of clip length."""

    def __init__(self, weights: Any = None, window: int = 14, trim: int = 2):
        self._p = net.load_params(weights)
        self._T = int(trim)
        self._W = max(int(window), 2 * self._T + 1)
        self.reset()

    def reset(self) -> None:
        self._frames: list = []     # sliding LR buffer (1,H,W,3)
        self._tokens: list = []
        self._base = 0              # global index of _frames[0]
        self._emitted = 0          # global index of the next frame to emit

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
        # Retain enough for the next interior window (back to emitted-T) AND a
        # full-width flush window (back to total-W); drop only what neither needs.
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
        sr = net.upscale(self._frames[ws - self._base:we - self._base], self._p)
        mx.eval(*sr)
        end = we if last else we - self._T
        out = [(sr[g - ws][0], self._tokens[g - self._base]) for g in range(self._emitted, end)]
        self._emitted = end
        return out


if __name__ == "__main__":
    import math

    def psnr(a, b):
        mse = float(mx.mean((a - b) ** 2))
        return 99.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)

    up = BasicVsrUpscaler(window=14, trim=2)
    N = 18
    mx.random.seed(0)
    frames = [mx.random.uniform(shape=(1, 40, 56, 3)) for _ in range(N)]
    mx.eval(*frames)
    full = net.upscale(frames, up._p)     # full-clip reference
    mx.eval(*full)

    emitted: list = []
    for i, f in enumerate(frames):
        emitted.extend(up.feed(f[0], token=i))
    emitted.extend(up.flush())
    toks = [t for _, t in emitted]
    print(f"emitted {len(emitted)}/{N} frames, order ok: {toks == list(range(N))}")
    for idx in (3, N // 2, N - 4):
        print(f"  interior frame {idx}: windowed-vs-fullclip PSNR = "
              f"{psnr(emitted[idx][0], full[idx][0]):.1f} dB")
