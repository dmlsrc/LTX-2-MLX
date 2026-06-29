"""Streaming windowed wrapper around the recurrent BasicVSR++ net so it fits a
frame-at-a-time pipeline. feed()/flush() mirror the FastDVDnet delay-line, but
every emitted frame is 4x upscaled.

BasicVSR++ is bidirectional + second-order recurrent over the whole clip, so a
single frame can't be upscaled in isolation. We slide a window of `window`
frames, emit its interior, and trim `trim` warm-up frames at each window join
(the propagation's transient edge). Interior frames match the full-clip result
to ~50 dB; only the clip ends use one-sided context, exactly as the reference.

The sliding-window feed()/flush() machinery lives in ../upscaler_base; this
wrapper only loads the BasicVSR++ weights and upscales each window.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..upscaler_base import WindowedUpscaler

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class BasicVsrUpscaler(WindowedUpscaler):
    """Windowed feed()/flush() driver for net.upscale. Each call to feed(rgb)
    buffers a frame and returns a list of (upscaled_rgb_4x, token) tuples once
    enough lookahead has arrived; flush() drains the tail. Memory is bounded to
    ~`window` buffered LR frames regardless of clip length."""

    def __init__(self, weights: Any = None, window: int = 14, trim: int = 2):
        self._p = net.load_params(weights)
        # Window must span both trim edges plus >=1 interior frame to emit.
        super().__init__(window=max(int(window), 2 * int(trim) + 1), trim=trim)

    def _upscale_window(self, frames: list) -> list:
        return net.upscale(frames, self._p)


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
