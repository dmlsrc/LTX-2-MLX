"""Reference-free unit tests for the MLX spatial upscaler.

These replace the old PyTorch-parity scripts (test_spatial_upscaler_parity.py
and test_upscaler_full_parity.py), which compared against the `ltx_core`
PyTorch reference. That reference does not import/run on macOS without
finagling, so those tests only ever errored here. `PixelShuffle2d` is the one
piece whose correctness can be pinned without the reference: it is validated
against the einops pixel-shuffle pattern, an independent implementation of the
same `(c p1 p2) -> (h p1) (w p2)` rearrangement.
"""

import mlx.core as mx
import numpy as np
from einops import rearrange

from LTX_2_MLX.model.upscaler.spatial import PixelShuffle2d


def test_pixel_shuffle_matches_einops():
    """PixelShuffle2d (NHWC) realizes the same rearrangement as einops
    'b (c p1 p2) h w -> b c (h p1) (w p2)'. Pure data movement, so the values
    must match exactly."""
    r, c_out, b, h, w = 2, 4, 1, 4, 5
    rng = np.random.default_rng(42)
    x_nchw = rng.standard_normal((b, c_out * r * r, h, w)).astype(np.float32)

    # Independent reference: einops on the NCHW layout (numpy-native, no torch).
    ref_nchw = rearrange(x_nchw, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=r, p2=r)

    # PixelShuffle2d works in NHWC; convert in and back out for comparison.
    x_nhwc = mx.array(x_nchw.transpose(0, 2, 3, 1))
    out_nchw = np.array(PixelShuffle2d(upscale_factor=r)(x_nhwc)).transpose(0, 3, 1, 2)

    assert out_nchw.shape == (b, c_out, h * r, w * r)
    assert np.array_equal(out_nchw, ref_nchw), (
        f"max abs diff {np.abs(out_nchw - ref_nchw).max()}"
    )


def test_pixel_shuffle_subpixel_mapping():
    """Channel (c, p1, p2) lands at output spatial offset (p1, p2): an arange
    input over the 4 sub-pixel channels fills a 2x2 block in row-major order."""
    r = 2
    x_nchw = np.arange(r * r, dtype=np.float32).reshape(1, r * r, 1, 1)
    x_nhwc = mx.array(x_nchw.transpose(0, 2, 3, 1))
    out_nchw = np.array(PixelShuffle2d(upscale_factor=r)(x_nhwc)).transpose(0, 3, 1, 2)

    assert out_nchw.shape == (1, 1, r, r)
    assert out_nchw[0, 0].tolist() == [[0.0, 1.0], [2.0, 3.0]]
