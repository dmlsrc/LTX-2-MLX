"""Tests for native image I/O (LTX_2_MLX.videotoolbox.images).

Covers the ImageIO / CoreImage / AppKit replacements for Pillow: decode, encode,
Lanczos resize, and text annotation. No Pillow.
"""

import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from LTX_2_MLX.videotoolbox import images as I


def _gradient(h=48, w=64):
    """(h, w, 3) uint8 gradient: red ramps over x, green over y, blue flat."""
    xr = mx.broadcast_to((mx.arange(w, dtype=mx.float32) * (255 / w))[None, :], (h, w))
    yg = mx.broadcast_to((mx.arange(h, dtype=mx.float32) * (255 / h))[:, None], (h, w))
    return mx.stack([xr, yg, mx.full((h, w), 100.0)], axis=2).astype(mx.uint8)


def test_save_load_png_byte_exact():
    src = _gradient()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rt.png"
        assert I.save_image(src, p) == p and p.exists()
        rt = I.load_image_rgb(p)
        assert rt.shape == src.shape
        assert int(mx.max(mx.abs(rt.astype(mx.int32) - src.astype(mx.int32))).item()) == 0


def test_load_image_rgb_orientation_top_first():
    # Top quarter white, rest black; row 0 must come back as the white top.
    h, w = 40, 16
    src = mx.concatenate(
        [mx.full((h // 4, w, 3), 255, dtype=mx.uint8), mx.zeros((h - h // 4, w, 3), dtype=mx.uint8)],
        axis=0,
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "orient.png"
        I.save_image(src, p)
        rt = I.load_image_rgb(p)
        assert int(rt[0, 0, 0].item()) > 240
        assert int(rt[h - 1, 0, 0].item()) < 15


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        I.load_image_rgb("/nonexistent/path/nope.png")


def test_resize_lanczos_exact_shape():
    src = _gradient(48, 64)
    down = I.resize_lanczos(src, 32, 24)
    up = I.resize_lanczos(src, 100, 60)
    assert down.shape == (24, 32, 3)
    assert up.shape == (60, 100, 3)
    assert int(mx.max(down).item()) > 0 and int(mx.max(up).item()) > 0


def test_resize_lanczos_same_size_noop():
    src = _gradient(40, 50)
    out = I.resize_lanczos(src, 50, 40)
    assert out.shape == (40, 50, 3)
    assert int(mx.max(mx.abs(out.astype(mx.int32) - src.astype(mx.int32))).item()) == 0


def test_resize_lanczos_anamorphic_dims():
    # Independent x/y scaling (aspect change), matching PIL resize((w, h)).
    out = I.resize_lanczos(_gradient(40, 40), 80, 20)
    assert out.shape == (20, 80, 3)


def test_grayscale_png_decodes_to_equal_rgb():
    from LTX_2_MLX.videotoolbox._compat import Foundation, Quartz

    val, w, h = 137, 8, 6
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gray.png"
        buf = bytearray([val] * (w * h))
        cs = Quartz.CGColorSpaceCreateDeviceGray()
        ctx = Quartz.CGBitmapContextCreate(buf, w, h, 8, w, cs, Quartz.kCGImageAlphaNone)
        cg = Quartz.CGBitmapContextCreateImage(ctx)
        url = Foundation.NSURL.fileURLWithPath_(str(p))
        dest = Quartz.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
        Quartz.CGImageDestinationAddImage(dest, cg, None)
        Quartz.CGImageDestinationFinalize(dest)
        g = I.load_image_rgb(p)
        assert g.shape == (h, w, 3)
        eq = int(mx.max(mx.abs(g[:, :, 0].astype(mx.int32) - g[:, :, 2].astype(mx.int32))).item())
        assert eq == 0
        assert abs(int(g[0, 0, 0].item()) - val) <= 2


def test_save_image_accepts_numpy_input():
    np = pytest.importorskip("numpy")
    arr = np.zeros((10, 12, 3), dtype=np.uint8)
    arr[:5] = (200, 100, 50)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "np.png"
        I.save_image(arr, p)
        rt = I.load_image_rgb(p)
        assert rt.shape == (10, 12, 3)
        assert int(rt[0, 0, 0].item()) == 200
        assert int(rt[9, 0, 0].item()) == 0


def test_draw_labels_renders_top_anchored_upright():
    ann = I.draw_labels(mx.zeros((44, 80, 3), dtype=mx.uint8), [(6, 3, "F")], font_size=30)
    assert ann.shape == (44, 80, 3)
    lum = mx.max(ann.astype(mx.float32), axis=2)
    rows = [r for r in range(44) if float(mx.sum(lum[r]).item()) > 30]
    assert rows, "no text rendered"
    assert min(rows) < 14  # anchored near the top (y=3), not the bottom
    lo, hi = min(rows), max(rows)
    mid = (lo + hi) // 2
    top_ink = float(mx.sum(lum[lo:mid + 1]).item())
    bot_ink = float(mx.sum(lum[mid + 1:hi + 1]).item())
    assert top_ink > bot_ink  # 'F' is top-heavy when upright
