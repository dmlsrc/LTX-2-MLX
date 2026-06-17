"""Regression net for the streaming VAE-decode -> per-frame converters.

Guards chunk_to_uint8_frames / chunk_to_rgba_fp16_frames at the BYTE level so the
numpy -> MLX-native rewrite of the video-export path is provably output-identical.
The frame container may be np.ndarray (legacy) or mlx.core.array (native); the
assertions compare raw bytes + shape + dtype-leaf, so they hold across that switch.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx

from LTX_2_MLX.pipelines.streaming import (
    chunk_to_rgba_fp16_frames,
    chunk_to_uint8_frames,
)

_CHUNK_SHAPE = (1, 3, 2, 5, 4)  # (B, C, T, H, W)

# (shape, dtype-leaf, sha256[:24]) per frame, captured from the numpy implementation.
_UINT8_GOLDEN = [
    ((5, 4, 3), "uint8", "d69055956d48cd6b784cce9c"),
    ((5, 4, 3), "uint8", "1180661b8f8c95b264bf7c3d"),
]
_RGBA16_GOLDEN = [
    ((5, 4, 4), "float16", "de493fae3c690e6d326a89c6"),
    ((5, 4, 4), "float16", "2746150b274d87ed3c632357"),
]


def _make_chunk() -> mx.array:
    b, c, t, h, w = _CHUNK_SHAPE
    n = b * c * t * h * w
    chunk = mx.arange(n, dtype=mx.float32).reshape(b, c, t, h, w) / (n - 1) * 2 - 1
    return chunk.astype(mx.bfloat16)


def _frame_bytes(frame) -> bytes:
    """Raw bytes of a frame, whether it is a numpy array or an mlx array."""
    tobytes = getattr(frame, "tobytes", None)
    if callable(tobytes):
        return tobytes()  # numpy
    return bytes(memoryview(mx.contiguous(frame)))  # mlx


def _describe(frames):
    out = []
    for frame in frames:
        dtype_leaf = str(frame.dtype).split(".")[-1]  # 'uint8'/'float16' for np AND mlx
        out.append((
            tuple(int(x) for x in frame.shape),
            dtype_leaf,
            hashlib.sha256(_frame_bytes(frame)).hexdigest()[:24],
        ))
    return out


def test_chunk_to_uint8_frames_byte_identical():
    assert _describe(chunk_to_uint8_frames(_make_chunk())) == _UINT8_GOLDEN


def test_chunk_to_rgba_fp16_frames_byte_identical():
    assert _describe(chunk_to_rgba_fp16_frames(_make_chunk())) == _RGBA16_GOLDEN
