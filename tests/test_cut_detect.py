"""Regression net for videotoolbox cut_detect (numpy -> MLX rewrite).

Pure-CPU scene-cut detection, so this runs everywhere (no pyobjc). Pins the
histogram, thumbnail, and is_cut decision sequence on deterministic frames so the
rewrite is behavior-identical; frames are given as both numpy and mlx.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.videotoolbox.cut_detect import (
    CutDetector,
    _frame_histogram,
    _frame_thumbnail,
)


def _sha(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:16]


def _as(container, arr):
    return mx.array(arr) if container == "mlx" else arr


def _gradient_rgb():
    return (np.arange(64 * 64 * 3) % 256).astype(np.uint8).reshape(64, 64, 3)


def _gradient_rgba_f16():
    return (np.arange(64 * 64 * 4, dtype=np.float32) / (64 * 64 * 4)).astype(
        np.float16
    ).reshape(64, 64, 4)


@pytest.mark.parametrize("container", ["numpy", "mlx"])
def test_frame_histogram(container):
    assert _sha(_frame_histogram(_as(container, _gradient_rgb()))) == "6a7ae0917b8ab179"
    assert _sha(_frame_histogram(_as(container, _gradient_rgba_f16()))) == "efa6accfaadad0a3"


@pytest.mark.parametrize("container", ["numpy", "mlx"])
def test_frame_thumbnail(container):
    thumb = _frame_thumbnail(_as(container, _gradient_rgb()))
    assert tuple(int(x) for x in thumb.shape) == (32, 32, 3)
    assert _sha(thumb) == "17a1da5ba991f760"


@pytest.mark.parametrize("container", ["numpy", "mlx"])
@pytest.mark.parametrize("mode,threshold", [("simple", 0.2), ("hist", 0.5)])
def test_cut_detection_sequence(container, mode, threshold):
    fb = np.zeros((64, 64, 3), np.uint8)
    fw = np.full((64, 64, 3), 255, np.uint8)
    det = CutDetector(mode, threshold)
    # black, black (no change), white (hard cut), white (no change)
    seq = [det.is_cut(_as(container, f)) for f in (fb, fb, fw, fw)]
    assert seq == [False, False, True, False]
