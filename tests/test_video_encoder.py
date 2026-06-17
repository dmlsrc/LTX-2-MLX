"""Regression net for the ffmpeg encode backend (video_encoder.py).

Pins the raw byte stream fed to ffmpeg (per frame, both bit depths) and the WAV
writers' output on deterministic synthetic input. The hashes were captured from
the numpy implementation, so they guard the numpy -> MLX-native rewrite: a
correct rewrite reproduces them exactly. A smoke test confirms the per-frame
streaming path still produces a valid file end to end.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess

import mlx.core as mx
import pytest

from LTX_2_MLX.video_encoder import (
    _frame_to_bytes,
    encode_video,
    write_wav_float32,
    write_wav_int16,
)


def _frames():
    return [
        (mx.arange(60, dtype=mx.int32) + k * 37).astype(mx.uint8).reshape(4, 5, 3)
        for k in range(3)
    ]


def _audio():
    return mx.arange(200, dtype=mx.float32).reshape(2, 100) / 199 * 2 - 1


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:24]


def test_frame_byte_stream_8bit():
    stream = b"".join(_frame_to_bytes(f, 8) for f in _frames())
    assert (_sha(stream), len(stream)) == ("ea4498f4c4f9dbd7a8b33f89", 180)


def test_frame_byte_stream_16bit():
    stream = b"".join(_frame_to_bytes(f, 16) for f in _frames())
    assert (_sha(stream), len(stream)) == ("79b71bf74558573206c5156e", 360)


def test_wav_int16_bytes(tmp_path):
    p = tmp_path / "a.wav"
    write_wav_int16(_audio(), p, 16000)
    data = p.read_bytes()
    assert (_sha(data), len(data)) == ("d7b5dbae0fc9d0b6fa815249", 444)


def test_wav_float32_bytes(tmp_path):
    p = tmp_path / "a.wav"
    write_wav_float32(_audio(), p, 16000)
    data = p.read_bytes()
    assert (_sha(data), len(data)) == ("29eaffe29d65a3658a508119", 844)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not in PATH")
def test_encode_video_smoke(tmp_path):
    # 5 mlx-native (H=8, W=6, 3) frames -> web tier (libx264) -> probe it back.
    frames = [
        (mx.arange(8 * 6 * 3, dtype=mx.int32) + k).astype(mx.uint8).reshape(8, 6, 3)
        for k in range(5)
    ]
    out = encode_video(frames, tmp_path / "clip.mp4", tier="web", fps=24.0, verbose=False)
    assert out.exists() and out.stat().st_size > 0
    if shutil.which("ffprobe"):
        dims = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
            text=True,
        ).strip()
        assert dims == "6,8"  # width=6, height=8
