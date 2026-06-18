"""Byte-level regression net for videotoolbox AudioTrack (numpy -> MLX rewrite).

Pins the interleaved PCM bytes AudioTrack hands to CMSampleBuffers, and the
save_wav output, on a deterministic waveform. Runs with both a numpy and an mlx
input waveform; both must match the golden captured from the numpy version.
Skipped where pyobjc / CoreMedia is unavailable.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest


def _have_pyobjc() -> bool:
    try:
        from LTX_2_MLX.videotoolbox import _compat
        _compat.require_pyobjc()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_pyobjc(), reason="pyobjc / CoreMedia unavailable")


def _waveform_np():
    return (np.arange(100, dtype=np.float32) / 99 * 2 - 1).reshape(2, 50)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:24]


@pytest.mark.parametrize("container", ["numpy", "mlx"])
def test_audiotrack_interleaved_bytes(container):
    from LTX_2_MLX.videotoolbox.audio import AudioTrack
    w = _waveform_np()
    if container == "mlx":
        w = mx.array(w)
    track = AudioTrack(w, 48000)
    assert (_sha(track._bytes), len(track._bytes)) == ("20ac23daffeedf0ac5439bbe", 400)


@pytest.mark.parametrize("container", ["numpy", "mlx"])
def test_audiotrack_save_wav(container, tmp_path):
    from LTX_2_MLX.videotoolbox.audio import AudioTrack
    w = _waveform_np()
    if container == "mlx":
        w = mx.array(w)
    track = AudioTrack(w, 48000)
    p = tmp_path / "a.wav"
    track.save_wav(p)
    data = p.read_bytes()
    assert (_sha(data), len(data)) == ("1f7316ffeb528dd8ab4883da", 444)


@pytest.mark.parametrize(
    "writer_name,tol",
    [("write_wav_int16", 2e-4), ("write_wav_float32", 1e-6)],
    ids=["int16", "float32"],
)
def test_read_wav_roundtrip(writer_name, tol, tmp_path):
    # AVFoundation AVAudioFile reads both PCM int16 and IEEE float32 WAV.
    from LTX_2_MLX import video_encoder
    from LTX_2_MLX.videotoolbox.audio import read_wav

    aud = mx.arange(200, dtype=mx.float32).reshape(2, 100) / 199 * 2 - 1  # (channels, frames)
    getattr(video_encoder, writer_name)(aud, tmp_path / "rt.wav", 48000)
    sr, samples = read_wav(tmp_path / "rt.wav")
    assert sr == 48000
    assert tuple(int(x) for x in samples.shape) == (2, 100)
    assert mx.max(mx.abs(samples - aud)).item() < tol
