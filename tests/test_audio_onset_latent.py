"""Latent-domain sequence-start onset DETECTION (causal-VAE first-frame spike).

The detector is the trigger for the waveform zero-fill mitigation, so it must
fire only on a *concentrated* single-channel frame-0 spike (the audible-click
signature), not on broad all-channel elevation (which decodes quietly).
"""

import mlx.core as mx
import numpy as np

from LTX_2_MLX.audio import detect_onset_latent_spike


def _latent(frame0_scale: np.ndarray, t: int = 64) -> mx.array:
    """(1, 8, t, 16) of N(0,1), with frame 0 scaled per channel by frame0_scale."""
    rng = np.random.RandomState(0)
    a = rng.randn(1, 8, t, 16).astype(np.float32)
    a[0, :, 0, :] *= frame0_scale[:, None]
    return mx.array(a)


def test_detects_concentrated_spike():
    s = np.ones(8)
    s[4] = 6.0
    assert detect_onset_latent_spike(_latent(s))


def test_ignores_broad_elevation():
    # every channel elevated at frame 0 -> not concentrated -> decodes quiet -> skip
    assert not detect_onset_latent_spike(_latent(np.full(8, 3.0)))


def test_ignores_normal_first_frame():
    assert not detect_onset_latent_spike(_latent(np.ones(8)))
