"""Latent-domain sequence-start onset mitigation (causal-VAE first-frame spike).

The detector must fire only on a *concentrated* single-channel frame-0 spike
(the audible-click signature), not on broad all-channel elevation; the flatten
must remove it while returning a NEW array so saved sidecars / stage-2 latents
stay exactly as they came out of the transformer.
"""

import numpy as np
import mlx.core as mx

from LTX_2_MLX.audio import (
    detect_onset_latent_spike,
    flatten_onset_latent,
    mitigate_onset_latent,
)


def _latent(frame0_scale: np.ndarray, t: int = 64) -> mx.array:
    """(1, 8, t, 16) of N(0,1), with frame 0 scaled per channel by frame0_scale."""
    rng = np.random.RandomState(0)
    a = rng.randn(1, 8, t, 16).astype(np.float32)
    a[0, :, 0, :] *= frame0_scale[:, None]
    return mx.array(a)


def test_detects_concentrated_spike():
    s = np.ones(8); s[4] = 6.0
    assert detect_onset_latent_spike(_latent(s))


def test_ignores_broad_elevation():
    # every channel elevated at frame 0 -> not concentrated -> no click -> skip
    assert not detect_onset_latent_spike(_latent(np.full(8, 3.0)))


def test_ignores_normal_first_frame():
    assert not detect_onset_latent_spike(_latent(np.ones(8)))


def test_flatten_returns_copy_and_removes_spike():
    s = np.ones(8); s[4] = 6.0
    lat = _latent(s)
    before = float(mx.max(mx.abs(lat[0, 4, 0, :])))
    out = mitigate_onset_latent(lat, mode="auto")
    # original (the would-be sidecar) is untouched
    assert float(mx.max(mx.abs(lat[0, 4, 0, :]))) == before
    # spike is gone in the decode-bound copy, frame count preserved
    assert float(mx.max(mx.abs(out[0, 4, 0, :]))) < before
    assert out.shape == lat.shape


def test_off_is_passthrough():
    lat = _latent(np.ones(8))
    assert mitigate_onset_latent(lat, mode="off") is lat


def test_force_flattens_even_without_spike():
    lat = _latent(np.ones(8))
    out = flatten_onset_latent(lat, n_frames=3)
    assert out.shape == lat.shape
    assert mitigate_onset_latent(lat, mode="force").shape == lat.shape
