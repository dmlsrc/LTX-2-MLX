"""Parity tests for the GaussianNoiser conditioning blend.

Pins the two-step clean_latent blend that matches Lightricks' GaussianNoiser
(`lerp(clean_latent, lerp(latent, noise, noise_scale), denoise_mask)`), which
sources conditioning values from clean_latent rather than the noisy latent field.
The key behaviors:
  * conditioned region (fractional mask) at noise_scale<1 follows the two-lerp;
  * a hard mask (0) returns clean_latent exactly (conditioning untouched);
  * at noise_scale==1 the two-lerp reduces to the old single-step blend, so
    stage 1 (noise_scale=1) is unchanged from before the port.
"""

import mlx.core as mx
import numpy as np

from LTX_2_MLX.components.noisers import GaussianNoiser
from LTX_2_MLX.types import LatentState


def _state(latent: mx.array, clean: mx.array, mask: mx.array) -> LatentState:
    b, t = latent.shape[0], latent.shape[1]
    return LatentState(
        latent=latent,
        clean_latent=clean,
        denoise_mask=mask,
        positions=mx.zeros((b, 3, t), dtype=mx.float32),
        uniform_mask=False,
    )


def _two_lerp(latent, clean, mask, noise, ns):
    inner = latent + ns * (noise - latent)
    return clean + mask * (inner - clean)


def _split_state(rng):
    """Generate region (latent==clean) + soft-conditioned region (latent=0, clean=tokens)."""
    b, c = 1, 4
    gen = mx.array(rng.randn(b, 3, c).astype(np.float32))
    tokens = mx.array(rng.randn(b, 3, c).astype(np.float32))
    latent = mx.concatenate([gen, mx.zeros((b, 3, c))], axis=1)
    clean = mx.concatenate([gen, tokens], axis=1)
    mask = mx.concatenate([mx.ones((b, 3, 1)), mx.full((b, 3, 1), 0.05)], axis=1)
    return latent, clean, mask, tokens


def test_noiser_matches_upstream_two_lerp():
    key = mx.random.key(7)
    latent, clean, mask, _ = _split_state(np.random.RandomState(3))
    ns = 0.4
    noise = mx.random.normal(shape=latent.shape, dtype=mx.float32, key=key)
    expected = _two_lerp(latent, clean, mask, noise, ns)
    out = GaussianNoiser(key=key)(_state(latent, clean, mask), noise_scale=ns)
    assert np.allclose(np.array(out.latent), np.array(expected), atol=1e-5)


def test_noiser_hard_mask_preserves_clean():
    # denoise_mask=0 -> conditioned positions come out exactly clean_latent
    key = mx.random.key(1)
    rng = np.random.RandomState(5)
    clean = mx.array(rng.randn(1, 4, 4).astype(np.float32))
    latent = mx.zeros((1, 4, 4))
    mask = mx.zeros((1, 4, 1))
    out = GaussianNoiser(key=key)(_state(latent, clean, mask), noise_scale=0.5)
    assert np.allclose(np.array(out.latent), np.array(clean), atol=1e-6)


def test_noiser_generate_region_is_pure_noise_schedule():
    # denoise_mask=1 -> output is lerp(latent, noise, ns); clean_latent is ignored
    key = mx.random.key(4)
    rng = np.random.RandomState(8)
    latent = mx.array(rng.randn(1, 4, 4).astype(np.float32))
    clean = mx.array(rng.randn(1, 4, 4).astype(np.float32))  # deliberately != latent
    mask = mx.ones((1, 4, 1))
    ns = 0.3
    noise = mx.random.normal(shape=latent.shape, dtype=mx.float32, key=key)
    out = GaussianNoiser(key=key)(_state(latent, clean, mask), noise_scale=ns)
    assert np.allclose(np.array(out.latent), np.array(latent + ns * (noise - latent)), atol=1e-5)


def test_noiser_stage1_equals_old_single_step():
    # noise_scale==1 -> the two-lerp reduces to the old `noise*mask + latent*(1-mask)`
    # blend even for a fractional mask (divergence is tokens*mask*(1-ns) = 0 at ns=1).
    key = mx.random.key(2)
    latent_new, clean, mask, tokens = _split_state(np.random.RandomState(9))
    # old-style latent held the tokens where new-style holds zeros
    latent_old = clean
    noise = mx.random.normal(shape=latent_new.shape, dtype=mx.float32, key=key)
    out_new = GaussianNoiser(key=key)(_state(latent_new, clean, mask), noise_scale=1.0)
    old = noise * mask + latent_old * (1 - mask)
    assert np.allclose(np.array(out_new.latent), np.array(old), atol=1e-5)
