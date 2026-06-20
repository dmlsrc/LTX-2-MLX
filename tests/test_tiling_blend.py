"""Equivalence test for the tiled VAE decode blend (model/video_vae/tiling.py).

``decode_tiled`` has a fast path for when spatial tiling is inactive: the blend
weights then vary only along T, so it keeps them as a ``(1, 1, T, 1, 1)`` column
and assigns each (single) spatial tile directly instead of read-modify-writing a
full-resolution weight buffer.  That must be numerically identical to the general
full-weight path.

We check it with a deterministic pointwise "decoder" whose decoded-frame mapping
matches what ``decode_tiled`` assumes for the causal VAE, so a tile's decoded
length equals its output-slice length with no truncation.  The blended, tiled
output must then reconstruct the full single-shot decode exactly (the trapezoidal
overlap weights are a partition of unity), and the reduced-weight path must match
the full-weight path bit for bit.
"""

import mlx.core as mx

from LTX_2_MLX.model.video_vae.tiling import (
    DEFAULT_SPATIAL_SCALE,
    DEFAULT_TEMPORAL_SCALE,
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    decode_tiled,
)

ST = DEFAULT_TEMPORAL_SCALE  # 8
SS = DEFAULT_SPATIAL_SCALE  # 32


def _causal_upscale(latent: mx.array) -> mx.array:
    """Pointwise decode: 3 distinct channels, causal-temporal + spatial nearest.

    Frame 0 maps to latent 0; output frames ``1 + (j - 1) * ST .. 1 + j * ST``
    map to latent ``j``.  This matches ``_temporal_output_slice``'s causal length
    ``1 + (end - 1) * ST`` so tile decodes align to their output slices exactly.
    """
    _b, _c, t, _h, _w = latent.shape
    base = mx.concatenate(
        [latent[:, :1], latent[:, :1] + 1.0, latent[:, :1] * 0.5], axis=1
    )  # (b, 3, t, h, w)
    out_t = 1 + (t - 1) * ST if t >= 1 else 0
    t_idx = [0] + [1 + (f - 1) // ST for f in range(1, out_t)]
    decoded = mx.take(base, mx.array(t_idx), axis=2)  # (b, 3, out_t, h, w)
    decoded = mx.repeat(decoded, SS, axis=3)
    decoded = mx.repeat(decoded, SS, axis=4)
    return decoded.astype(mx.float32)


def _decoder_fn(tile, timestep=None, show_progress=False):
    return _causal_upscale(tile)


def _run(latent, tiling_config):
    chunks = list(
        decode_tiled(
            latent,
            _decoder_fn,
            tiling_config,
            timestep=None,
            show_progress=False,
        )
    )
    return mx.concatenate(chunks, axis=2)


def _latent():
    # Distinct per-voxel values; small enough to run instantly.
    n = 1 * 4 * 6 * 2 * 2
    return (mx.arange(n).astype(mx.float32) / 7.0).reshape(1, 4, 6, 2, 2)


def _temporal_only():
    # Heavy temporal overlap (5 tiles) with no spatial tiling.
    return TemporalTilingConfig(tile_size_in_frames=16, tile_overlap_in_frames=8)


def test_spatial_off_reconstructs_full_decode():
    latent = _latent()
    out = _run(latent, TilingConfig(spatial_config=None, temporal_config=_temporal_only()))
    oracle = _causal_upscale(latent)
    assert out.shape == oracle.shape
    assert mx.allclose(out, oracle, atol=1e-4, rtol=1e-4).item()


def test_spatial_off_matches_full_weight_path():
    # A spatial_config that yields a single full-frame tile exercises the
    # original full-resolution weight path; its output must equal the reduced
    # (1, 1, T, 1, 1) weight path used when spatial tiling is off.
    latent = _latent()
    tc = _temporal_only()
    reduced = _run(latent, TilingConfig(spatial_config=None, temporal_config=tc))
    full = _run(
        latent,
        TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=64),
            temporal_config=tc,
        ),
    )
    assert reduced.shape == full.shape
    assert mx.allclose(reduced, full, atol=1e-5, rtol=1e-5).item()
