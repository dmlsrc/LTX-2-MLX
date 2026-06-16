"""Tests for custom Metal kernels in ``LTX_2_MLX.kernels.fused_ops``.

Covers the two T2V production-shape kernels:
  - ``adaln_norm_fused``      replaces ``_adaln_inline``
  - ``gated_add_fused``       replaces ``_residual_gate_inline``

Both must:
  1. Match the MLX reference path to cos sim >= 0.9999 at production shape
     (B=1, T=14640, C=4096, x bf16, scale/shift/gate fp32 (B, 1, C)).
  2. Fall back to MLX for non-matching shape/dtype (audio C=2048, I2V
     per-token (B, T, C) scale/shift, fp32 x, etc.).

These kernels are gated behind ``LTX_FUSED_ADALN`` and ``LTX_FUSED_GATED_ADD``
env vars at the transformer dispatcher; this test exercises the kernel
functions directly to verify numerics independent of the env-var wiring.
"""

import mlx.core as mx
import pytest

from LTX_2_MLX.kernels.fused_ops import (
    adaln_norm_fused,
    gated_add_fused,
    _adaln_norm_mlx,
    _gated_add_mlx,
    _adaln_t2v_broadcast_compatible,
    _gated_add_t2v_broadcast_compatible,
)


# Production T2V video shape: B=1, T=14640, C=4096.  C derives from
# num_attention_heads (32) x attention_head_dim (128) in
# LTX_2_MLX/model/transformer/model.py.
PROD_T = 14640
PROD_C = 4096
EPS = 1e-6


def _cos_sim(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    num = (af * bf).sum()
    denom = mx.sqrt((af * af).sum()) * mx.sqrt((bf * bf).sum())
    return float(num / denom)


@pytest.fixture
def prod_inputs():
    """Build a single T2V production-shape input set."""
    mx.random.seed(42)
    x = mx.random.normal(shape=(1, PROD_T, PROD_C), dtype=mx.float32).astype(mx.bfloat16)
    scale = mx.random.normal(shape=(1, 1, PROD_C), dtype=mx.float32) * 0.1
    shift = mx.random.normal(shape=(1, 1, PROD_C), dtype=mx.float32) * 0.1
    gate = mx.random.normal(shape=(1, 1, PROD_C), dtype=mx.float32) * 0.5
    residual = mx.random.normal(shape=(1, PROD_T, PROD_C), dtype=mx.float32).astype(mx.bfloat16)
    branch = mx.random.normal(shape=(1, PROD_T, PROD_C), dtype=mx.float32).astype(mx.bfloat16)
    mx.eval(x, scale, shift, gate, residual, branch)
    return dict(x=x, scale=scale, shift=shift, gate=gate, residual=residual, branch=branch)


# ── adaln_norm_fused ───────────────────────────────────────────────────────


def test_adaln_norm_fused_t2v_parity(prod_inputs):
    """Kernel matches MLX path at production T2V shape (cos sim >= 0.9999)."""
    x, scale, shift = prod_inputs["x"], prod_inputs["scale"], prod_inputs["shift"]
    y_mlx = _adaln_norm_mlx(x, scale, shift, EPS)
    y_fused = adaln_norm_fused(x, scale, shift, EPS)
    mx.eval(y_mlx, y_fused)

    assert y_fused.shape == y_mlx.shape
    assert y_fused.dtype == mx.bfloat16
    assert _cos_sim(y_mlx, y_fused) >= 0.9999


def test_adaln_t2v_gate_predicate(prod_inputs):
    """Gate predicate returns True for the production T2V shape."""
    x, scale, shift = prod_inputs["x"], prod_inputs["scale"], prod_inputs["shift"]
    assert _adaln_t2v_broadcast_compatible(x, scale, shift)


def test_adaln_fallback_on_fp32_x(prod_inputs):
    """fp32 x doesn't match the gate → MLX fallback preserves fp32 output."""
    x = prod_inputs["x"].astype(mx.float32)
    scale, shift = prod_inputs["scale"], prod_inputs["shift"]
    y = adaln_norm_fused(x, scale, shift, EPS)
    mx.eval(y)
    assert y.dtype == mx.float32  # fallback path produces fp32 output


def test_adaln_fallback_on_per_token_scale(prod_inputs):
    """I2V per-token scale/shift (B, T, C) doesn't match gate → MLX fallback."""
    x = prod_inputs["x"]
    scale_pt = mx.broadcast_to(prod_inputs["scale"], (1, PROD_T, PROD_C))
    shift_pt = mx.broadcast_to(prod_inputs["shift"], (1, PROD_T, PROD_C))
    mx.eval(scale_pt, shift_pt)

    assert not _adaln_t2v_broadcast_compatible(x, scale_pt, shift_pt)

    y_mlx = _adaln_norm_mlx(x, scale_pt, shift_pt, EPS)
    y_fused = adaln_norm_fused(x, scale_pt, shift_pt, EPS)
    mx.eval(y_mlx, y_fused)
    # Both should be the MLX path, hence identical.
    assert _cos_sim(y_mlx, y_fused) >= 0.99999


def test_adaln_audio_dim_t2v_parity():
    """C=2048 (audio inner dim) hits the audio kernel variant, parity vs MLX."""
    B, T, C = 1, 1024, 2048
    mx.random.seed(0)
    x = mx.random.normal(shape=(B, T, C), dtype=mx.float32).astype(mx.bfloat16)
    scale = mx.random.normal(shape=(B, 1, C), dtype=mx.float32) * 0.1
    shift = mx.random.normal(shape=(B, 1, C), dtype=mx.float32) * 0.1
    mx.eval(x, scale, shift)

    # Audio C is now a supported kernel variant (BLOCK=256, VPT=2) so the
    # T2V gate predicate passes for it.
    assert _adaln_t2v_broadcast_compatible(x, scale, shift)

    y_mlx = _adaln_norm_mlx(x, scale, shift, EPS)
    y_fused = adaln_norm_fused(x, scale, shift, EPS)
    mx.eval(y_mlx, y_fused)
    assert y_fused.shape == y_mlx.shape
    assert y_fused.dtype == mx.bfloat16
    assert _cos_sim(y_mlx, y_fused) >= 0.9999


def test_adaln_unsupported_dim_falls_back():
    """An unsupported C (e.g. 3584) doesn't match any registered kernel and
    routes to MLX fallback."""
    B, T, C = 1, 256, 3584  # not in _ADALN_FUSED_CONFIGS
    mx.random.seed(0)
    x = mx.random.normal(shape=(B, T, C), dtype=mx.float32).astype(mx.bfloat16)
    scale = mx.random.normal(shape=(B, 1, C), dtype=mx.float32) * 0.1
    shift = mx.random.normal(shape=(B, 1, C), dtype=mx.float32) * 0.1
    mx.eval(x, scale, shift)

    assert not _adaln_t2v_broadcast_compatible(x, scale, shift)

    y_mlx = _adaln_norm_mlx(x, scale, shift, EPS)
    y_fused = adaln_norm_fused(x, scale, shift, EPS)
    mx.eval(y_mlx, y_fused)
    assert _cos_sim(y_mlx, y_fused) >= 0.99999  # both = MLX path


# ── gated_add_fused ────────────────────────────────────────────────────────


def test_gated_add_fused_t2v_parity(prod_inputs):
    """Kernel matches MLX path at production T2V shape (cos sim >= 0.9999)."""
    residual = prod_inputs["residual"]
    branch = prod_inputs["branch"]
    gate = prod_inputs["gate"]
    z_mlx = _gated_add_mlx(residual, branch, gate)
    z_fused = gated_add_fused(residual, branch, gate)
    mx.eval(z_mlx, z_fused)

    assert z_fused.shape == z_mlx.shape
    assert z_fused.dtype == mx.bfloat16
    assert _cos_sim(z_mlx, z_fused) >= 0.9999


def test_gated_add_t2v_gate_predicate(prod_inputs):
    """Gate predicate returns True for the production T2V shape."""
    assert _gated_add_t2v_broadcast_compatible(
        prod_inputs["residual"], prod_inputs["branch"], prod_inputs["gate"]
    )


def test_gated_add_fallback_on_per_token_gate(prod_inputs):
    """I2V per-token gate (B, T, C) doesn't match gate → MLX fallback."""
    residual = prod_inputs["residual"]
    branch = prod_inputs["branch"]
    gate_pt = mx.broadcast_to(prod_inputs["gate"], (1, PROD_T, PROD_C))
    mx.eval(gate_pt)

    assert not _gated_add_t2v_broadcast_compatible(residual, branch, gate_pt)

    z_mlx = _gated_add_mlx(residual, branch, gate_pt)
    z_fused = gated_add_fused(residual, branch, gate_pt)
    mx.eval(z_mlx, z_fused)
    assert _cos_sim(z_mlx, z_fused) >= 0.99999


def test_gated_add_audio_dim_t2v_parity():
    """C=2048 (audio inner dim) hits the audio kernel variant, parity vs MLX."""
    B, T, C = 1, 1024, 2048
    mx.random.seed(0)
    residual = mx.random.normal(shape=(B, T, C), dtype=mx.float32).astype(mx.bfloat16)
    branch = mx.random.normal(shape=(B, T, C), dtype=mx.float32).astype(mx.bfloat16)
    gate = mx.random.normal(shape=(B, 1, C), dtype=mx.float32) * 0.5
    mx.eval(residual, branch, gate)

    assert _gated_add_t2v_broadcast_compatible(residual, branch, gate)

    z_mlx = _gated_add_mlx(residual, branch, gate)
    z_fused = gated_add_fused(residual, branch, gate)
    mx.eval(z_mlx, z_fused)
    assert z_fused.shape == z_mlx.shape
    assert z_fused.dtype == mx.bfloat16
    assert _cos_sim(z_mlx, z_fused) >= 0.9999


def test_gated_add_unsupported_dim_falls_back():
    """An unsupported C doesn't match any registered kernel → MLX fallback."""
    B, T, C = 1, 256, 3584  # not in _GATED_ADD_FUSED_CONFIGS
    mx.random.seed(0)
    residual = mx.random.normal(shape=(B, T, C), dtype=mx.float32).astype(mx.bfloat16)
    branch = mx.random.normal(shape=(B, T, C), dtype=mx.float32).astype(mx.bfloat16)
    gate = mx.random.normal(shape=(B, 1, C), dtype=mx.float32) * 0.5
    mx.eval(residual, branch, gate)

    assert not _gated_add_t2v_broadcast_compatible(residual, branch, gate)

    z_mlx = _gated_add_mlx(residual, branch, gate)
    z_fused = gated_add_fused(residual, branch, gate)
    mx.eval(z_mlx, z_fused)
    assert _cos_sim(z_mlx, z_fused) >= 0.99999
