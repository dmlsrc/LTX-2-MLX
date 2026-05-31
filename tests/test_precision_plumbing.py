"""Precision plumbing tests for the LTX-2.3 hot path."""

import mlx.core as mx
import numpy as np

import LTX_2_MLX.model.transformer.model as transformer_model
from LTX_2_MLX.kernels import fused_ops
from LTX_2_MLX.model.transformer.attention import RMSNorm
from LTX_2_MLX.model.transformer.model import (
    Modality,
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from LTX_2_MLX.model.transformer.rope import LTXRopeType


def test_float_attention_mask_is_reshaped_like_bool_mask():
    preprocessor = TransformerArgsPreprocessor(
        patchify_proj=None,
        adaln=None,
        caption_projection=None,
        inner_dim=128,
        max_pos=[20, 2048, 2048],
        num_attention_heads=32,
    )

    mask_2d = mx.array([[0.0, -1000.0]], dtype=mx.float32)
    out_2d = preprocessor._prepare_attention_mask(mask_2d, target_dtype=mx.bfloat16)
    mx.eval(out_2d)

    assert out_2d.shape == (1, 1, 1, 2)
    assert out_2d.dtype == mx.bfloat16

    mask_3d = mx.zeros((1, 2, 3), dtype=mx.float32)
    out_3d = preprocessor._prepare_attention_mask(mask_3d, target_dtype=mx.float32)
    mx.eval(out_3d)

    assert out_3d.shape == (1, 1, 2, 3)


def test_rope_double_precision_flag_reaches_simple_and_cross_precompute(monkeypatch):
    calls = []

    def fake_precompute_freqs_cis(**kwargs):
        calls.append(kwargs)
        return mx.zeros((1,), dtype=mx.float32), mx.zeros((1,), dtype=mx.float32)

    monkeypatch.setattr(transformer_model, "precompute_freqs_cis", fake_precompute_freqs_cis)

    simple = TransformerArgsPreprocessor(
        patchify_proj=None,
        adaln=None,
        caption_projection=None,
        inner_dim=128,
        max_pos=[20, 2048, 2048],
        num_attention_heads=32,
        use_middle_indices_grid=True,
        positional_embedding_theta=10000.0,
        rope_type=LTXRopeType.SPLIT,
        use_double_precision=True,
    )
    multimodal = MultiModalTransformerArgsPreprocessor(
        simple_preprocessor=simple,
        cross_scale_shift_adaln=None,
        cross_gate_adaln=None,
        cross_pe_max_pos=20,
        audio_cross_attention_dim=64,
    )

    positions = mx.zeros((1, 3, 2, 2), dtype=mx.float32)
    simple._prepare_positional_embeddings(positions)
    multimodal._prepare_cross_positional_embeddings(positions)

    assert len(calls) == 2
    assert calls[0]["use_double_precision"] is True
    assert calls[1]["use_double_precision"] is True


def test_multimodal_preprocessor_reuses_supplied_cross_rope(monkeypatch):
    supplied_self_rope = (
        mx.ones((1,), dtype=mx.float32),
        mx.zeros((1,), dtype=mx.float32),
    )
    supplied_cross_rope = (
        mx.full((1,), 2.0, dtype=mx.float32),
        mx.full((1,), 3.0, dtype=mx.float32),
    )

    class FakeSimplePreprocessor:
        inner_dim = 4
        timestep_scale_multiplier = 1000

        def prepare(self, modality):
            return TransformerArgs(
                x=mx.ones((1, 2, 4), dtype=mx.bfloat16),
                context=mx.ones((1, 2, 4), dtype=mx.bfloat16),
                timesteps=mx.ones((1, 1, 1, 4), dtype=mx.float32),
                positional_embeddings=modality.positional_embeddings,
            )

    multimodal = MultiModalTransformerArgsPreprocessor(
        simple_preprocessor=FakeSimplePreprocessor(),
        cross_scale_shift_adaln=None,
        cross_gate_adaln=None,
        cross_pe_max_pos=20,
        audio_cross_attention_dim=64,
    )

    def fail_if_recomputed(_positions):
        raise AssertionError("cross RoPE should have been reused")

    monkeypatch.setattr(
        multimodal,
        "_prepare_cross_positional_embeddings",
        fail_if_recomputed,
    )
    monkeypatch.setattr(
        multimodal,
        "_prepare_cross_attention_timestep",
        lambda *_args: (
            mx.ones((1, 1, 4, 4), dtype=mx.float32),
            mx.ones((1, 1, 1, 4), dtype=mx.float32),
        ),
    )

    modality = Modality(
        latent=mx.ones((1, 2, 4), dtype=mx.bfloat16),
        context=mx.ones((1, 2, 4), dtype=mx.bfloat16),
        context_mask=None,
        timesteps=mx.array([1.0], dtype=mx.float32),
        positions=mx.zeros((1, 3, 2, 2), dtype=mx.float32),
        sigma=mx.array([1.0], dtype=mx.float32),
        positional_embeddings=supplied_self_rope,
        cross_positional_embeddings=supplied_cross_rope,
    )
    cross_modality = Modality(
        latent=mx.ones((1, 2, 4), dtype=mx.bfloat16),
        context=mx.ones((1, 2, 4), dtype=mx.bfloat16),
        context_mask=None,
        timesteps=mx.array([1.0], dtype=mx.float32),
        positions=mx.zeros((1, 3, 2, 2), dtype=mx.float32),
        sigma=mx.array([1.0], dtype=mx.float32),
    )

    args = multimodal.prepare(modality, cross_modality)

    assert args.cross_positional_embeddings is supplied_cross_rope


def test_qk_rms_norm_casts_back_to_input_dtype_with_fp32_weight():
    norm = RMSNorm(8)
    norm.weight = norm.weight.astype(mx.float32)
    x = mx.ones((1, 2, 8), dtype=mx.bfloat16)

    out = norm(x)
    mx.eval(out)

    assert out.dtype == mx.bfloat16


def test_fused_adaln_falls_back_for_non_default_eps(monkeypatch):
    x = mx.ones((1, 2, 4096), dtype=mx.bfloat16)
    scale = mx.zeros((1, 1, 4096), dtype=mx.float32)
    shift = mx.zeros((1, 1, 4096), dtype=mx.float32)
    calls = {"fallback": 0}

    def fake_fallback(x_arg, scale_arg, shift_arg, eps_arg):
        calls["fallback"] += 1
        assert scale_arg is scale
        assert shift_arg is shift
        assert eps_arg == 1e-5
        return x_arg

    monkeypatch.setattr(fused_ops, "_adaln_norm_mlx", fake_fallback)

    out = fused_ops.adaln_norm_fused(x, scale, shift, 1e-5)

    assert out is x
    assert calls["fallback"] == 1


def test_interleaved_rope_fused_path_preserves_fp32_cos_sin_math():
    x = mx.array(
        [[[1.125, -2.25, 3.5, -4.75], [0.5, 1.5, -2.5, 4.0]]],
        dtype=mx.bfloat16,
    )
    cos = mx.array(
        [[[0.90625, 0.90625, 0.53125, 0.53125], [0.25, 0.25, -0.75, -0.75]]],
        dtype=mx.float32,
    )
    sin = mx.array(
        [[[0.421875, 0.421875, -0.84375, -0.84375], [0.96875, 0.96875, 0.65625, 0.65625]]],
        dtype=mx.float32,
    )

    out = fused_ops.interleaved_rope(x, cos, sin)
    x_dup = x.reshape(1, 2, 2, 2)
    x1 = x_dup[..., 0]
    x2 = x_dup[..., 1]
    x_rot = mx.stack([-x2, x1], axis=-1).reshape(x.shape)
    expected = (x * cos + x_rot * sin).astype(x.dtype)
    mx.eval(out, expected)

    np.testing.assert_array_equal(
        np.array(out.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


def test_load_av_transformer_uses_transformer_precision_metadata(monkeypatch):
    from scripts import generate as gen

    captured = {}

    class FakeLTXAVModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gen, "LTXAVModel", FakeLTXAVModel)
    monkeypatch.setattr(
        gen,
        "_read_transformer_config",
        lambda _path: {
            "frequencies_precision": "float64",
            "rope_type": "split",
            "positional_embedding_max_pos": [20, 2048, 2048],
            "av_ca_timestep_scale_multiplier": 1000.0,
        },
    )

    gen.load_av_transformer(
        weights_path="/nonexistent/transformer.safetensors",
        config_weights_path="/nonexistent/config.safetensors",
        cross_attention_adaln=True,
        apply_gated_attention=True,
    )

    assert captured["use_double_precision_rope"] is True
    assert captured["rope_type"] == LTXRopeType.SPLIT
    assert captured["positional_embedding_max_pos"] == [20, 2048, 2048]
    assert captured["av_ca_timestep_scale_multiplier"] == 1000
