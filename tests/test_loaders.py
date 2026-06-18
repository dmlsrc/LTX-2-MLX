"""Tests for loader module: weight conversion, LoRA, and registry."""


import mlx.core as mx
import mlx.nn as nn
import pytest

from LTX_2_MLX.loader.lora_loader import (
    LoRAConfig,
    _lora_key_categories,
    format_lora_stage_scale_lines,
    fuse_loras_into_model,
    lora_configs_for_stage,
    lora_configs_for_stage_delta,
    lora_configs_have_stage_strengths,
)
from LTX_2_MLX.loader.transformer_cache import (
    TRANSFORMER_CACHE_RESTORE_ATTR,
    get_transformer_cache_restore_state,
    restore_transformer_cache_state,
)

# Import the modules under test
from LTX_2_MLX.loader.weight_converter import (
    _convert_numeric_dicts_to_lists,
    _flatten_to_nested,
    convert_pytorch_key_to_mlx,
)

# ============================================================================
# Weight Converter Tests
# ============================================================================

class TestConvertPytorchKeyToMlx:
    """Tests for convert_pytorch_key_to_mlx function."""

    def test_skips_audio_keys_by_default(self):
        """Test audio/av_ca keys are skipped by default."""
        audio_keys = [
            "av_ca.to_q.weight",
            "audio_ff.net.0.proj.weight",
        ]
        for key in audio_keys:
            result = convert_pytorch_key_to_mlx(key, include_audio=False)
            assert result is None

    def test_includes_audio_when_flag_set(self):
        """Test audio keys are included when include_audio=True."""
        key = "audio_ff.net.0.proj.weight"
        result = convert_pytorch_key_to_mlx(key, include_audio=True)
        assert result is not None
        # The function converts ff.net.0.proj -> ff.project_in.proj
        assert "audio_ff.project_in.proj" in result or result == "audio_ff.net.0.proj.weight"

    def test_skips_video_embeddings_connector(self):
        """Test video_embeddings_connector keys are skipped."""
        key = "video_embeddings_connector.proj.weight"
        result = convert_pytorch_key_to_mlx(key)
        assert result is None

    def test_to_out_conversion(self):
        """Test to_out.0 -> to_out conversion."""
        key = "attn.to_out.0.weight"
        result = convert_pytorch_key_to_mlx(key)
        assert result == "attn.to_out.weight"

    def test_ff_net_conversions(self):
        """Test feed-forward module name conversion."""
        assert (
            convert_pytorch_key_to_mlx("ff.net.0.proj.weight")
            == "ff.project_in.proj.weight"
        )
        assert convert_pytorch_key_to_mlx("ff.net.2.weight") == "ff.project_out.weight"

    def test_audio_ff_conversions(self):
        """Test audio feed-forward module name conversion when audio is included."""
        assert (
            convert_pytorch_key_to_mlx("audio_ff.net.0.proj.weight", include_audio=True)
            == "audio_ff.project_in.proj.weight"
        )
        assert (
            convert_pytorch_key_to_mlx("audio_ff.net.2.weight", include_audio=True)
            == "audio_ff.project_out.weight"
        )

    def test_skips_audio_embeddings_connector(self):
        """Test audio_embeddings_connector keys are skipped."""
        key = "audio_embeddings_connector.proj.weight"
        result = convert_pytorch_key_to_mlx(key, include_audio=True)
        assert result is None


class TestFlattenToNested:
    """Tests for _flatten_to_nested function."""

    def test_simple_nesting(self):
        """Test simple key nesting."""
        flat = {
            "layer.weight": mx.array([1, 2, 3]),
            "layer.bias": mx.array([4, 5, 6]),
        }
        nested = _flatten_to_nested(flat)
        assert "layer" in nested
        assert "weight" in nested["layer"]
        assert "bias" in nested["layer"]

    def test_deep_nesting(self):
        """Test deep key nesting."""
        flat = {
            "model.layers.0.attn.weight": mx.array([1, 2]),
        }
        nested = _flatten_to_nested(flat)
        # Note: numeric keys like "0" get converted to list indices
        # So layers becomes a list, and we access via index
        assert nested["model"]["layers"][0]["attn"]["weight"] is not None


class TestConvertNumericDictsToLists:
    """Tests for _convert_numeric_dicts_to_lists function."""

    def test_converts_numeric_dict_to_list(self):
        """Test numeric string keys become list indices."""
        d = {"0": "a", "1": "b", "2": "c"}
        result = _convert_numeric_dicts_to_lists(d)
        assert result == ["a", "b", "c"]

    def test_preserves_non_numeric_dict(self):
        """Test non-numeric dicts are preserved."""
        d = {"weight": mx.array([1]), "bias": mx.array([2])}
        result = _convert_numeric_dicts_to_lists(d)
        assert isinstance(result, dict)
        assert "weight" in result

    def test_handles_gaps_in_indices(self):
        """Test handles gaps in numeric indices."""
        d = {"0": "a", "2": "c"}
        result = _convert_numeric_dicts_to_lists(d)
        assert result == ["a", None, "c"]

    def test_recursive_conversion(self):
        """Test recursive conversion of nested structures."""
        d = {
            "layers": {
                "0": {"weight": mx.array([1])},
                "1": {"weight": mx.array([2])},
            }
        }
        result = _convert_numeric_dicts_to_lists(d)
        assert isinstance(result["layers"], list)
        assert len(result["layers"]) == 2


# ============================================================================
# LoRA Loader Tests
# ============================================================================

class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_strength(self):
        """Test default LoRA strength is 1.0."""
        config = LoRAConfig(path="/path/to/lora.safetensors")
        assert config.strength == 1.0

    def test_custom_strength(self):
        """Test custom LoRA strength."""
        config = LoRAConfig(path="/path/to/lora.safetensors", strength=0.5)
        assert config.strength == 0.5

    def test_rejects_out_of_range_strength(self):
        """Test strength validation (-2.0 to 2.0)."""
        with pytest.raises(ValueError):
            LoRAConfig(path="/path", strength=3.0)

        with pytest.raises(ValueError):
            LoRAConfig(path="/path", strength=-3.0)

    def test_accepts_edge_strengths(self):
        """Test edge values are accepted."""
        config1 = LoRAConfig(path="/path", strength=-2.0)
        config2 = LoRAConfig(path="/path", strength=2.0)
        assert config1.strength == -2.0
        assert config2.strength == 2.0

    def test_accepts_control_path_exclude_tags(self):
        """Test V2 control-path LoRA exclude tags validate."""
        config = LoRAConfig(
            path="/path",
            exclude=(
                "adaln",
                "prompt_adaln",
                "scale_shift",
                "prompt_scale_shift",
                "gate_adaln",
                "av_ca",
                "cross_control",
                "distill_control",
            ),
        )

        assert config.exclude == (
            "adaln",
            "prompt_adaln",
            "scale_shift",
            "prompt_scale_shift",
            "gate_adaln",
            "av_ca",
            "cross_control",
            "distill_control",
        )

    def test_rejects_unknown_exclude_tag(self):
        """Test unknown LoRA exclude categories are rejected."""
        with pytest.raises(ValueError, match="Unknown LoRA exclude categories"):
            LoRAConfig(path="/path", exclude=("sideways",))

    def test_stage_strength_defaults_to_scalar_strength(self):
        """Test stage strengths fall back to the scalar strength."""
        config = LoRAConfig(path="/path", strength=0.4)

        assert not config.has_stage_strengths()
        assert config.strength_for_stage(1) == 0.4
        assert config.strength_for_stage(2) == 0.4

    def test_stage_strength_overrides(self):
        """Test per-stage LoRA strengths override scalar strength."""
        config = LoRAConfig(
            path="/path",
            strength=0.4,
            stage_1_strength=0.25,
            stage_2_strength=0.75,
        )

        assert config.has_stage_strengths()
        assert config.strength_for_stage(1) == 0.25
        assert config.strength_for_stage(2) == 0.75

    def test_rejects_out_of_range_stage_strengths(self):
        """Test stage strength validation (-2.0 to 2.0)."""
        with pytest.raises(ValueError):
            LoRAConfig(path="/path", stage_1_strength=3.0)

        with pytest.raises(ValueError):
            LoRAConfig(path="/path", stage_2_strength=-3.0)

    def test_rejects_invalid_stage(self):
        """Test only stages 1 and 2 are accepted."""
        config = LoRAConfig(path="/path")

        with pytest.raises(ValueError, match="stage must be 1 or 2"):
            config.strength_for_stage(3)

    def test_lora_configs_have_stage_strengths(self):
        """Test stage-strength detection."""
        configs = [
            LoRAConfig(path="/plain", strength=0.8),
            LoRAConfig(path="/stage", strength=0.8, stage_2_strength=0.5),
        ]

        assert lora_configs_have_stage_strengths(configs)
        assert not lora_configs_have_stage_strengths(configs[:1])
        assert not lora_configs_have_stage_strengths(None)

    def test_lora_configs_for_stage_uses_fallback_and_filters_zero(self):
        """Test stage config derivation keeps default behavior unless overridden."""
        configs = [
            LoRAConfig(path="/plain", strength=0.8, exclude=("audio",)),
            LoRAConfig(path="/stage", strength=0.7, stage_1_strength=0.0, stage_2_strength=0.5),
        ]

        stage_1 = lora_configs_for_stage(configs, 1)
        stage_2 = lora_configs_for_stage(configs, 2)

        assert [(cfg.path, cfg.strength, cfg.exclude) for cfg in stage_1] == [
            ("/plain", 0.8, ("audio",)),
        ]
        assert [(cfg.path, cfg.strength, cfg.exclude) for cfg in stage_2] == [
            ("/plain", 0.8, ("audio",)),
            ("/stage", 0.5, ()),
        ]

    def test_lora_configs_for_stage_delta(self):
        """Test stage deltas are computed from resolved strengths."""
        configs = [
            LoRAConfig(path="/unchanged", strength=0.8),
            LoRAConfig(path="/increase", strength=0.7, stage_1_strength=0.25, stage_2_strength=0.5),
            LoRAConfig(path="/decrease", strength=0.7, stage_1_strength=0.5, stage_2_strength=0.25),
        ]

        delta = lora_configs_for_stage_delta(configs, from_stage=1, to_stage=2)

        assert [(cfg.path, cfg.strength) for cfg in delta] == [
            ("/increase", 0.25),
            ("/decrease", -0.25),
        ]

    def test_format_lora_stage_scale_lines(self):
        """Test staged LoRA scale logs include stage totals and changes."""
        configs = [
            LoRAConfig(path="/loras/unchanged.safetensors", strength=0.8),
            LoRAConfig(
                path="/loras/increase.safetensors",
                strength=0.7,
                stage_1_strength=0.25,
                stage_2_strength=0.5,
            ),
            LoRAConfig(
                path="/loras/decrease.safetensors",
                strength=0.7,
                stage_1_strength=0.5,
                stage_2_strength=0.25,
            ),
        ]

        assert format_lora_stage_scale_lines(configs, 1) == [
            "    unchanged.safetensors: total=0.8000",
            "    increase.safetensors: total=0.2500",
            "    decrease.safetensors: total=0.5000",
        ]
        assert format_lora_stage_scale_lines(configs, 2, from_stage=1) == [
            "    increase.safetensors: total=0.5000, change=+0.2500",
            "    decrease.safetensors: total=0.2500, change=-0.2500",
        ]
        assert format_lora_stage_scale_lines(
            configs,
            2,
            from_stage=1,
            include_unchanged=True,
        ) == [
            "    unchanged.safetensors: total=0.8000, change=+0.0000",
            "    increase.safetensors: total=0.5000, change=+0.2500",
            "    decrease.safetensors: total=0.2500, change=-0.2500",
        ]


class TestLoRACategories:
    """Tests for LoRA target exclude category classification."""

    def test_distill_control_catches_cross_attention_bridge(self):
        cats = _lora_key_categories(
            "transformer_blocks.0.audio_to_video_attn.to_v.weight"
        )

        assert {
            "cross",
            "audio_to_video_attn",
            "to_v",
            "cross_control",
            "distill_control",
        } <= cats

    def test_cross_control_catches_video_cross_attention(self):
        cats = _lora_key_categories("transformer_blocks.0.attn2.to_q.weight")

        assert {"video", "attn", "attn2", "to_q", "cross_control"} <= cats
        assert "distill_control" not in cats

    def test_cross_control_catches_audio_cross_attention(self):
        cats = _lora_key_categories("transformer_blocks.0.audio_attn2.to_q.weight")

        assert {"audio", "attn", "audio_attn2", "to_q", "cross_control"} <= cats
        assert "distill_control" not in cats

    def test_distill_control_catches_gate_logits(self):
        cats = _lora_key_categories("transformer_blocks.0.attn1.to_gate_logits.weight")

        assert {"video", "gate", "attn1", "to_gate_logits", "distill_control"} <= cats
        assert "cross_control" not in cats

    def test_control_path_tags_top_level_adaln(self):
        cats = _lora_key_categories("prompt_adaln_single.linear.weight")

        assert {
            "video",
            "adaln",
            "prompt_adaln",
            "prompt_scale_shift",
            "cross_control",
            "distill_control",
        } <= cats

    def test_control_path_tags_audio_adaln_as_audio(self):
        cats = _lora_key_categories("audio_prompt_adaln_single.linear.weight")

        assert {
            "audio",
            "adaln",
            "prompt_adaln",
            "prompt_scale_shift",
            "cross_control",
            "distill_control",
        } <= cats

    def test_control_path_tags_av_ca_scale_shift_and_gate_adaln(self):
        scale_shift = _lora_key_categories(
            "av_ca_video_scale_shift_adaln_single.linear.weight"
        )
        gate = _lora_key_categories("av_ca_a2v_gate_adaln_single.linear.weight")

        assert {
            "video",
            "av_ca",
            "adaln",
            "scale_shift",
            "cross_control",
            "distill_control",
        } <= scale_shift
        assert {
            "video",
            "av_ca",
            "adaln",
            "gate_adaln",
            "cross_control",
            "distill_control",
        } <= gate


class TinyLoRALinear(nn.Module):
    def __init__(self, shape, dtype=mx.float32):
        super().__init__()
        self.weight = mx.zeros(shape, dtype=dtype)


class TinyLoRAFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_out = TinyLoRALinear((2, 2), dtype=mx.float32)


class TinyLoRAAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = TinyLoRALinear((1, 1), dtype=mx.float16)


class TinyLoRABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = TinyLoRAFF()
        self.attn1 = TinyLoRAAttn()


class TinyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = [TinyLoRABlock()]


class TestFuseLorasIntoModel:
    """Tests for the unified model-level LoRA fuser."""

    def test_converts_ff_key_and_ignores_alpha(self, tmp_path):
        path = tmp_path / "ff_lora.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": mx.array(
                    [[1.0, 2.0]], dtype=mx.float32
                ),
                "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": mx.array(
                    [[3.0], [4.0]], dtype=mx.float32
                ),
                "diffusion_model.transformer_blocks.0.ff.net.2.alpha": mx.array(0.5),
            },
        )
        model = TinyLoRAModel()
        setattr(model, TRANSFORMER_CACHE_RESTORE_ATTR, {
            "valid": True,
            "cache_path": tmp_path / "transformer.safetensors",
            "transformer_cache_quantize": "off",
            "video_ff_quantize_specs": (),
            "video_ff_quantize_group_size": None,
            "video_ff_quantize_bits": None,
        })

        fuse_loras_into_model(model, [LoRAConfig(str(path))], verbose=False)

        expected = mx.array([[3.0, 6.0], [4.0, 8.0]], dtype=mx.float32)
        assert mx.allclose(model.transformer_blocks[0].ff.project_out.weight, expected)
        assert getattr(model, TRANSFORMER_CACHE_RESTORE_ATTR)["valid"] is False

    def test_fuses_bfloat16_safetensors_lora(self, tmp_path):
        path = tmp_path / "bf16_lora.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "transformer_blocks.0.ff.project_out.lora_A.weight": mx.array(
                    [[1.0, 2.0]], dtype=mx.bfloat16
                ),
                "transformer_blocks.0.ff.project_out.lora_B.weight": mx.array(
                    [[3.0], [4.0]], dtype=mx.bfloat16
                ),
            },
        )
        model = TinyLoRAModel()

        fuse_loras_into_model(
            model,
            [LoRAConfig(str(path))],
            verbose=False,
        )

        expected = mx.array([[3.0, 6.0], [4.0, 8.0]], dtype=mx.float32)
        assert mx.allclose(model.transformer_blocks[0].ff.project_out.weight, expected)

    def test_fuses_non_square_direct_target(self, tmp_path):
        path = tmp_path / "direct_lora.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "transformer_blocks.0.ff.project_out.lora_A.weight": mx.array(
                    [[1.0, 2.0, 3.0]], dtype=mx.float32
                ),
                "transformer_blocks.0.ff.project_out.lora_B.weight": mx.array(
                    [[4.0], [5.0]], dtype=mx.float32
                ),
            },
        )
        model = TinyLoRAModel()
        model.transformer_blocks[0].ff.project_out.weight = mx.zeros(
            (2, 3), dtype=mx.float32
        )

        fuse_loras_into_model(model, [LoRAConfig(str(path))], verbose=False)

        expected = mx.array(
            [[4.0, 8.0, 12.0], [5.0, 10.0, 15.0]], dtype=mx.float32
        )
        assert mx.allclose(model.transformer_blocks[0].ff.project_out.weight, expected)

    def test_fuses_non_square_pretransposed_target(self, tmp_path):
        path = tmp_path / "pretransposed_lora.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "transformer_blocks.0.ff.project_out.lora_A.weight": mx.array(
                    [[1.0, 2.0, 3.0]], dtype=mx.float32
                ),
                "transformer_blocks.0.ff.project_out.lora_B.weight": mx.array(
                    [[4.0], [5.0]], dtype=mx.float32
                ),
            },
        )
        model = TinyLoRAModel()
        del model.transformer_blocks[0].ff.project_out.weight
        model.transformer_blocks[0].ff._project_out_weight_t = mx.zeros(
            (3, 2), dtype=mx.float32
        )

        fuse_loras_into_model(model, [LoRAConfig(str(path))], verbose=False)

        expected = mx.array(
            [[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]], dtype=mx.float32
        )
        assert mx.allclose(
            model.transformer_blocks[0].ff._project_out_weight_t,
            expected,
        )

    def test_fused_range_guard_is_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LTX_LORA_FUSE_RANGE_GUARD", raising=False)
        path = tmp_path / "overflow_lora.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                    [[70000.0]], dtype=mx.float32
                ),
                "transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                    [[1.0]], dtype=mx.float32
                ),
            },
        )
        model = TinyLoRAModel()

        fuse_loras_into_model(model, [LoRAConfig(str(path))], verbose=False)

        assert bool(mx.isinf(model.transformer_blocks[0].attn1.to_q.weight).item())

    def test_rejects_fused_value_outside_target_dtype_when_guard_enabled(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setenv("LTX_LORA_FUSE_RANGE_GUARD", "1")
        path = tmp_path / "overflow_lora.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                    [[70000.0]], dtype=mx.float32
                ),
                "transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                    [[1.0]], dtype=mx.float32
                ),
            },
        )
        model = TinyLoRAModel()

        with pytest.raises(ValueError, match="LoRA fusion overflows"):
            fuse_loras_into_model(model, [LoRAConfig(str(path))], verbose=False)

    def test_restore_state_requires_cache_backed_transformer(self):
        model = TinyLoRAModel()

        with pytest.raises(RuntimeError, match="cache-backed transformer"):
            get_transformer_cache_restore_state(model)

    def test_restore_state_reloads_from_cache_source(self, monkeypatch, tmp_path):
        import LTX_2_MLX.loader.transformer_cache as tc

        model = TinyLoRAModel()
        cache_path = tmp_path / "transformer.safetensors"
        setattr(model, TRANSFORMER_CACHE_RESTORE_ATTR, {
            "valid": True,
            "cache_path": cache_path,
            "transformer_cache_quantize": "off",
            "video_ff_quantize_specs": (("project_out", "mxfp8"),),
            "video_ff_quantize_group_size": 32,
            "video_ff_quantize_bits": 8,
        })
        state = get_transformer_cache_restore_state(model)
        assert state["cache_path"] == cache_path
        assert "parameters" not in state

        calls = {}

        def fake_load_transformer_cache(target, path, **kwargs):
            calls["target"] = target
            calls["path"] = path
            calls["kwargs"] = kwargs
            return (0, 0, 0)

        monkeypatch.setattr(tc, "load_transformer_cache", fake_load_transformer_cache)

        restore_transformer_cache_state(model, state)

        assert calls["target"] is model
        assert calls["path"] == cache_path
        assert calls["kwargs"]["video_ff_quantize_specs"] == (("project_out", "mxfp8"),)
        assert calls["kwargs"]["video_ff_quantize_group_size"] == 32
        assert calls["kwargs"]["video_ff_quantize_bits"] == 8
        assert getattr(model, TRANSFORMER_CACHE_RESTORE_ATTR)["valid"] is True

# ============================================================================
# Integration Tests (require actual weights - marked for skip)
# ============================================================================

@pytest.mark.requires_weights
class TestLoaderIntegration:
    """Integration tests requiring actual model weights."""

    def test_load_transformer_weights_real(self, weights_dir):
        """Check real checkpoint metadata without materializing tensors."""
        weight_files = list(weights_dir.glob("**/*.safetensors"))
        if not weight_files:
            pytest.skip("No weight files found")

        # Just verify we can read the file header without loading all weights
        from LTX_2_MLX.safetensors_header import read_safetensors_header
        header = read_safetensors_header(str(weight_files[0]))
        keys = [k for k in header if k != "__metadata__"]
        assert len(keys) > 0
