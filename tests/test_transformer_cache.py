"""Focused tests for transformer-cache edge cases."""

import json
import struct

import mlx.core as mx
import mlx.nn as nn

import LTX_2_MLX.loader.transformer_cache as tc
from LTX_2_MLX.loader.transformer_cache import (
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
    TransformerBlockStreamer,
    TransformerCacheResult,
    _cache_payload,
    _fp8_scale_companions,
    _iter_fp8_checkpoint_weights,
)


def _write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for key, dtype, shape, data in tensors:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array([0.0], dtype=mx.float32)


def test_fp8_companion_filter_scopes_comfy_quant_to_fp8_weights():
    header_dtypes = {
        "layer.weight": "F8_E4M3",
        "layer.weight_scale": "F32",
        "layer.input_scale": "F32",
        "layer.comfy_quant": "U8",
        "unrelated.comfy_quant": "U8",
    }

    fp8, weight_scales, input_scales, comfy = _fp8_scale_companions(header_dtypes)

    assert fp8 == {"layer.weight"}
    assert weight_scales == {"layer.weight_scale"}
    assert input_scales == {"layer.input_scale"}
    assert comfy == {"layer.comfy_quant"}


def test_fp8_iterator_dequantizes_scaled_e4m3_and_drops_companions(tmp_path):
    path = tmp_path / "fp8.safetensors"
    _write_safetensors(
        path,
        [
            ("model.diffusion_model.layer.weight", "F8_E4M3", [1], bytes([0x38])),
            (
                "model.diffusion_model.layer.weight_scale",
                "F32",
                [1],
                struct.pack("<f", 2.0),
            ),
            (
                "model.diffusion_model.layer.input_scale",
                "F32",
                [1],
                struct.pack("<f", 7.0),
            ),
            (
                "model.diffusion_model.layer.comfy_quant",
                "U8",
                [26],
                b'{"format":"float8_e4m3fn"}',
            ),
        ],
    )

    items = dict(_iter_fp8_checkpoint_weights(str(path), mx.bfloat16))

    assert set(items) == {"model.diffusion_model.layer.weight"}
    assert items["model.diffusion_model.layer.weight"].dtype == mx.bfloat16
    assert float(items["model.diffusion_model.layer.weight"].item()) == 2.0


def test_transformer_cache_payload_keys_quant_group_overrides(tmp_path):
    path = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(path), {"x": mx.array([1.0])})

    common = dict(
        include_audio=False,
        video_ff_layout_specs=(),
        video_ff_layout_layers=(),
        video_attn_layout_specs=(),
        video_attn_layout_layers=(),
        transformer_cache_quantize=TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
    )
    a = _cache_payload(str(path), video_ff_quantize_group_size=32, **common)
    b = _cache_payload(str(path), video_ff_quantize_group_size=64, **common)

    assert a["transformer_cache_quantize_group_size"] == 32
    assert b["transformer_cache_quantize_group_size"] == 64
    assert a != b


def test_load_transformer_weights_cached_threads_quant_specs(monkeypatch, tmp_path):
    cache_file = tmp_path / "transformer.safetensors"
    calls = {}
    model = type("DummyModel", (), {})()

    def fake_ensure_transformer_cache(weights_path, **kwargs):
        calls["ensure"] = kwargs
        return TransformerCacheResult(
            cache_path=cache_file,
            rebuilt=False,
            loaded_count=0,
            layout_count=0,
        )

    def fake_load_transformer_cache(model, path, **kwargs):
        calls["load"] = kwargs
        assert path == cache_file
        return (1, 0, 1)

    monkeypatch.setattr(tc, "ensure_transformer_cache", fake_ensure_transformer_cache)
    monkeypatch.setattr(tc, "load_transformer_cache", fake_load_transformer_cache)

    specs = (("project_out", "mxfp8"),)
    layers = (1, 3)
    tc.load_transformer_weights_cached(
        model,
        str(tmp_path / "weights.safetensors"),
        cache_mode="auto",
        cache_root=None,
        include_audio=False,
        video_ff_layout_specs=(),
        video_ff_layout_layers=(),
        video_attn_layout_specs=(),
        video_attn_layout_layers=(),
        transformer_cache_quantize=TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
        video_ff_quantize_specs=specs,
        video_ff_quantize_layers=layers,
        video_ff_quantize_group_size=32,
        video_ff_quantize_bits=8,
    )

    assert calls["ensure"]["video_ff_quantize_specs"] == specs
    assert calls["ensure"]["video_ff_quantize_layers"] == layers
    assert calls["load"]["video_ff_quantize_specs"] == specs
    assert calls["load"]["video_ff_quantize_group_size"] == 32
    assert calls["load"]["video_ff_quantize_bits"] == 8
    assert model._lora_restore_cache_source == {
        "valid": True,
        "cache_path": cache_file,
        "transformer_cache_quantize": TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
        "video_ff_quantize_specs": specs,
        "video_ff_quantize_group_size": 32,
        "video_ff_quantize_bits": 8,
        "persistent_loras": (),
    }


def test_block_streamer_reloads_from_sharded_cache_after_eviction(tmp_path):
    cache_file = tmp_path / "transformer.safetensors"
    mx.save_safetensors(
        str(tmp_path / "transformer-00000.safetensors"),
        {"transformer_blocks.0.weight": mx.array([1.0], dtype=mx.float32)},
    )
    mx.save_safetensors(
        str(tmp_path / "transformer-00001.safetensors"),
        {"transformer_blocks.1.weight": mx.array([2.0], dtype=mx.float32)},
    )

    streamer = TransformerBlockStreamer(cache_file)
    streamer.bind(TinyBlock(), 0, evict_block_idx=1)
    block = streamer.bind(TinyBlock(), 1)

    assert float(block.weight.item()) == 2.0
