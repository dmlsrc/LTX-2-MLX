"""Tests for native safetensors header reading (no safetensors lib)."""

import mlx.core as mx

from LTX_2_MLX.safetensors_header import (
    read_safetensors_dtypes,
    read_safetensors_header,
    read_safetensors_metadata,
)


def test_metadata_and_dtypes_roundtrip(tmp_path):
    p = tmp_path / "t.safetensors"
    arrays = {"w": mx.zeros((2, 3), dtype=mx.float32), "b": mx.ones((3,), dtype=mx.bfloat16)}
    mx.save_safetensors(str(p), arrays, metadata={"model_version": "2.3.0", "config": '{"x": 1}'})

    meta = read_safetensors_metadata(p)
    assert meta["model_version"] == "2.3.0"
    assert meta["config"] == '{"x": 1}'

    dtypes = read_safetensors_dtypes(p)
    assert set(dtypes) == {"w", "b"}
    assert dtypes["w"] == "F32"
    assert dtypes["b"] == "BF16"

    header = read_safetensors_header(p)
    assert "w" in header and "b" in header


def test_metadata_empty_when_absent(tmp_path):
    p = tmp_path / "n.safetensors"
    mx.save_safetensors(str(p), {"w": mx.zeros((1,))})
    assert read_safetensors_metadata(p) == {}
