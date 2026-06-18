"""Tests for the native Gemma 3 tokenizer (HF tokenizers, no transformers).

Golden IDs are the verified AutoTokenizer output for google/gemma-3-12b-it, so
these assert the wrapper reproduces it without importing transformers. Gated on a
cached gemma-3-12b-it tokenizer.json (skipped when absent).
"""

import os
from pathlib import Path

import mlx.core as mx
import pytest

from LTX_2_MLX.model.text_encoder.gemma_tokenizer import GemmaTokenizer


def _find_gemma_tokenizer_json():
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    for repo in ("models--google--gemma-3-12b-it", "models--mlx-community--gemma-3-12b-it-4bit"):
        hits = sorted((hub / repo).glob("snapshots/*/tokenizer.json"))
        if hits:
            return hits[0]
    hits = sorted(hub.glob("models--*gemma-3*/snapshots/*/tokenizer.json"))
    return hits[0] if hits else None


_TOK = _find_gemma_tokenizer_json()
pytestmark = pytest.mark.skipif(_TOK is None, reason="No cached gemma-3 tokenizer.json")


def _row(a: mx.array) -> list[int]:
    return [int(v) for v in a[0]]


def test_special_token_ids():
    t = GemmaTokenizer(_TOK)
    assert (t.bos_id, t.eos_id, t.pad_id) == (2, 1, 0)


def test_encode_padded_golden():
    ids, mask = GemmaTokenizer(_TOK).encode("kitten", max_length=8)
    assert ids.shape == (1, 8) and mask.shape == (1, 8)
    assert ids.dtype == mx.int32
    assert _row(ids) == [0, 0, 0, 0, 0, 0, 2, 231775]   # left <pad>, then <bos>, "kitten"
    assert _row(mask) == [0, 0, 0, 0, 0, 0, 1, 1]


def test_encode_strips_whitespace():
    ids, _ = GemmaTokenizer(_TOK).encode("   kitten\n ", max_length=8)
    assert _row(ids) == [0, 0, 0, 0, 0, 0, 2, 231775]


def test_encode_unpadded():
    ids, mask = GemmaTokenizer(_TOK).encode("kitten", max_length=8, pad_to_max=False)
    assert _row(ids) == [2, 231775]
    assert _row(mask) == [1, 1]


def test_encode_truncates_to_max_length():
    ids, mask = GemmaTokenizer(_TOK).encode(
        "A red fox jumps over the lazy dog in a snowy forest.", max_length=4
    )
    assert ids.shape == (1, 4)
    assert int(mask.sum()) == 4  # all real tokens, nothing to pad
