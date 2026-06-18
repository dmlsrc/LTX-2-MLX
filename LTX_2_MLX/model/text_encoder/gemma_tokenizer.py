"""Native Gemma 3 tokenizer: the HF `tokenizers` (Rust) lib on tokenizer.json.

Drop-in replacement for `transformers.AutoTokenizer` in the text-encoder path.
It reads the same ``tokenizer.json`` the fast tokenizer uses, so the token IDs
and attention mask are byte-identical to AutoTokenizer - verified across plain,
empty, whitespace, multiline, and punctuation prompts - without pulling in
transformers and its dependency tree.

Behavior matches the stock LTX-2 ``LTXVGemmaTokenizer``: strip whitespace, prepend
BOS (via the tokenizer.json post-processor), truncate to ``max_length``, and
left-pad to ``max_length`` with ``<pad>`` (``pad_to_max=False`` skips padding).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

_TOKENIZER_FILE = "tokenizer.json"


def _resolve_tokenizer_json(model_path: str | Path) -> Path:
    """Accept a Gemma model directory or a direct tokenizer.json path."""
    p = Path(model_path)
    candidate = p / _TOKENIZER_FILE if p.is_dir() else p
    if not candidate.exists():
        raise FileNotFoundError(
            f"Gemma tokenizer.json not found at {candidate}. Expected a "
            f"tokenizer.json in the Gemma model directory ({model_path})."
        )
    return candidate


class GemmaTokenizer:
    """Encode prompts to ``(input_ids, attention_mask)`` MLX arrays."""

    def __init__(self, model_path: str | Path):
        from tokenizers import Tokenizer  # Rust-backed; no transformers

        self._tok = Tokenizer.from_file(str(_resolve_tokenizer_json(model_path)))
        self.bos_id = self._tok.token_to_id("<bos>")
        self.eos_id = self._tok.token_to_id("<eos>")
        self.pad_id = self._tok.token_to_id("<pad>")
        self.pad_token = "<pad>"
        if self.pad_id is None:  # defensive: fall back to EOS as the pad token
            self.pad_id, self.pad_token = self.eos_id, "<eos>"

    def encode(
        self, prompt: str, max_length: int = 1024, pad_to_max: bool = True
    ) -> tuple[mx.array, mx.array]:
        """``(input_ids, attention_mask)``, each ``(1, L)`` int32.

        BOS-prefixed and truncated to ``max_length``. When ``pad_to_max`` is True
        (the default) the sequence is left-padded with ``<pad>`` so ``L ==
        max_length``; otherwise ``L`` is the real token count and the mask is all
        ones.
        """
        self._tok.enable_truncation(max_length=max_length)
        if pad_to_max:
            self._tok.enable_padding(
                direction="left",
                pad_id=self.pad_id,
                pad_token=self.pad_token,
                length=max_length,
            )
        else:
            self._tok.no_padding()

        enc = self._tok.encode((prompt or "").strip())
        input_ids = mx.array(enc.ids, dtype=mx.int32)[None, :]
        attention_mask = mx.array(enc.attention_mask, dtype=mx.int32)[None, :]
        return input_ids, attention_mask
