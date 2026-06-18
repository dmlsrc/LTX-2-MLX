"""Native Gemma 3 tokenizer: `sentencepiece` on tokenizer.model.

Drop-in replacement for ``transformers.AutoTokenizer`` in the text-encoder path,
backed by the self-contained sentencepiece library. This avoids both transformers
and the HF ``tokenizers`` lib - the latter hard-depends on ``huggingface-hub`` for
a Hub-download feature this path never uses (see
github.com/huggingface/tokenizers/issues/1973). sentencepiece has no Python
dependencies at all.

Token IDs and attention mask are byte-identical to AutoTokenizer for prompt text,
validated across a large corpus (natural language, CJK / RTL / emoji unicode,
punctuation, whitespace, code, casing, truncation). The one intentional
difference: a literal control-token *string* (e.g. ``"<bos>"``,
``"<start_of_turn>"``) is tokenized as ordinary text rather than the special
token. That is safer for user prompt text (no control-token injection) and never
occurs on the prompt path, which deliberately skips chat templates.

Behavior matches the stock LTX-2 ``LTXVGemmaTokenizer``: strip, prepend BOS,
truncate to ``max_length``, left-pad with ``<pad>`` (``pad_to_max=False`` skips
padding).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

_TOKENIZER_FILE = "tokenizer.model"


def _resolve_tokenizer_model(model_path: str | Path) -> Path:
    """Accept a Gemma model directory or a direct tokenizer.model path."""
    p = Path(model_path)
    candidate = p / _TOKENIZER_FILE if p.is_dir() else p
    if not candidate.exists():
        raise FileNotFoundError(
            f"Gemma tokenizer.model not found at {candidate}. Expected a "
            f"tokenizer.model in the Gemma model directory ({model_path})."
        )
    return candidate


class GemmaTokenizer:
    """Encode prompts to ``(input_ids, attention_mask)`` MLX arrays."""

    def __init__(self, model_path: str | Path):
        import sentencepiece  # self-contained; no huggingface-hub / transformers

        self._sp = sentencepiece.SentencePieceProcessor(
            model_file=str(_resolve_tokenizer_model(model_path))
        )
        self.bos_id = self._sp.bos_id() if self._sp.bos_id() >= 0 else 2
        self.eos_id = self._sp.eos_id() if self._sp.eos_id() >= 0 else 1
        self.pad_id = self._sp.pad_id() if self._sp.pad_id() >= 0 else 0

    def encode(
        self, prompt: str, max_length: int = 1024, pad_to_max: bool = True
    ) -> tuple[mx.array, mx.array]:
        """``(input_ids, attention_mask)``, each ``(1, L)`` int32.

        BOS-prefixed and truncated to ``max_length`` (keeping BOS + leading
        content). When ``pad_to_max`` is True (the default) the sequence is
        left-padded with ``<pad>`` so ``L == max_length``; otherwise ``L`` is the
        real token count and the mask is all ones.
        """
        ids = [self.bos_id, *self._sp.encode((prompt or "").strip())]
        ids = ids[:max_length]
        n = len(ids)
        if pad_to_max:
            pad = max_length - n
            ids = [self.pad_id] * pad + ids
            mask = [0] * pad + [1] * n
        else:
            mask = [1] * n
        input_ids = mx.array(ids, dtype=mx.int32)[None, :]
        attention_mask = mx.array(mask, dtype=mx.int32)[None, :]
        return input_ids, attention_mask
