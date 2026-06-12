"""Disposable cache for converted transformer and auxiliary weight families."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .weight_converter import _flatten_to_nested, convert_pytorch_key_to_mlx


# Schema versions for cache invalidation.  Transformer and family caches
# are split so a family-layout bump (cheap rebuild) doesn't drag the
# ~22 GB transformer cache.  The cache digest also folds layout/quant/dtype
# specs in, so most meaningful changes invalidate without a version bump.
CACHE_SCHEMA_VERSION = 1
FAMILY_CACHE_SCHEMA_VERSION = 2  # v2: video_vae + audio_vae Conv weights baked channels-last
LAYOUT_KEY_PREFIX = "__layout__."
QUANT_KEY_PREFIX = "__quant__."
TRANSFORMER_CACHE_QUANTIZE_OFF = "off"
TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS = "mxfp8-blocks"
TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE = "mxfp8-blocks-pretranspose"
TRANSFORMER_CACHE_QUANTIZE_MODES = (
    TRANSFORMER_CACHE_QUANTIZE_OFF,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)
_ATTENTION_QUANT_PROJECTIONS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "to_gate_logits",
)
_BLOCK_ATTENTION_MODULES = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)
_CACHE_QUANTIZED_BLOCK_LINEAR_BASES = tuple(
    f"{attn}.{projection}"
    for attn in _BLOCK_ATTENTION_MODULES
    for projection in _ATTENTION_QUANT_PROJECTIONS
) + (
    "ff.project_in.proj",
    "ff.project_out",
    "audio_ff.project_in.proj",
    "audio_ff.project_out",
)
_PRETRANSPOSED_CACHE_QUANTIZE_MODES = (
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)
WEIGHT_FAMILIES = ("connector", "video_vae", "audio_vae", "vocoder")
WEIGHT_FAMILY_FILENAMES = {
    "connector": "connector.safetensors",
    "video_vae": "video_vae.safetensors",
    "audio_vae": "audio_vae.safetensors",
    "vocoder": "vocoder.safetensors",
}
WEIGHT_FAMILY_LABELS = {
    "connector": "Connector",
    "video_vae": "Video VAE",
    "audio_vae": "Audio VAE",
    "vocoder": "Vocoder",
}


@dataclass(frozen=True)
class TransformerCacheResult:
    """Result metadata for a transformer cache load."""

    cache_path: Path
    rebuilt: bool
    loaded_count: int
    layout_count: int
    quant_count: int = 0


class _PretransposedQuantizedLinear(nn.Module):
    """Quantized linear whose packed weight was built from ``weight.T``."""

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=False,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            x = x + self["bias"]
        return x


_QUANTIZED_LINEAR_TYPES = (nn.QuantizedLinear, _PretransposedQuantizedLinear)


class TransformerBlockStreamer:
    """Bind cached transformer-block weights into a small resident block pool."""

    def __init__(
        self,
        cache_file: Path,
        *,
        transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
        video_ff_quantize_specs: Tuple[Tuple[str, str], ...] = (),
        video_ff_quantize_group_size: int | None = None,
        video_ff_quantize_bits: int | None = None,
    ) -> None:
        self.cache_file = cache_file
        self.transformer_cache_quantize = transformer_cache_quantize
        self.video_ff_quantize_specs = tuple(video_ff_quantize_specs)
        self.video_ff_quantize_group_size = video_ff_quantize_group_size
        self.video_ff_quantize_bits = video_ff_quantize_bits
        self._weights = mx.load(str(cache_file))
        self._block_keys: dict[int, list[tuple[str, str]]] = {}
        self._layout_keys: dict[int, list[tuple[str, str]]] = {}
        self._quant_keys: dict[int, list[tuple[str, str]]] = {}

        for full_key in list(self._weights):
            is_layout = full_key.startswith(LAYOUT_KEY_PREFIX)
            is_quant = full_key.startswith(QUANT_KEY_PREFIX)
            if is_layout:
                logical_key = full_key[len(LAYOUT_KEY_PREFIX) :]
            elif is_quant:
                logical_key = full_key[len(QUANT_KEY_PREFIX) :]
            else:
                logical_key = full_key
            parts = logical_key.split(".")
            if len(parts) < 3 or parts[0] != "transformer_blocks":
                self._weights.pop(full_key, None)
                continue

            try:
                block_idx = int(parts[1])
            except ValueError:
                self._weights.pop(full_key, None)
                continue

            block_key = ".".join(parts[2:])
            if is_layout:
                target = self._layout_keys
            elif is_quant:
                target = self._quant_keys
            else:
                target = self._block_keys
            target.setdefault(block_idx, []).append((full_key, block_key))

        discovered = set(self._block_keys) | set(self._layout_keys) | set(self._quant_keys)
        if not discovered:
            raise ValueError(f"No transformer block weights found in cache {cache_file}")
        self.block_count = max(discovered) + 1
        missing = [idx for idx in range(self.block_count) if idx not in discovered]
        if missing:
            raise ValueError(
                f"Transformer cache {cache_file} is missing block weights for layers {missing}"
            )
        self.loaded_count = sum(len(items) for items in self._block_keys.values()) + sum(
            len(items) for items in self._layout_keys.values()
        ) + sum(
            len(items) for items in self._quant_keys.values()
        )
        self.layout_count = sum(len(items) for items in self._layout_keys.values())
        self.quant_count = sum(len(items) for items in self._quant_keys.values())

    def bind(
        self,
        block: nn.Module,
        block_idx: int,
        *,
        evict_block_idx: int | None = None,
    ) -> nn.Module:
        """Load one block's cached weights into ``block`` and return it."""
        if block_idx < 0 or block_idx >= self.block_count:
            raise IndexError(f"block index {block_idx} is outside 0-{self.block_count - 1}")

        if evict_block_idx is not None and evict_block_idx != block_idx:
            for full_key, _ in self._block_keys.get(evict_block_idx, ()):
                self._weights.pop(full_key, None)
            for full_key, _ in self._layout_keys.get(evict_block_idx, ()):
                self._weights.pop(full_key, None)
            for full_key, _ in self._quant_keys.get(evict_block_idx, ()):
                self._weights.pop(full_key, None)

        normal_keys = self._block_keys.get(block_idx, ())
        layout_keys = self._layout_keys.get(block_idx, ())
        quant_keys = self._quant_keys.get(block_idx, ())
        sample_key = None
        if normal_keys:
            sample_key = normal_keys[0][0]
        elif layout_keys:
            sample_key = layout_keys[0][0]
        elif quant_keys:
            sample_key = quant_keys[0][0]
        if sample_key is not None and sample_key not in self._weights:
            self._weights = mx.load(str(self.cache_file))
            self._drop_non_block_keys()

        quant_bases = _quant_bases_for_block_keys(quant_keys)
        _restore_block_quantized_linears(block, keep_bases=quant_bases)
        _clear_block_layout_weights(block)
        if quant_bases:
            _prepare_block_quantized_linears(
                block,
                quant_bases,
                quant_keys,
                normal_keys,
                transformer_cache_quantize=self.transformer_cache_quantize,
                quantization_specs=self.video_ff_quantize_specs,
                group_size=self.video_ff_quantize_group_size,
                bits=self.video_ff_quantize_bits,
            )
        normal_weights = {
            block_key: self._weights[full_key]
            for full_key, block_key in normal_keys
        }
        if normal_weights:
            block.update(_flatten_to_nested(normal_weights))
        quant_weights = {
            block_key: self._weights[full_key]
            for full_key, block_key in quant_keys
        }
        if quant_weights:
            block.update(_flatten_to_nested(quant_weights))
        for full_key, layout_key in layout_keys:
            _install_block_layout_weight(block, layout_key, self._weights[full_key])
        if hasattr(block, "idx"):
            block.idx = block_idx
        return block

    def close(self) -> None:
        self._weights = {}
        self._block_keys = {}
        self._layout_keys = {}
        self._quant_keys = {}

    def _drop_non_block_keys(self) -> None:
        for full_key in list(self._weights):
            if full_key.startswith(LAYOUT_KEY_PREFIX):
                logical_key = full_key[len(LAYOUT_KEY_PREFIX) :]
            elif full_key.startswith(QUANT_KEY_PREFIX):
                logical_key = full_key[len(QUANT_KEY_PREFIX) :]
            else:
                logical_key = full_key
            if not logical_key.startswith("transformer_blocks."):
                self._weights.pop(full_key, None)


@dataclass(frozen=True)
class WeightFamilyCacheResult:
    """Result metadata for auxiliary weight family cache loads."""

    cache_paths: dict[str, Path]
    rebuilt: bool
    loaded_count: int


def default_transformer_cache_root() -> Path:
    """Return the default location for disposable weight cache artifacts.

    Resolution order:
    1. ``LTX_MLX_WEIGHTS_CACHE_DIR`` (preferred name)
    2. ``LTX_MLX_TRANSFORMER_CACHE_DIR`` (legacy alias, still honored)
    3. ``~/.cache/ltx-2-mlx/weights-cache``
    """
    env_path = os.environ.get("LTX_MLX_WEIGHTS_CACHE_DIR")
    if not env_path:
        env_path = os.environ.get("LTX_MLX_TRANSFORMER_CACHE_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".cache" / "ltx-2-mlx" / "weights-cache"


def _file_signature(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _canonical_specs(specs: Tuple[Tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"target": target, "layout": layout} for target, layout in specs]


def _canonical_quant_specs(specs: Tuple[Tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"target": target, "mode": mode} for target, mode in specs]


_VIDEO_FF_KEY_PATTERNS = ("ff.project_in.proj", "ff.project_out")
_AUDIO_FF_KEY_PATTERNS = ("audio_ff.project_in.proj", "audio_ff.project_out")


def _ff_cache_dtype_for_key(
    mlx_key: str,
    *,
    video_ff_dtype: Optional[mx.Dtype],
    audio_ff_dtype: Optional[mx.Dtype] = None,
) -> Optional[mx.Dtype]:
    """Return the target cache dtype for a video / audio FF projection key.

    Casts both `.weight` and `.bias` keys belonging to the targeted FF
    projections.  Attention projections, RMSNorm, AdaLN, etc. return
    None (kept at checkpoint dtype).  Attention dtype was tried (see
    PERFORMANCE_NOTES.md) and dropped — at the SDPA boundary the in/out
    cast overhead outweighs the FP16 matmul win, and attn projections
    do not hit the kernel-selection cliff that FF project_out does at
    K=16384.

    Audio FF is independent of video FF: the bench in
    `scripts/bench_pretranspose_dtype.py` showed audio K=8192 has no
    cliff (FP16 naive is FASTER than BF16 naive there) and per-call
    savings are sub-millisecond.  Provided for completeness / real-world
    A/B; expected savings are far below noise on bakery-class workloads.
    """
    if video_ff_dtype is None and audio_ff_dtype is None:
        return None
    # Strip the trailing suffix so `.weight` and `.bias` both match.
    base = mlx_key.rsplit(".", 1)[0]
    if not base.startswith("transformer_blocks."):
        return None
    parts = base.split(".", 2)
    if len(parts) < 3:
        return None
    suffix = parts[2]  # e.g. "ff.project_in.proj" or "audio_ff.project_in.proj"
    if video_ff_dtype is not None:
        for pat in _VIDEO_FF_KEY_PATTERNS:
            if suffix == pat:
                return video_ff_dtype
    if audio_ff_dtype is not None:
        for pat in _AUDIO_FF_KEY_PATTERNS:
            if suffix == pat:
                return audio_ff_dtype
    return None


def _dtype_to_payload_name(dtype: Optional[mx.Dtype]) -> Optional[str]:
    """Map an MLX dtype to a JSON-friendly name for the payload digest."""
    if dtype is None:
        return None
    if dtype == mx.bfloat16:
        return "bfloat16"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.float32:
        return "float32"
    return str(dtype)


def _cache_payload(
    weights_path: str,
    *,
    transformer_dtype: Optional[mx.Dtype] = None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    audio_ff_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_ff_layout_layers: Tuple[int, ...] = (),
    audio_attn_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_attn_layout_layers: Tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...] = (),
    video_ff_quantize_layers: Tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: Optional[mx.Dtype] = None,
    audio_ff_dtype: Optional[mx.Dtype] = None,
) -> dict[str, Any]:
    if transformer_cache_quantize not in TRANSFORMER_CACHE_QUANTIZE_MODES:
        raise ValueError(f"Unsupported transformer cache quantization mode: {transformer_cache_quantize}")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": _file_signature(weights_path),
        "include_audio": include_audio,
        "video_ff_layout_specs": _canonical_specs(video_ff_layout_specs),
        "video_ff_layout_layers": list(video_ff_layout_layers),
        "video_attn_layout_specs": _canonical_specs(video_attn_layout_specs),
        "video_attn_layout_layers": list(video_attn_layout_layers),
    }
    # Only include audio layout keys in the payload when present, so existing
    # audio-pretranspose-free caches keep their hash and stay valid.
    if audio_ff_layout_specs:
        payload["audio_ff_layout_specs"] = _canonical_specs(audio_ff_layout_specs)
        payload["audio_ff_layout_layers"] = list(audio_ff_layout_layers)
    if audio_attn_layout_specs:
        payload["audio_attn_layout_specs"] = _canonical_specs(audio_attn_layout_specs)
        payload["audio_attn_layout_layers"] = list(audio_attn_layout_layers)
    if adaln_pretranspose:
        payload["adaln_pretranspose"] = True
    if transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF:
        payload["transformer_cache_quantize"] = transformer_cache_quantize
    if video_ff_quantize_specs:
        payload.update({
            "video_ff_quantize_specs": _canonical_quant_specs(video_ff_quantize_specs),
            "video_ff_quantize_layers": list(video_ff_quantize_layers),
            "video_ff_quantize_group_size": video_ff_quantize_group_size,
            "video_ff_quantize_bits": video_ff_quantize_bits,
        })
    # Only include dtype keys when non-default — keeps existing BF16 caches
    # valid (their hash doesn't include these keys).
    if video_ff_dtype is not None and video_ff_dtype != mx.bfloat16:
        payload["video_ff_dtype"] = _dtype_to_payload_name(video_ff_dtype)
    if audio_ff_dtype is not None and audio_ff_dtype != mx.bfloat16:
        payload["audio_ff_dtype"] = _dtype_to_payload_name(audio_ff_dtype)
    if transformer_dtype is not None and transformer_dtype != mx.bfloat16:
        payload["transformer_dtype"] = _dtype_to_payload_name(transformer_dtype)
    return payload


def _source_payload(weights_path: str, *, kind: str) -> dict[str, Any]:
    return {
        "schema_version": FAMILY_CACHE_SCHEMA_VERSION,
        "source": _file_signature(weights_path),
        "kind": kind,
    }


def _source_dir_payload(weights_path: str) -> dict[str, Any]:
    return {
        "schema_version": FAMILY_CACHE_SCHEMA_VERSION,
        "source": _file_signature(weights_path),
    }


def _payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _safe_stem(path: str) -> str:
    stem = Path(path).stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in stem)
    return safe[:80] or "transformer"


def transformer_cache_paths(
    weights_path: str,
    cache_root: str | None,
    *,
    transformer_dtype: Optional[mx.Dtype] = None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    audio_ff_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_ff_layout_layers: Tuple[int, ...] = (),
    audio_attn_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_attn_layout_layers: Tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...] = (),
    video_ff_quantize_layers: Tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: Optional[mx.Dtype] = None,
    audio_ff_dtype: Optional[mx.Dtype] = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve cache artifact paths and expected metadata payload."""
    payload = _cache_payload(
        weights_path,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        adaln_pretranspose=adaln_pretranspose,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
        transformer_dtype=transformer_dtype,
    )
    root = Path(cache_root).expanduser() if cache_root else default_transformer_cache_root()
    cache_dir = root / f"{_safe_stem(weights_path)}-{_payload_digest(payload)}"
    return cache_dir / "transformer.safetensors", cache_dir / "metadata.json", payload


def weight_family_cache_paths(
    weights_path: str,
    cache_root: str | None,
    family: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve source-level cache paths and metadata for one weight family."""
    if family not in WEIGHT_FAMILIES:
        raise ValueError(f"Unsupported weight family: {family}")

    payload = _source_payload(weights_path, kind=family)
    dir_payload = _source_dir_payload(weights_path)
    root = Path(cache_root).expanduser() if cache_root else default_transformer_cache_root()
    cache_dir = root / f"{_safe_stem(weights_path)}-{_payload_digest(dir_payload)}"
    cache_file = cache_dir / WEIGHT_FAMILY_FILENAMES[family]
    metadata_file = cache_dir / f"{Path(cache_file).stem}.metadata.json"
    return cache_file, metadata_file, payload


def _selected_layers(layers: Tuple[int, ...], num_layers: int = 48) -> set[int]:
    return set(layers) if layers else set(range(num_layers))


def _has_spec(specs: Tuple[Tuple[str, str], ...], target: str) -> bool:
    return (target, "pretranspose") in specs


def _quant_mode_for_target(
    specs: Tuple[Tuple[str, str], ...],
    target: str,
) -> str | None:
    for spec_target, mode in specs:
        if spec_target == target:
            return mode
    return None


def _quant_defaults(
    mode: str,
    group_size: int | None,
    bits: int | None,
) -> tuple[int, int]:
    defaults = {
        "affine": (64, 4),
        "mxfp4": (32, 4),
        "mxfp8": (32, 8),
        "nvfp4": (16, 4),
    }
    if mode not in defaults:
        raise ValueError(f"Unsupported FF quantization mode: {mode}")
    default_group_size, default_bits = defaults[mode]
    return group_size or default_group_size, bits or default_bits


def _block_linear_base_for_key(mlx_key: str) -> tuple[int, str, str] | None:
    parts = mlx_key.split(".")
    if len(parts) < 5 or parts[0] != "transformer_blocks" or parts[-1] != "weight":
        return None

    try:
        layer = int(parts[1])
    except ValueError:
        return None

    base = ".".join(parts[2:-1])
    if base not in _CACHE_QUANTIZED_BLOCK_LINEAR_BASES:
        return None
    return layer, base, parts[-1]


def _cache_quant_mode_for_key(
    mlx_key: str,
    *,
    transformer_cache_quantize: str,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...],
    video_ff_quantize_layers: Tuple[int, ...],
) -> str | None:
    if transformer_cache_quantize not in TRANSFORMER_CACHE_QUANTIZE_MODES:
        raise ValueError(f"Unsupported transformer cache quantization mode: {transformer_cache_quantize}")

    block_linear = _block_linear_base_for_key(mlx_key)
    if (
        transformer_cache_quantize in (
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
        )
        and block_linear is not None
    ):
        return "mxfp8"

    return _ff_quant_mode_for_key(
        mlx_key,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
    )


def _cache_quant_pretransposed(transformer_cache_quantize: str) -> bool:
    return transformer_cache_quantize in _PRETRANSPOSED_CACHE_QUANTIZE_MODES


def _ff_quant_mode_for_key(
    mlx_key: str,
    *,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...],
    video_ff_quantize_layers: Tuple[int, ...],
) -> str | None:
    if not video_ff_quantize_specs:
        return None

    block_linear = _block_linear_base_for_key(mlx_key)
    if block_linear is None:
        return None

    layer, base, _param = block_linear

    if layer not in _selected_layers(video_ff_quantize_layers):
        return None

    target_specs = {
        "project_in": "ff.project_in.proj",
        "project_out": "ff.project_out",
    }
    for target, expected_base in target_specs.items():
        mode = _quant_mode_for_target(video_ff_quantize_specs, target)
        if mode is not None and base == expected_base:
            return mode
    return None


# Top-level (non-transformer_blocks) AdaLayerNormSingle paths whose ``linear``
# weights benefit from pretranspose.  When ``adaln_pretranspose`` is enabled,
# the cache build emits a layout entry for each present key.
_ADALN_TOP_LEVEL_LINEARS = (
    "adaln_single.linear",
    "audio_adaln_single.linear",
    "prompt_adaln_single.linear",
    "audio_prompt_adaln_single.linear",
    "av_ca_video_scale_shift_adaln_single.linear",
    "av_ca_audio_scale_shift_adaln_single.linear",
    "av_ca_a2v_gate_adaln_single.linear",
    "av_ca_v2a_gate_adaln_single.linear",
)


def _layout_cache_key(
    mlx_key: str,
    *,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    audio_ff_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_ff_layout_layers: Tuple[int, ...] = (),
    audio_attn_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_attn_layout_layers: Tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
) -> str | None:
    parts = mlx_key.split(".")
    if not parts or parts[-1] != "weight":
        return None

    # Top-level AdaLayerNormSingle.linear keys (no per-layer index).
    if adaln_pretranspose and parts[0] != "transformer_blocks":
        base = ".".join(parts[:-1])  # strip ".weight"
        if base in _ADALN_TOP_LEVEL_LINEARS:
            return f"{base}.weight_t"

    if len(parts) < 5 or parts[0] != "transformer_blocks":
        return None

    try:
        layer = int(parts[1])
    except ValueError:
        return None

    suffix = ".".join(parts[2:])
    ff_layers = _selected_layers(video_ff_layout_layers)
    attn_layers = _selected_layers(video_attn_layout_layers)
    audio_ff_layers = _selected_layers(audio_ff_layout_layers)
    audio_attn_layers = _selected_layers(audio_attn_layout_layers)

    if layer in ff_layers:
        if _has_spec(video_ff_layout_specs, "project_in") and suffix == "ff.project_in.proj.weight":
            return f"transformer_blocks.{layer}.ff.project_in.proj.weight_t"
        if _has_spec(video_ff_layout_specs, "project_out") and suffix == "ff.project_out.weight":
            return f"transformer_blocks.{layer}.ff.project_out.weight_t"

    _VIDEO_ATTN_MODULES = ("attn1", "attn2", "audio_to_video_attn")
    _AUDIO_ATTN_MODULES = ("audio_attn1", "audio_attn2", "video_to_audio_attn")
    _PROJ_TARGETS = ("to_out", "to_q", "to_k", "to_v", "to_gate_logits")

    if layer in attn_layers:
        for proj in _PROJ_TARGETS:
            if not _has_spec(video_attn_layout_specs, proj):
                continue
            for module in _VIDEO_ATTN_MODULES:
                if suffix == f"{module}.{proj}.weight":
                    return f"transformer_blocks.{layer}.{module}.{proj}.weight_t"

    # Audio-side layouts (V2/AV models only — no-op when keys aren't present
    # in the source checkpoint).
    if layer in audio_ff_layers:
        if _has_spec(audio_ff_layout_specs, "project_in") and suffix == "audio_ff.project_in.proj.weight":
            return f"transformer_blocks.{layer}.audio_ff.project_in.proj.weight_t"
        if _has_spec(audio_ff_layout_specs, "project_out") and suffix == "audio_ff.project_out.weight":
            return f"transformer_blocks.{layer}.audio_ff.project_out.weight_t"

    if layer in audio_attn_layers:
        for proj in _PROJ_TARGETS:
            if not _has_spec(audio_attn_layout_specs, proj):
                continue
            for module in _AUDIO_ATTN_MODULES:
                if suffix == f"{module}.{proj}.weight":
                    return f"transformer_blocks.{layer}.{module}.{proj}.weight_t"

    return None


def _metadata_matches(metadata_path: Path, expected_payload: dict[str, Any]) -> bool:
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected_payload


def _weight_family_for_key(key: str) -> str | None:
    """Return the auxiliary weight family for a stock checkpoint key."""
    if key.startswith("vae."):
        return "video_vae"
    if key.startswith("audio_vae."):
        return "audio_vae"
    if key.startswith("vocoder."):
        return "vocoder"
    if key.startswith("text_embedding_projection."):
        return "connector"
    if key.startswith(
        (
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
            "model.diffusion_model.embeddings_connector.",
        )
    ):
        return "connector"
    return None


def _write_metadata(metadata_file: Path, payload: dict[str, Any]) -> None:
    tmp_metadata = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    with tmp_metadata.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_metadata, metadata_file)


def _bake_conv_layout_for_family(family: str, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Transpose Conv weights to MLX channels-last layout in-place at cache build.

    PyTorch Conv layouts are stored channels-second (NCHW / NCDHW), but
    MLX's ``mx.conv2d`` / ``mx.conv3d`` kernels expect channels-last
    (NHWC / NDHWC).  Historically each VAE/audio_vae loader called
    ``value.transpose(...)`` at load time on every process start; that
    transpose was a metadata-only view in MLX but the first conv eval
    had to materialize the contiguous channels-last buffer.  Baking
    the transpose into the cache build moves that one-time
    materialization off the hot path so subsequent runs load straight
    into the right layout.

    Bakes both ``video_vae`` (5D Conv3d weights, NCDHW -> NDHWC) and
    ``audio_vae`` (4D Conv2d weights, NCHW -> NHWC).  Identified by
    tensor rank since every 5D weight in ``vae.*`` is a Conv3d weight
    and every 4D weight in ``audio_vae.*`` is a Conv2d weight (the
    only other tensors in those families are 0D/1D scalars or
    per-channel statistics).

    Vocoder 3D weights are NOT baked because Conv1d
    ((O, I, k) -> (O, k, I), ``transpose(0, 2, 1)``) and
    ConvTranspose1d ((I, O, k) -> (O, k, I), ``transpose(1, 2, 0)``)
    need different transposes that can't be picked apart from a key
    name alone; the vocoder load path still does the right transpose
    at runtime per layer.

    ``mx.contiguous`` materializes the rearranged bytes so the
    safetensors file on disk holds the channels-last layout directly,
    not a strided view of the channels-second source.
    """
    if family == "video_vae":
        target_ndim = 5
        perm = (0, 2, 3, 4, 1)  # (O, I, T, H, W) -> (O, T, H, W, I)
    elif family == "audio_vae":
        target_ndim = 4
        perm = (0, 2, 3, 1)  # (O, I, H, W) -> (O, H, W, I)
    else:
        return weights

    converted: Dict[str, mx.array] = {}
    for key, value in weights.items():
        if value.ndim == target_ndim and key.endswith(".weight"):
            value = mx.contiguous(value.transpose(*perm))
        converted[key] = value
    return converted


def _save_weight_family_cache(
    family: str,
    cache_file: Path,
    metadata_file: Path,
    payload: dict[str, Any],
    weights: Dict[str, mx.array],
) -> int:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    weights = _bake_conv_layout_for_family(family, weights)
    tmp_cache = cache_file.parent / f".{cache_file.stem}.tmp.safetensors"
    mx.save_safetensors(str(tmp_cache), weights)
    os.replace(tmp_cache, cache_file)
    _write_metadata(metadata_file, payload)
    loaded_count = len(weights)
    print(f"  Built {WEIGHT_FAMILY_LABELS[family]} cache: {loaded_count} tensors")
    return loaded_count


def build_weight_family_caches(
    weights_path: str,
    cache_root: str | None,
    families: tuple[str, ...],
) -> int:
    """Build one or more source-level auxiliary weight caches in a single pass."""
    families = tuple(dict.fromkeys(families))
    for family in families:
        if family not in WEIGHT_FAMILIES:
            raise ValueError(f"Unsupported weight family: {family}")

    raw_weights = mx.load(weights_path)
    buckets: dict[str, Dict[str, mx.array]] = {family: {} for family in families}
    for key, value in raw_weights.items():
        family = _weight_family_for_key(key)
        if family in buckets:
            buckets[family][key] = value
    del raw_weights
    gc.collect()

    loaded_count = 0
    for family, family_weights in buckets.items():
        cache_file, metadata_file, payload = weight_family_cache_paths(
            weights_path,
            cache_root,
            family,
        )
        loaded_count += _save_weight_family_cache(
            family,
            cache_file,
            metadata_file,
            payload,
            family_weights,
        )
    return loaded_count


def build_transformer_cache(
    weights_path: str,
    cache_file: Path,
    metadata_file: Path,
    payload: dict[str, Any],
    *,
    transformer_dtype: Optional[mx.Dtype] = None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    audio_ff_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_ff_layout_layers: Tuple[int, ...] = (),
    audio_attn_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_attn_layout_layers: Tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...] = (),
    video_ff_quantize_layers: Tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: Optional[mx.Dtype] = None,
    audio_ff_dtype: Optional[mx.Dtype] = None,
) -> tuple[int, int, int]:
    """Build a converted transformer-only cache from a stock checkpoint."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    raw_weights = mx.load(weights_path)
    cache_weights: Dict[str, mx.array] = {}
    loaded_count = 0
    layout_count = 0
    quant_count = 0
    skipped_count = 0

    for pytorch_key, value in raw_weights.items():
        if not pytorch_key.startswith("model.diffusion_model."):
            continue

        key = pytorch_key.replace("model.diffusion_model.", "")
        mlx_key = convert_pytorch_key_to_mlx(key, include_audio=include_audio)
        if mlx_key is None:
            skipped_count += 1
            continue

        quant_mode = _cache_quant_mode_for_key(
            mlx_key,
            transformer_cache_quantize=transformer_cache_quantize,
            video_ff_quantize_specs=video_ff_quantize_specs,
            video_ff_quantize_layers=video_ff_quantize_layers,
        )
        if quant_mode is not None:
            quant_key = f"{QUANT_KEY_PREFIX}{mlx_key}"
            quant_value = (
                mx.contiguous(value.T)
                if _cache_quant_pretransposed(transformer_cache_quantize)
                else value
            )
            # Whole-transformer dtype: quantize from the target dtype so the
            # produced scales/biases match it (BF16 scales against FP16
            # activations would reintroduce mixed-dtype promotion).
            if transformer_dtype is not None and quant_value.dtype == mx.bfloat16:
                quant_value = quant_value.astype(transformer_dtype)
            group_size, bits = _quant_defaults(
                quant_mode,
                video_ff_quantize_group_size,
                video_ff_quantize_bits,
            )
            q_weight, q_scales, *q_biases = mx.quantize(
                quant_value,
                group_size,
                bits,
                mode=quant_mode,
            )
            base_key = quant_key[: -len(".weight")]
            cache_weights[f"{base_key}.weight"] = q_weight
            cache_weights[f"{base_key}.scales"] = q_scales
            quant_count += 2
            if q_biases:
                cache_weights[f"{base_key}.biases"] = q_biases[0]
                quant_count += 1
            loaded_count += 1
            continue

        layout_key = _layout_cache_key(
            mlx_key,
            video_ff_layout_specs=video_ff_layout_specs,
            video_ff_layout_layers=video_ff_layout_layers,
            video_attn_layout_specs=video_attn_layout_specs,
            video_attn_layout_layers=video_attn_layout_layers,
            audio_ff_layout_specs=audio_ff_layout_specs,
            audio_ff_layout_layers=audio_ff_layout_layers,
            audio_attn_layout_specs=audio_attn_layout_specs,
            audio_attn_layout_layers=audio_attn_layout_layers,
            adaln_pretranspose=adaln_pretranspose,
        )
        # When `--video-ff-dtype` / `--audio-ff-dtype` is set, cast the
        # cached tensor BEFORE writing so the on-disk safetensors holds
        # the target dtype.  Subsequent runs load the pretransposed
        # FP16 weight directly — no BF16 copy ever lives in RAM.
        # Applies to both the pretransposed `.weight_t` tensors and to
        # the plain `.bias` tensors that flow through the fall-through
        # branch.  `transformer_dtype` extends the same baking to every
        # remaining checkpoint-dtype (BF16) tensor, so a non-BF16
        # transformer loads its weights directly with no load-time cast.
        target_dtype = _ff_cache_dtype_for_key(
            mlx_key,
            video_ff_dtype=video_ff_dtype,
            audio_ff_dtype=audio_ff_dtype,
        )
        if (
            target_dtype is None
            and transformer_dtype is not None
            and value.dtype == mx.bfloat16
        ):
            target_dtype = transformer_dtype
        if layout_key is not None:
            stored = mx.contiguous(value.T)
            if target_dtype is not None and stored.dtype != target_dtype:
                stored = stored.astype(target_dtype)
            cache_weights[f"{LAYOUT_KEY_PREFIX}{layout_key}"] = stored
            layout_count += 1
        else:
            if target_dtype is not None and value.dtype != target_dtype:
                value = value.astype(target_dtype)
            cache_weights[mlx_key] = value
        loaded_count += 1

    del raw_weights
    gc.collect()

    tmp_cache = cache_file.parent / f".{cache_file.stem}.tmp.safetensors"
    mx.save_safetensors(str(tmp_cache), cache_weights)
    os.replace(tmp_cache, cache_file)

    _write_metadata(metadata_file, payload)

    print(
        f"  Built transformer cache: {loaded_count} tensors "
        f"({layout_count} pretransposed, {quant_count} quantized, "
        f"skipped {skipped_count})"
    )
    return loaded_count, layout_count, quant_count


def _clear_block_layout_weights(block: nn.Module) -> None:
    ff = getattr(block, "ff", None)
    if ff is not None:
        if hasattr(ff, "_project_in_weight_t"):
            ff._project_in_weight_t = None
        if hasattr(ff, "_project_out_weight_t"):
            ff._project_out_weight_t = None

    audio_ff = getattr(block, "audio_ff", None)
    if audio_ff is not None:
        if hasattr(audio_ff, "_project_in_weight_t"):
            audio_ff._project_in_weight_t = None
        if hasattr(audio_ff, "_project_out_weight_t"):
            audio_ff._project_out_weight_t = None

    for attn_name in (
        "attn1", "attn2", "audio_to_video_attn",
        "audio_attn1", "audio_attn2", "video_to_audio_attn",
    ):
        attn = getattr(block, attn_name, None)
        if attn is None:
            continue
        for cache_attr in (
            "_to_out_weight_t",
            "_to_q_weight_t",
            "_to_k_weight_t",
            "_to_v_weight_t",
            "_to_gate_logits_weight_t",
        ):
            if hasattr(attn, cache_attr):
                setattr(attn, cache_attr, None)


def _empty_linear(*, has_bias: bool = True) -> nn.Linear:
    linear = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(linear)
    linear.weight = mx.zeros((0, 0), dtype=mx.float32)
    if has_bias:
        linear.bias = mx.zeros((0,), dtype=mx.float32)
    return linear


def _resolve_linear_parent(block: nn.Module, base: str) -> tuple[nn.Module, str] | None:
    parts = base.split(".")
    parent = block
    for part in parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None
    return parent, parts[-1]


def _restore_block_quantized_linears(
    block: nn.Module,
    keep_bases: set[str],
) -> None:
    for base in _CACHE_QUANTIZED_BLOCK_LINEAR_BASES:
        if base in keep_bases:
            continue
        resolved = _resolve_linear_parent(block, base)
        if resolved is None:
            continue
        parent, attr = resolved
        current = getattr(parent, attr, None)
        if isinstance(current, _QUANTIZED_LINEAR_TYPES):
            setattr(parent, attr, _empty_linear())


def _quant_bases_for_block_keys(
    quant_keys: Tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> set[str]:
    bases: set[str] = set()
    for _full_key, block_key in quant_keys:
        for suffix in (".weight", ".scales", ".biases"):
            if block_key.endswith(suffix):
                bases.add(block_key[: -len(suffix)])
                break
    return bases


def _quant_params_for_base(
    quant_keys: Tuple[tuple[str, str], ...] | list[tuple[str, str]],
    base: str,
) -> set[str]:
    prefix = f"{base}."
    return {
        block_key[len(prefix) :]
        for _full_key, block_key in quant_keys
        if block_key.startswith(prefix)
    }


def _empty_quantized_linear(
    *,
    mode: str,
    group_size: int,
    bits: int,
    has_bias: bool,
    has_quant_biases: bool,
    pretransposed: bool,
) -> nn.Module:
    linear_cls = (
        _PretransposedQuantizedLinear
        if pretransposed
        else nn.QuantizedLinear
    )
    linear = linear_cls.__new__(linear_cls)
    nn.Module.__init__(linear)
    linear.group_size = group_size
    linear.bits = bits
    linear.mode = mode
    linear.weight = mx.zeros((0, 0), dtype=mx.uint32)
    linear.scales = mx.zeros((0, 0), dtype=mx.uint8)
    if has_quant_biases:
        linear.biases = mx.zeros((0, 0), dtype=mx.float32)
    if has_bias:
        linear.bias = mx.zeros((0,), dtype=mx.float32)
    linear.freeze()
    return linear


def _ensure_quantized_linear(
    current,
    *,
    mode: str,
    group_size: int,
    bits: int,
    has_bias: bool,
    has_quant_biases: bool,
    pretransposed: bool,
) -> nn.Module:
    expected_cls = (
        _PretransposedQuantizedLinear
        if pretransposed
        else nn.QuantizedLinear
    )
    if (
        not isinstance(current, expected_cls)
        or current.mode != mode
        or current.group_size != group_size
        or current.bits != bits
    ):
        return _empty_quantized_linear(
            mode=mode,
            group_size=group_size,
            bits=bits,
            has_bias=has_bias,
            has_quant_biases=has_quant_biases,
            pretransposed=pretransposed,
        )

    if has_quant_biases and "biases" not in current:
        current.biases = mx.zeros((0, 0), dtype=mx.float32)
    elif not has_quant_biases and "biases" in current:
        del current.biases
    if has_bias and "bias" not in current:
        current.bias = mx.zeros((0,), dtype=mx.float32)
    elif not has_bias and "bias" in current:
        del current.bias
    return current


def _quant_mode_for_base(
    base: str,
    *,
    transformer_cache_quantize: str,
    quantization_specs: Tuple[Tuple[str, str], ...],
) -> str | None:
    if (
        transformer_cache_quantize in (
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
        )
        and base in _CACHE_QUANTIZED_BLOCK_LINEAR_BASES
    ):
        return "mxfp8"
    if base == "ff.project_in.proj":
        return _quant_mode_for_target(quantization_specs, "project_in")
    if base == "ff.project_out":
        return _quant_mode_for_target(quantization_specs, "project_out")
    return None


def _prepare_block_quantized_linears(
    block: nn.Module,
    quant_bases: set[str],
    quant_keys: Tuple[tuple[str, str], ...] | list[tuple[str, str]],
    normal_keys: Tuple[tuple[str, str], ...] | list[tuple[str, str]],
    *,
    transformer_cache_quantize: str,
    quantization_specs: Tuple[Tuple[str, str], ...],
    group_size: int | None,
    bits: int | None,
) -> None:
    normal_block_keys = {block_key for _full_key, block_key in normal_keys}
    pretransposed = _cache_quant_pretransposed(transformer_cache_quantize)
    for base in quant_bases:
        mode = _quant_mode_for_base(
            base,
            transformer_cache_quantize=transformer_cache_quantize,
            quantization_specs=quantization_specs,
        )
        if mode is None:
            raise ValueError(f"Missing cached quantization mode for transformer target: {base}")
        normalized_group_size, normalized_bits = _quant_defaults(mode, group_size, bits)
        params = _quant_params_for_base(quant_keys, base)
        has_bias = f"{base}.bias" in normal_block_keys
        has_quant_biases = "biases" in params

        resolved = _resolve_linear_parent(block, base)
        if resolved is None:
            raise ValueError(f"Missing transformer module for cached quantization target: {base}")
        parent, attr = resolved
        current = getattr(parent, attr, None)
        if current is None:
            raise ValueError(f"Missing transformer linear for cached quantization target: {base}")
        setattr(
            parent,
            attr,
            _ensure_quantized_linear(
                current,
                mode=mode,
                group_size=normalized_group_size,
                bits=normalized_bits,
                has_bias=has_bias,
                has_quant_biases=has_quant_biases,
                pretransposed=pretransposed,
            ),
        )


_ATTN_PROJ_TO_CACHE_ATTR = {
    "to_out": "_to_out_weight_t",
    "to_q":   "_to_q_weight_t",
    "to_k":   "_to_k_weight_t",
    "to_v":   "_to_v_weight_t",
    "to_gate_logits": "_to_gate_logits_weight_t",
}


def _install_block_layout_weight(block: nn.Module, layout_key: str, value: mx.array) -> None:
    if layout_key == "ff.project_in.proj.weight_t":
        block.ff._project_in_weight_t = value
        if "weight" in block.ff.project_in.proj:
            del block.ff.project_in.proj.weight
        return

    if layout_key == "ff.project_out.weight_t":
        block.ff._project_out_weight_t = value
        if "weight" in block.ff.project_out:
            del block.ff.project_out.weight
        return

    if layout_key == "audio_ff.project_in.proj.weight_t":
        block.audio_ff._project_in_weight_t = value
        if "weight" in block.audio_ff.project_in.proj:
            del block.audio_ff.project_in.proj.weight
        return

    if layout_key == "audio_ff.project_out.weight_t":
        block.audio_ff._project_out_weight_t = value
        if "weight" in block.audio_ff.project_out:
            del block.audio_ff.project_out.weight
        return

    for attn_name in (
        "attn1", "attn2", "audio_to_video_attn",
        "audio_attn1", "audio_attn2", "video_to_audio_attn",
    ):
        for proj_name, cache_attr in _ATTN_PROJ_TO_CACHE_ATTR.items():
            if layout_key == f"{attn_name}.{proj_name}.weight_t":
                attn = getattr(block, attn_name, None)
                if attn is None:
                    raise ValueError(f"Missing attention module for cache key: {layout_key}")
                setattr(attn, cache_attr, value)
                linear = getattr(attn, proj_name, None)
                # to_gate_logits is None on non-V2 attentions — install the
                # cached weight but skip the source-weight delete.
                if linear is not None and "weight" in linear:
                    del linear.weight
                return

    raise ValueError(f"Unsupported transformer cache layout key: {layout_key}")


def _install_top_level_layout_weight(model: nn.Module, layout_key: str, value: mx.array) -> None:
    """Install a top-level (non-transformer_blocks) layout tensor.

    Currently used for ``<adaln>.linear.weight_t`` keys — sets the cached
    pretransposed weight on the AdaLayerNormSingle and drops the original
    weight to keep memory flat.
    """
    if not layout_key.endswith(".linear.weight_t"):
        raise ValueError(f"Unsupported top-level layout key: {layout_key}")
    base = layout_key[: -len(".linear.weight_t")]
    module = getattr(model, base, None)
    if module is None:
        raise ValueError(f"Missing module for layout key: {layout_key} (looked for self.{base})")
    if not hasattr(module, "_linear_weight_t"):
        raise ValueError(
            f"Module self.{base} doesn't support pretranspose "
            "(expected AdaLayerNormSingle-style _linear_weight_t)"
        )
    module._linear_weight_t = value
    if hasattr(module, "linear") and "weight" in module.linear:
        del module.linear.weight


def _install_layout_weight(model: nn.Module, layout_key: str, value: mx.array) -> None:
    parts = layout_key.split(".")
    if not parts:
        raise ValueError(f"Invalid transformer cache layout key: {layout_key}")

    if parts[0] != "transformer_blocks":
        _install_top_level_layout_weight(model, layout_key, value)
        return

    if len(parts) < 5:
        raise ValueError(f"Invalid transformer cache layout key: {layout_key}")
    layer = int(parts[1])
    block = model.transformer_blocks[layer]
    _install_block_layout_weight(block, ".".join(parts[2:]), value)


def load_transformer_cache(
    model: nn.Module,
    cache_file: Path,
    *,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
) -> tuple[int, int, int]:
    """Load a converted transformer cache into an existing model instance."""
    cached_weights = mx.load(str(cache_file))
    normal_weights: Dict[str, mx.array] = {}
    layout_weights: Dict[str, mx.array] = {}
    quant_weights: Dict[str, mx.array] = {}
    normal_keys_by_block: dict[int, list[tuple[str, str]]] = {}
    quant_keys_by_block: dict[int, list[tuple[str, str]]] = {}

    for key, value in cached_weights.items():
        if key.startswith(LAYOUT_KEY_PREFIX):
            layout_weights[key[len(LAYOUT_KEY_PREFIX) :]] = value
        elif key.startswith(QUANT_KEY_PREFIX):
            logical_key = key[len(QUANT_KEY_PREFIX) :]
            quant_weights[logical_key] = value
            parts = logical_key.split(".")
            if len(parts) >= 3 and parts[0] == "transformer_blocks":
                try:
                    block_idx = int(parts[1])
                except ValueError:
                    pass
                else:
                    quant_keys_by_block.setdefault(block_idx, []).append(
                        (key, ".".join(parts[2:]))
                    )
        else:
            normal_weights[key] = value
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "transformer_blocks":
                try:
                    block_idx = int(parts[1])
                except ValueError:
                    pass
                else:
                    normal_keys_by_block.setdefault(block_idx, []).append(
                        (key, ".".join(parts[2:]))
                    )

    for block_idx, quant_keys in quant_keys_by_block.items():
        if block_idx >= len(model.transformer_blocks):
            raise ValueError(
                f"Quantized transformer cache has block {block_idx}, "
                f"but model only has {len(model.transformer_blocks)} blocks"
            )
        block = model.transformer_blocks[block_idx]
        quant_bases = _quant_bases_for_block_keys(quant_keys)
        _restore_block_quantized_linears(block, keep_bases=quant_bases)
        _prepare_block_quantized_linears(
            block,
            quant_bases,
            quant_keys,
            normal_keys_by_block.get(block_idx, ()),
            transformer_cache_quantize=transformer_cache_quantize,
            quantization_specs=video_ff_quantize_specs,
            group_size=video_ff_quantize_group_size,
            bits=video_ff_quantize_bits,
        )

    if normal_weights:
        model.update(_flatten_to_nested(normal_weights))
    if quant_weights:
        model.update(_flatten_to_nested(quant_weights))
    for key, value in layout_weights.items():
        _install_layout_weight(model, key, value)

    loaded_count = len(normal_weights) + len(layout_weights) + len(quant_weights)
    layout_count = len(layout_weights)
    quant_count = len(quant_weights)
    del cached_weights, normal_weights, layout_weights, quant_weights
    gc.collect()
    return loaded_count, layout_count, quant_count


def ensure_transformer_cache(
    weights_path: str,
    *,
    transformer_dtype: Optional[mx.Dtype] = None,
    cache_mode: str,
    cache_root: str | None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    audio_ff_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_ff_layout_layers: Tuple[int, ...] = (),
    audio_attn_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_attn_layout_layers: Tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...] = (),
    video_ff_quantize_layers: Tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: Optional[mx.Dtype] = None,
    audio_ff_dtype: Optional[mx.Dtype] = None,
) -> TransformerCacheResult:
    """Build a matching transformer cache artifact if needed."""
    if cache_mode not in {"auto", "rebuild"}:
        raise ValueError(f"Unsupported transformer cache mode: {cache_mode}")

    cache_file, metadata_file, payload = transformer_cache_paths(
        weights_path,
        cache_root,
        transformer_dtype=transformer_dtype,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        adaln_pretranspose=adaln_pretranspose,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
    )
    rebuilt = False
    cache_valid = cache_file.exists() and _metadata_matches(metadata_file, payload)
    if cache_mode == "rebuild" or not cache_valid:
        print(f"  Transformer cache: building {cache_file}")
        build_transformer_cache(
            weights_path,
            cache_file,
            metadata_file,
            payload,
            transformer_dtype=transformer_dtype,
            include_audio=include_audio,
            video_ff_layout_specs=video_ff_layout_specs,
            video_ff_layout_layers=video_ff_layout_layers,
            video_attn_layout_specs=video_attn_layout_specs,
            video_attn_layout_layers=video_attn_layout_layers,
            audio_ff_layout_specs=audio_ff_layout_specs,
            audio_ff_layout_layers=audio_ff_layout_layers,
            audio_attn_layout_specs=audio_attn_layout_specs,
            audio_attn_layout_layers=audio_attn_layout_layers,
            adaln_pretranspose=adaln_pretranspose,
            transformer_cache_quantize=transformer_cache_quantize,
            video_ff_quantize_specs=video_ff_quantize_specs,
            video_ff_quantize_layers=video_ff_quantize_layers,
            video_ff_quantize_group_size=video_ff_quantize_group_size,
            video_ff_quantize_bits=video_ff_quantize_bits,
            video_ff_dtype=video_ff_dtype,
            audio_ff_dtype=audio_ff_dtype,
        )
        rebuilt = True
    else:
        print(f"  Transformer cache: using {cache_file}")

    return TransformerCacheResult(
        cache_path=cache_file,
        rebuilt=rebuilt,
        loaded_count=0,
        layout_count=0,
    )


def load_transformer_weights_cached(
    model: nn.Module,
    weights_path: str,
    *,
    transformer_dtype: Optional[mx.Dtype] = None,
    cache_mode: str,
    cache_root: str | None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    audio_ff_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_ff_layout_layers: Tuple[int, ...] = (),
    audio_attn_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_attn_layout_layers: Tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_dtype: Optional[mx.Dtype] = None,
    audio_ff_dtype: Optional[mx.Dtype] = None,
) -> TransformerCacheResult:
    """Build if needed, then load a transformer cache into ``model``."""
    result = ensure_transformer_cache(
        weights_path,
        transformer_dtype=transformer_dtype,
        cache_mode=cache_mode,
        cache_root=cache_root,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        adaln_pretranspose=adaln_pretranspose,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
    )

    loaded_count, layout_count, quant_count = load_transformer_cache(
        model,
        result.cache_path,
        transformer_cache_quantize=transformer_cache_quantize,
    )
    print(
        f"  Loaded transformer cache: {loaded_count} tensors "
        f"({layout_count} layout tensors, {quant_count} quant tensors)"
    )
    return TransformerCacheResult(
        cache_path=result.cache_path,
        rebuilt=result.rebuilt,
        loaded_count=loaded_count,
        layout_count=layout_count,
        quant_count=quant_count,
    )


def load_transformer_weights_cached_streaming(
    model: nn.Module,
    weights_path: str,
    *,
    transformer_dtype: Optional[mx.Dtype] = None,
    cache_mode: str,
    cache_root: str | None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    resident_blocks: int,
    audio_ff_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_ff_layout_layers: Tuple[int, ...] = (),
    audio_attn_layout_specs: Tuple[Tuple[str, str], ...] = (),
    audio_attn_layout_layers: Tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: Tuple[Tuple[str, str], ...] = (),
    video_ff_quantize_layers: Tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: Optional[mx.Dtype] = None,
    audio_ff_dtype: Optional[mx.Dtype] = None,
) -> TransformerCacheResult:
    """Load non-block weights and stream transformer blocks from the cache."""
    if resident_blocks <= 0:
        raise ValueError("resident_blocks must be positive for transformer block streaming")

    result = ensure_transformer_cache(
        weights_path,
        transformer_dtype=transformer_dtype,
        cache_mode=cache_mode,
        cache_root=cache_root,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        adaln_pretranspose=adaln_pretranspose,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
    )

    if resident_blocks > len(model.transformer_blocks):
        raise ValueError(
            f"resident_blocks={resident_blocks} exceeds model block count "
            f"{len(model.transformer_blocks)}"
        )
    model.transformer_blocks = model.transformer_blocks[:resident_blocks]

    cached_weights = mx.load(str(result.cache_path))
    non_block_weights: Dict[str, mx.array] = {}
    for key, value in cached_weights.items():
        if key.startswith(LAYOUT_KEY_PREFIX):
            logical_key = key[len(LAYOUT_KEY_PREFIX) :]
        elif key.startswith(QUANT_KEY_PREFIX):
            logical_key = key[len(QUANT_KEY_PREFIX) :]
        else:
            logical_key = key
        if logical_key.startswith("transformer_blocks."):
            continue
        if key.startswith(LAYOUT_KEY_PREFIX):
            raise ValueError(f"Unexpected non-block layout cache key: {key}")
        if key.startswith(QUANT_KEY_PREFIX):
            raise ValueError(f"Unexpected non-block quant cache key: {key}")
        non_block_weights[key] = value

    if non_block_weights:
        model.update(_flatten_to_nested(non_block_weights))

    del cached_weights, non_block_weights
    gc.collect()

    streamer = TransformerBlockStreamer(
        result.cache_path,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
    )
    model.transformer_block_streamer = streamer
    print(
        f"  Loaded transformer cache: {streamer.loaded_count} streamed block tensors "
        f"({streamer.layout_count} layout tensors, "
        f"{streamer.quant_count} quant tensors), "
        f"{resident_blocks}/{streamer.block_count} blocks resident"
    )
    return TransformerCacheResult(
        cache_path=result.cache_path,
        rebuilt=result.rebuilt,
        loaded_count=streamer.loaded_count,
        layout_count=streamer.layout_count,
        quant_count=streamer.quant_count,
    )


def ensure_weight_family_caches(
    weights_path: str,
    *,
    families: tuple[str, ...],
    cache_mode: str,
    cache_root: str | None,
) -> WeightFamilyCacheResult:
    """Ensure named auxiliary weight family caches exist and return their paths."""
    if cache_mode not in {"auto", "rebuild"}:
        raise ValueError(f"Unsupported weight family cache mode: {cache_mode}")

    families = tuple(dict.fromkeys(families))
    for family in families:
        if family not in WEIGHT_FAMILIES:
            raise ValueError(f"Unsupported weight family: {family}")

    cache_paths: dict[str, Path] = {}
    missing_families: list[str] = []
    for family in families:
        cache_file, metadata_file, payload = weight_family_cache_paths(
            weights_path,
            cache_root,
            family,
        )
        cache_paths[family] = cache_file
        cache_valid = cache_file.exists() and _metadata_matches(metadata_file, payload)
        if cache_mode == "rebuild" or not cache_valid:
            missing_families.append(family)

    rebuilt = False
    if missing_families:
        for family in missing_families:
            print(
                f"  {WEIGHT_FAMILY_LABELS[family]} cache: building "
                f"{cache_paths[family]}"
            )
        loaded_count = build_weight_family_caches(
            weights_path,
            cache_root,
            tuple(missing_families),
        )
        rebuilt = True
    else:
        loaded_count = 0
        for family in families:
            print(
                f"  {WEIGHT_FAMILY_LABELS[family]} cache: using "
                f"{cache_paths[family]}"
            )

    return WeightFamilyCacheResult(
        cache_paths=cache_paths,
        rebuilt=rebuilt,
        loaded_count=loaded_count,
    )
