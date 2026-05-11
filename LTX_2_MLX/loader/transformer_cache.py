"""Disposable cache for converted transformer and component weights."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn

from .weight_converter import _flatten_to_nested, convert_pytorch_key_to_mlx


CACHE_SCHEMA_VERSION = 1
LAYOUT_KEY_PREFIX = "__layout__."


@dataclass(frozen=True)
class TransformerCacheResult:
    """Result metadata for a transformer cache load."""

    cache_path: Path
    rebuilt: bool
    loaded_count: int
    layout_count: int


class TransformerBlockStreamer:
    """Bind cached transformer-block weights into a small resident block pool."""

    def __init__(self, cache_file: Path) -> None:
        self.cache_file = cache_file
        self._weights = mx.load(str(cache_file))
        self._block_keys: dict[int, list[tuple[str, str]]] = {}
        self._layout_keys: dict[int, list[tuple[str, str]]] = {}

        for full_key in list(self._weights):
            is_layout = full_key.startswith(LAYOUT_KEY_PREFIX)
            logical_key = full_key[len(LAYOUT_KEY_PREFIX) :] if is_layout else full_key
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
            target = self._layout_keys if is_layout else self._block_keys
            target.setdefault(block_idx, []).append((full_key, block_key))

        discovered = set(self._block_keys) | set(self._layout_keys)
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
        )
        self.layout_count = sum(len(items) for items in self._layout_keys.values())

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

        normal_keys = self._block_keys.get(block_idx, ())
        layout_keys = self._layout_keys.get(block_idx, ())
        sample_key = None
        if normal_keys:
            sample_key = normal_keys[0][0]
        elif layout_keys:
            sample_key = layout_keys[0][0]
        if sample_key is not None and sample_key not in self._weights:
            self._weights = mx.load(str(self.cache_file))
            self._drop_non_block_keys()

        _clear_block_layout_weights(block)
        normal_weights = {
            block_key: self._weights[full_key]
            for full_key, block_key in normal_keys
        }
        if normal_weights:
            block.update(_flatten_to_nested(normal_weights))
        for full_key, layout_key in layout_keys:
            _install_block_layout_weight(block, layout_key, self._weights[full_key])
        if hasattr(block, "idx"):
            block.idx = block_idx
        return block

    def close(self) -> None:
        self._weights = {}
        self._block_keys = {}
        self._layout_keys = {}

    def _drop_non_block_keys(self) -> None:
        for full_key in list(self._weights):
            logical_key = (
                full_key[len(LAYOUT_KEY_PREFIX) :]
                if full_key.startswith(LAYOUT_KEY_PREFIX)
                else full_key
            )
            if not logical_key.startswith("transformer_blocks."):
                self._weights.pop(full_key, None)


@dataclass(frozen=True)
class ComponentCacheResult:
    """Result metadata for a component cache load."""

    cache_path: Path
    rebuilt: bool
    loaded_count: int


def default_transformer_cache_root() -> Path:
    """Return the default location for disposable weight cache artifacts."""
    env_path = os.environ.get("LTX_MLX_WEIGHTS_CACHE_DIR")
    if not env_path:
        env_path = os.environ.get("LTX_MLX_TRANSFORMER_CACHE_DIR")
    if env_path:
        return Path(env_path).expanduser()

    shared_mlx = Path("/Users/Shared/huggingface/mlx")
    if shared_mlx.exists():
        return shared_mlx / "LTX-2-MLX-cache"

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


def _cache_payload(
    weights_path: str,
    *,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": _file_signature(weights_path),
        "include_audio": include_audio,
        "video_ff_layout_specs": _canonical_specs(video_ff_layout_specs),
        "video_ff_layout_layers": list(video_ff_layout_layers),
        "video_attn_layout_specs": _canonical_specs(video_attn_layout_specs),
        "video_attn_layout_layers": list(video_attn_layout_layers),
    }


def _source_payload(weights_path: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": _file_signature(weights_path),
        "kind": "components",
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
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve cache artifact paths and expected metadata payload."""
    payload = _cache_payload(
        weights_path,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
    )
    root = Path(cache_root).expanduser() if cache_root else default_transformer_cache_root()
    cache_dir = root / f"{_safe_stem(weights_path)}-{_payload_digest(payload)}"
    return cache_dir / "transformer.safetensors", cache_dir / "metadata.json", payload


def component_cache_paths(
    weights_path: str,
    cache_root: str | None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve source-level component cache paths and expected metadata."""
    payload = _source_payload(weights_path)
    root = Path(cache_root).expanduser() if cache_root else default_transformer_cache_root()
    cache_dir = root / f"{_safe_stem(weights_path)}-{_payload_digest(payload)}"
    return cache_dir / "components.safetensors", cache_dir / "components.metadata.json", payload


def _selected_layers(layers: Tuple[int, ...], num_layers: int = 48) -> set[int]:
    return set(layers) if layers else set(range(num_layers))


def _has_spec(specs: Tuple[Tuple[str, str], ...], target: str) -> bool:
    return (target, "pretranspose") in specs


def _layout_cache_key(
    mlx_key: str,
    *,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
) -> str | None:
    parts = mlx_key.split(".")
    if len(parts) < 5 or parts[0] != "transformer_blocks" or parts[-1] != "weight":
        return None

    try:
        layer = int(parts[1])
    except ValueError:
        return None

    suffix = ".".join(parts[2:])
    ff_layers = _selected_layers(video_ff_layout_layers)
    attn_layers = _selected_layers(video_attn_layout_layers)

    if layer in ff_layers:
        if _has_spec(video_ff_layout_specs, "project_in") and suffix == "ff.project_in.proj.weight":
            return f"transformer_blocks.{layer}.ff.project_in.proj.weight_t"
        if _has_spec(video_ff_layout_specs, "project_out") and suffix == "ff.project_out.weight":
            return f"transformer_blocks.{layer}.ff.project_out.weight_t"

    if layer in attn_layers and _has_spec(video_attn_layout_specs, "to_out"):
        if suffix in (
            "attn1.to_out.weight",
            "attn2.to_out.weight",
            "audio_to_video_attn.to_out.weight",
        ):
            return f"transformer_blocks.{layer}.{suffix[:-len('weight')]}weight_t"

    return None


def _metadata_matches(metadata_path: Path, expected_payload: dict[str, Any]) -> bool:
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected_payload


def _is_component_cache_key(key: str) -> bool:
    """Return True for non-transformer component weights worth caching separately."""
    if key.startswith(("vae.", "audio_vae.", "vocoder.", "text_embedding_projection.")):
        return True
    return key.startswith(
        (
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
            "model.diffusion_model.embeddings_connector.",
        )
    )


def _write_metadata(metadata_file: Path, payload: dict[str, Any]) -> None:
    tmp_metadata = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    with tmp_metadata.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_metadata, metadata_file)


def _save_component_cache(
    cache_file: Path,
    metadata_file: Path,
    payload: dict[str, Any],
    component_weights: Dict[str, mx.array],
) -> int:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache = cache_file.parent / f".{cache_file.stem}.tmp.safetensors"
    mx.save_safetensors(str(tmp_cache), component_weights)
    os.replace(tmp_cache, cache_file)
    _write_metadata(metadata_file, payload)
    loaded_count = len(component_weights)
    print(f"  Built component cache: {loaded_count} tensors")
    return loaded_count


def build_component_cache(
    weights_path: str,
    cache_file: Path,
    metadata_file: Path,
    payload: dict[str, Any],
) -> int:
    """Build a source-level cache for VAE/audio/text connector components."""
    raw_weights = mx.load(weights_path)
    component_weights = {
        key: value for key, value in raw_weights.items() if _is_component_cache_key(key)
    }
    del raw_weights
    gc.collect()
    return _save_component_cache(cache_file, metadata_file, payload, component_weights)


def build_transformer_cache(
    weights_path: str,
    cache_file: Path,
    metadata_file: Path,
    payload: dict[str, Any],
    *,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    component_cache_file: Path | None = None,
    component_metadata_file: Path | None = None,
    component_payload: dict[str, Any] | None = None,
    build_component_file: bool = False,
) -> tuple[int, int]:
    """Build a converted transformer-only cache from a stock checkpoint."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    raw_weights = mx.load(weights_path)
    cache_weights: Dict[str, mx.array] = {}
    component_weights: Dict[str, mx.array] = {}
    loaded_count = 0
    layout_count = 0
    skipped_count = 0

    for pytorch_key, value in raw_weights.items():
        if build_component_file and _is_component_cache_key(pytorch_key):
            component_weights[pytorch_key] = value

        if not pytorch_key.startswith("model.diffusion_model."):
            continue

        key = pytorch_key.replace("model.diffusion_model.", "")
        mlx_key = convert_pytorch_key_to_mlx(key, include_audio=include_audio)
        if mlx_key is None:
            skipped_count += 1
            continue

        layout_key = _layout_cache_key(
            mlx_key,
            video_ff_layout_specs=video_ff_layout_specs,
            video_ff_layout_layers=video_ff_layout_layers,
            video_attn_layout_specs=video_attn_layout_specs,
            video_attn_layout_layers=video_attn_layout_layers,
        )
        if layout_key is not None:
            cache_weights[f"{LAYOUT_KEY_PREFIX}{layout_key}"] = mx.contiguous(value.T)
            layout_count += 1
        else:
            cache_weights[mlx_key] = value
        loaded_count += 1

    del raw_weights
    gc.collect()

    tmp_cache = cache_file.parent / f".{cache_file.stem}.tmp.safetensors"
    mx.save_safetensors(str(tmp_cache), cache_weights)
    os.replace(tmp_cache, cache_file)

    _write_metadata(metadata_file, payload)

    if (
        build_component_file
        and component_cache_file is not None
        and component_metadata_file is not None
        and component_payload is not None
    ):
        _save_component_cache(
            component_cache_file,
            component_metadata_file,
            component_payload,
            component_weights,
        )

    print(
        f"  Built transformer cache: {loaded_count} tensors "
        f"({layout_count} pretransposed, skipped {skipped_count})"
    )
    return loaded_count, layout_count


def _clear_block_layout_weights(block: nn.Module) -> None:
    ff = getattr(block, "ff", None)
    if ff is not None:
        if hasattr(ff, "_project_in_weight_t"):
            ff._project_in_weight_t = None
        if hasattr(ff, "_project_out_weight_t"):
            ff._project_out_weight_t = None

    for attn_name in ("attn1", "attn2", "audio_to_video_attn"):
        attn = getattr(block, attn_name, None)
        if attn is None:
            continue
        if hasattr(attn, "_to_out_weight_t"):
            attn._to_out_weight_t = None


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

    for attn_name in ("attn1", "attn2", "audio_to_video_attn"):
        if layout_key == f"{attn_name}.to_out.weight_t":
            attn = getattr(block, attn_name, None)
            if attn is None:
                raise ValueError(f"Missing attention module for cache key: {layout_key}")
            attn._to_out_weight_t = value
            if "weight" in attn.to_out:
                del attn.to_out.weight
            return

    raise ValueError(f"Unsupported transformer cache layout key: {layout_key}")


def _install_layout_weight(model: nn.Module, layout_key: str, value: mx.array) -> None:
    parts = layout_key.split(".")
    if len(parts) < 5 or parts[0] != "transformer_blocks":
        raise ValueError(f"Invalid transformer cache layout key: {layout_key}")

    layer = int(parts[1])
    block = model.transformer_blocks[layer]
    _install_block_layout_weight(block, ".".join(parts[2:]), value)


def load_transformer_cache(model: nn.Module, cache_file: Path) -> tuple[int, int]:
    """Load a converted transformer cache into an existing model instance."""
    cached_weights = mx.load(str(cache_file))
    normal_weights: Dict[str, mx.array] = {}
    layout_weights: Dict[str, mx.array] = {}

    for key, value in cached_weights.items():
        if key.startswith(LAYOUT_KEY_PREFIX):
            layout_weights[key[len(LAYOUT_KEY_PREFIX) :]] = value
        else:
            normal_weights[key] = value

    if normal_weights:
        model.update(_flatten_to_nested(normal_weights))
    for key, value in layout_weights.items():
        _install_layout_weight(model, key, value)

    loaded_count = len(normal_weights) + len(layout_weights)
    layout_count = len(layout_weights)
    del cached_weights, normal_weights, layout_weights
    gc.collect()
    return loaded_count, layout_count


def ensure_transformer_cache(
    weights_path: str,
    *,
    cache_mode: str,
    cache_root: str | None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
) -> TransformerCacheResult:
    """Build matching transformer/component cache artifacts if needed."""
    if cache_mode not in {"auto", "rebuild"}:
        raise ValueError(f"Unsupported transformer cache mode: {cache_mode}")

    cache_file, metadata_file, payload = transformer_cache_paths(
        weights_path,
        cache_root,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
    )
    component_cache_file, component_metadata_file, component_payload = component_cache_paths(
        weights_path,
        cache_root,
    )

    rebuilt = False
    cache_valid = cache_file.exists() and _metadata_matches(metadata_file, payload)
    component_cache_valid = component_cache_file.exists() and _metadata_matches(
        component_metadata_file,
        component_payload,
    )
    if cache_mode == "rebuild" or not cache_valid:
        print(f"  Transformer cache: building {cache_file}")
        build_transformer_cache(
            weights_path,
            cache_file,
            metadata_file,
            payload,
            include_audio=include_audio,
            video_ff_layout_specs=video_ff_layout_specs,
            video_ff_layout_layers=video_ff_layout_layers,
            video_attn_layout_specs=video_attn_layout_specs,
            video_attn_layout_layers=video_attn_layout_layers,
            component_cache_file=component_cache_file,
            component_metadata_file=component_metadata_file,
            component_payload=component_payload,
            build_component_file=cache_mode == "rebuild" or not component_cache_valid,
        )
        rebuilt = True
    elif not component_cache_valid:
        print(f"  Component cache: building {component_cache_file}")
        build_component_cache(
            weights_path,
            component_cache_file,
            component_metadata_file,
            component_payload,
        )
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
    cache_mode: str,
    cache_root: str | None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
) -> TransformerCacheResult:
    """Build if needed, then load a transformer cache into ``model``."""
    result = ensure_transformer_cache(
        weights_path,
        cache_mode=cache_mode,
        cache_root=cache_root,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
    )

    loaded_count, layout_count = load_transformer_cache(model, result.cache_path)
    print(
        f"  Loaded transformer cache: {loaded_count} tensors "
        f"({layout_count} layout tensors)"
    )
    return TransformerCacheResult(
        cache_path=result.cache_path,
        rebuilt=result.rebuilt,
        loaded_count=loaded_count,
        layout_count=layout_count,
    )


def load_transformer_weights_cached_streaming(
    model: nn.Module,
    weights_path: str,
    *,
    cache_mode: str,
    cache_root: str | None,
    include_audio: bool,
    video_ff_layout_specs: Tuple[Tuple[str, str], ...],
    video_ff_layout_layers: Tuple[int, ...],
    video_attn_layout_specs: Tuple[Tuple[str, str], ...],
    video_attn_layout_layers: Tuple[int, ...],
    resident_blocks: int,
) -> TransformerCacheResult:
    """Load non-block weights and stream transformer blocks from the cache."""
    if resident_blocks <= 0:
        raise ValueError("resident_blocks must be positive for transformer block streaming")

    result = ensure_transformer_cache(
        weights_path,
        cache_mode=cache_mode,
        cache_root=cache_root,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
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
        logical_key = key[len(LAYOUT_KEY_PREFIX) :] if key.startswith(LAYOUT_KEY_PREFIX) else key
        if logical_key.startswith("transformer_blocks."):
            continue
        if key.startswith(LAYOUT_KEY_PREFIX):
            raise ValueError(f"Unexpected non-block layout cache key: {key}")
        non_block_weights[key] = value

    if non_block_weights:
        model.update(_flatten_to_nested(non_block_weights))

    del cached_weights, non_block_weights
    gc.collect()

    streamer = TransformerBlockStreamer(result.cache_path)
    model.transformer_block_streamer = streamer
    print(
        f"  Loaded transformer cache: {streamer.loaded_count} streamed block tensors "
        f"({streamer.layout_count} layout tensors), "
        f"{resident_blocks}/{streamer.block_count} blocks resident"
    )
    return TransformerCacheResult(
        cache_path=result.cache_path,
        rebuilt=result.rebuilt,
        loaded_count=streamer.loaded_count,
        layout_count=streamer.layout_count,
    )


def load_component_weights_cached(
    weights_path: str,
    *,
    cache_mode: str,
    cache_root: str | None,
) -> ComponentCacheResult:
    """Ensure component cache exists and return the smaller cache file path."""
    if cache_mode not in {"auto", "rebuild"}:
        raise ValueError(f"Unsupported component cache mode: {cache_mode}")

    cache_file, metadata_file, payload = component_cache_paths(weights_path, cache_root)
    rebuilt = False
    cache_valid = cache_file.exists() and _metadata_matches(metadata_file, payload)
    if cache_mode == "rebuild" or not cache_valid:
        print(f"  Component cache: building {cache_file}")
        loaded_count = build_component_cache(weights_path, cache_file, metadata_file, payload)
        rebuilt = True
    else:
        loaded_count = 0
        print(f"  Component cache: using {cache_file}")

    return ComponentCacheResult(
        cache_path=cache_file,
        rebuilt=rebuilt,
        loaded_count=loaded_count,
    )
