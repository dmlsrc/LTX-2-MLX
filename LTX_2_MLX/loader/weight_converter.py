"""Transformer weight key conversion and fallback checkpoint loading."""

import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def convert_pytorch_key_to_mlx(pytorch_key: str, include_audio: bool = False) -> str | None:
    """
    Convert PyTorch weight key to MLX model key path.

    Args:
        pytorch_key: Original PyTorch key (after removing model.diffusion_model.).
        include_audio: If True, include audio/av_ca related keys.

    Returns:
        MLX-compatible key path, or None if should be skipped.
    """
    key = pytorch_key

    # Skip audio/video cross-attention keys unless explicitly included
    if not include_audio:
        if "av_ca" in key or "a2v" in key or "audio" in key.lower():
            return None

    # Skip embeddings connectors - these are text encoder weights loaded separately
    # via load_text_encoder_weights(), not part of the transformer model
    if "video_embeddings_connector" in key or "audio_embeddings_connector" in key:
        return None

    # Handle to_out.0 -> to_out (PyTorch Sequential vs MLX direct)
    key = re.sub(r"\.to_out\.0\.", ".to_out.", key)

    # Handle ff.net.0.proj -> ff.project_in.proj
    key = re.sub(r"(^|\.)ff\.net\.0\.proj\.", r"\1ff.project_in.proj.", key)

    # Handle ff.net.2 -> ff.project_out
    key = re.sub(r"(^|\.)ff\.net\.2\.", r"\1ff.project_out.", key)

    # Handle audio ff.net.0.proj -> audio_ff.project_in.proj
    key = re.sub(
        r"(^|\.)audio_ff\.net\.0\.proj\.",
        r"\1audio_ff.project_in.proj.",
        key,
    )

    # Handle audio ff.net.2 -> audio_ff.project_out
    key = re.sub(r"(^|\.)audio_ff\.net\.2\.", r"\1audio_ff.project_out.", key)

    return key


def load_transformer_weights(
    model: nn.Module,
    weights_path: str,
    include_audio: bool = False,
) -> None:
    """
    Load transformer weights into an MLX model.

    Args:
        model: MLX model to load weights into.
        weights_path: Path to safetensors file.
        include_audio: If True, include audio-related weights (for AudioVideo model).
    """
    import gc

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print(f"Loading weights from {weights_path}...")

    raw_weights = mx.load(weights_path)
    if has_tqdm:
        key_iter = tqdm(
            raw_weights.items(),
            desc="Loading transformer",
            ncols=80,
            total=len(raw_weights),
            ascii=True,
            mininterval=1.0,
        )
    else:
        key_iter = raw_weights.items()

    weights_dict = {}
    loaded_count = 0
    skipped_count = 0

    for pytorch_key, value in key_iter:
        # Only process diffusion model keys
        if not pytorch_key.startswith("model.diffusion_model."):
            continue

        # Remove prefix
        key = pytorch_key.replace("model.diffusion_model.", "")

        # Convert key
        mlx_key = convert_pytorch_key_to_mlx(key, include_audio=include_audio)
        if mlx_key is None:
            skipped_count += 1
            continue

        weights_dict[mlx_key] = value
        loaded_count += 1

    del raw_weights
    gc.collect()

    print(f"  Converted {loaded_count} weight tensors (skipped {skipped_count})")

    # Update the model using the update method
    # MLX models use a nested dict structure for model.update()
    nested_weights = _flatten_to_nested(weights_dict)

    # Load weights into model
    model.update(nested_weights)

    print("  Successfully loaded weights into model")


def _flatten_to_nested(flat_dict: dict[str, mx.array]) -> dict[str, Any]:
    """
    Convert flat dict with dotted keys to nested dict for model.update().

    Args:
        flat_dict: Dictionary with keys like "transformer_blocks.0.attn1.to_q.weight"

    Returns:
        Nested dictionary structure with lists where indices are numeric.
    """
    nested = {}

    for key, value in flat_dict.items():
        parts = key.split(".")
        current = nested

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    # Convert dicts with numeric string keys to lists
    return _convert_numeric_dicts_to_lists(nested)


def _convert_numeric_dicts_to_lists(obj: Any) -> Any:
    """
    Recursively convert dicts with numeric string keys to lists.

    For example: {"0": {...}, "1": {...}} -> [{...}, {...}]

    This is needed because MLX model.update() expects lists for
    list-type attributes like transformer_blocks.
    """
    if isinstance(obj, dict):
        # First, recursively process all values
        processed = {k: _convert_numeric_dicts_to_lists(v) for k, v in obj.items()}

        # Check if all keys are numeric strings
        if processed and all(k.isdigit() for k in processed):
            # Convert to list, handling potential gaps
            max_idx = max(int(k) for k in processed)
            result = [None] * (max_idx + 1)
            for k, v in processed.items():
                result[int(k)] = v
            return result

        return processed

    return obj


def load_av_transformer_weights(
    model: nn.Module,
    weights_path: str,
) -> None:
    """
    Load AudioVideo transformer weights into an MLX model.

    Convenience function that calls load_transformer_weights with include_audio=True.

    Args:
        model: MLX AudioVideo model to load weights into.
        weights_path: Path to safetensors file.
    """
    load_transformer_weights(
        model=model,
        weights_path=weights_path,
        include_audio=True,
    )
