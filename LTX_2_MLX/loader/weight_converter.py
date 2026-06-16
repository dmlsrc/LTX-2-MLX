"""Weight conversion from PyTorch safetensors to MLX format."""

import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def load_safetensors(path: str) -> dict[str, mx.array]:
    """
    Load weights from a safetensors file.

    Args:
        path: Path to safetensors file.

    Returns:
        Dictionary of weight name to mx.array.
    """
    return dict(mx.load(path))


def transpose_linear_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """
    Transpose 2D weight matrices in a weights dictionary.

    Note: Both MLX and PyTorch Linear layers store weights as [out_features, in_features],
    so this function is NOT needed for standard weight loading. It exists for testing
    and special cases where explicit transposition is required.

    This function transposes 2D ".weight" tensors, excluding pure embeddings
    (but including projection layers that contain "proj" in the name).

    Args:
        weights: Dictionary of weights.

    Returns:
        Dictionary with transposed 2D weight matrices.
    """
    transposed = {}
    for key, value in weights.items():
        # Linear layers have 2D weights
        if value.ndim == 2 and ".weight" in key:
            # Check if this is a linear layer (not embedding, etc.)
            if "embed" not in key.lower() or "proj" in key.lower():
                transposed[key] = value.T
            else:
                transposed[key] = value
        else:
            transposed[key] = value
    return transposed


def convert_transformer_key(pytorch_key: str) -> str | None:
    """
    Convert a PyTorch transformer key to MLX format.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    # Skip audio-related weights
    if "audio" in pytorch_key.lower():
        return None

    # Skip vocoder weights
    if pytorch_key.startswith("vocoder"):
        return None

    # Skip audio VAE weights
    if pytorch_key.startswith("audio_vae"):
        return None

    # Map prefixes
    key = pytorch_key

    # model.diffusion_model -> transformer
    key = key.replace("model.diffusion_model.", "")

    # Handle to_out.0 -> to_out (MLX doesn't use Sequential)
    key = re.sub(r"\.to_out\.0\.", ".to_out.", key)

    # Handle ff.net.0.proj -> ff.project_in.proj
    key = re.sub(r"\.ff\.net\.0\.proj\.", ".ff.project_in.proj.", key)

    # Handle ff.net.2 -> ff.project_out
    key = re.sub(r"\.ff\.net\.2\.", ".ff.project_out.", key)

    # Handle norm weight (RMSNorm doesn't have bias in our impl)
    # q_norm.weight, k_norm.weight stay as is

    return key


def convert_vae_key(pytorch_key: str) -> str | None:
    """
    Convert a PyTorch VAE key to MLX format.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    # Only process video VAE
    if not pytorch_key.startswith("vae."):
        return None

    key = pytorch_key.replace("vae.", "")

    # Handle conv.weight/bias inside DualConv3d
    # In PyTorch: conv.weight -> In MLX: spatial_conv.weight or time_conv.weight
    # For now, keep the structure similar

    return key


def convert_text_encoder_key(pytorch_key: str) -> str | None:
    """
    Convert text encoder weight key.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    if pytorch_key.startswith("text_embedding_projection."):
        # text_embedding_projection.aggregate_embed.weight
        # -> feature_extractor.aggregate_embed.weight
        return pytorch_key.replace(
            "text_embedding_projection.",
            "feature_extractor.",
        )

    # Handle video_embeddings_connector (video-only encoder) first
    if "video_embeddings_connector" in pytorch_key:
        # model.diffusion_model.video_embeddings_connector.xxx
        # -> embeddings_connector.xxx
        return pytorch_key.replace(
            "model.diffusion_model.video_embeddings_connector.",
            "embeddings_connector.",
        )

    # Handle generic embeddings_connector (for audio/AV encoder)
    if "embeddings_connector" in pytorch_key:
        # model.diffusion_model.embeddings_connector.xxx
        # -> embeddings_connector.xxx
        return pytorch_key.replace(
            "model.diffusion_model.embeddings_connector.",
            "embeddings_connector.",
        )

    return None


def convert_upsampler_key(pytorch_key: str) -> str | None:
    """
    Convert upsampler weight key.

    Args:
        pytorch_key: Original PyTorch weight key.

    Returns:
        Converted MLX key, or None if should be skipped.
    """
    # Upsampler weights are stored directly without prefix
    # Just return as-is for now
    return pytorch_key


def extract_transformer_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """
    Extract and convert transformer model weights.

    Args:
        weights: Full weights dictionary.

    Returns:
        Converted transformer weights for MLX model.
    """
    converted = {}

    for key, value in weights.items():
        # Only process diffusion model weights
        if not key.startswith("model.diffusion_model."):
            continue

        # Convert key
        new_key = convert_transformer_key(key)
        if new_key is None:
            continue

        # Note: MLX and PyTorch both store Linear weights as [out_features, in_features],
        # so NO transpose is needed here. See load_transformer_weights() for reference.

        converted[new_key] = value

    return converted


def extract_vae_weights(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """
    Extract encoder and decoder weights from VAE.

    Args:
        weights: Full weights dictionary.

    Returns:
        Tuple of (encoder_weights, decoder_weights).
    """
    encoder_weights = {}
    decoder_weights = {}

    for key, value in weights.items():
        if not key.startswith("vae."):
            continue

        mlx_key = key.replace("vae.", "")

        # Note: MLX and PyTorch both store Linear weights as [out_features, in_features],
        # so NO transpose is needed here.

        if mlx_key.startswith("encoder."):
            encoder_weights[mlx_key.replace("encoder.", "")] = value
        elif mlx_key.startswith("decoder."):
            decoder_weights[mlx_key.replace("decoder.", "")] = value
        else:
            # Shared weights (e.g., per_channel_statistics)
            encoder_weights[mlx_key] = value
            decoder_weights[mlx_key] = value

    return encoder_weights, decoder_weights


def extract_text_encoder_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """
    Extract text encoder weights.

    Args:
        weights: Full weights dictionary.

    Returns:
        Text encoder weights for MLX model.
    """
    converted = {}

    for key, value in weights.items():
        new_key = convert_text_encoder_key(key)
        if new_key is None:
            continue

        # Note: MLX and PyTorch both store Linear weights as [out_features, in_features],
        # so NO transpose is needed here.

        converted[new_key] = value

    return converted


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
    key = re.sub(r"\.ff\.net\.0\.proj\.", ".ff.project_in.proj.", key)

    # Handle ff.net.2 -> ff.project_out
    key = re.sub(r"\.ff\.net\.2\.", ".ff.project_out.", key)

    # Handle audio ff.net.0.proj -> audio_ff.project_in.proj
    key = re.sub(r"\.audio_ff\.net\.0\.proj\.", ".audio_ff.project_in.proj.", key)

    # Handle audio ff.net.2 -> audio_ff.project_out
    key = re.sub(r"\.audio_ff\.net\.2\.", ".audio_ff.project_out.", key)

    return key


def load_transformer_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False,
    include_audio: bool = False,
    streaming: bool = True,
) -> None:
    """
    Load transformer weights into an MLX model.

    Args:
        model: MLX model to load weights into.
        weights_path: Path to safetensors file.
        strict: If True, raise error on missing/extra keys.
        include_audio: If True, include audio-related weights (for AudioVideo model).
        streaming: If True, use memory-efficient streaming load (default True).
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


def save_mlx_weights(weights: dict[str, mx.array], path: str) -> None:
    """
    Save weights in MLX format (npz).

    Args:
        weights: Dictionary of weights.
        path: Output path.
    """
    mx.savez(path, **weights)


def load_mlx_weights(path: str) -> dict[str, mx.array]:
    """
    Load weights from MLX format (npz).

    Args:
        path: Path to npz file.

    Returns:
        Dictionary of weights.
    """
    return dict(mx.load(path))


def load_av_transformer_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False,
) -> None:
    """
    Load AudioVideo transformer weights into an MLX model.

    Convenience function that calls load_transformer_weights with include_audio=True.

    Args:
        model: MLX AudioVideo model to load weights into.
        weights_path: Path to safetensors file.
        strict: If True, raise error on missing/extra keys.
    """
    load_transformer_weights(
        model=model,
        weights_path=weights_path,
        strict=strict,
        include_audio=True,
    )
