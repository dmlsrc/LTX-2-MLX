"""Gemma feature extractor for LTX-2."""

import math

import mlx.core as mx
import mlx.nn as nn


def norm_and_concat_per_token_rms(
    encoded_text: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    """Per-token RMS normalization for V2 models.

    Args:
        encoded_text: [B, T, D, L]
        attention_mask: [B, T] binary mask

    Returns:
        [B, T, D*L] normalized tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape
    variance = mx.mean(encoded_text ** 2, axis=2, keepdims=True)  # [B,T,1,L]
    normed = encoded_text * mx.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.astype(mx.bool_)[:, :, None]  # [B, T, 1]
    return mx.where(mask_3d, normed, mx.zeros_like(normed))


class GemmaFeaturesExtractorV2(nn.Module):
    """V2 feature extractor for LTX-2.3 (22B).

    Uses per-token RMS normalization and dual aggregate embeddings
    that project directly to transformer-native dimensions.

    Returns separate video and audio features at different dimensions.
    """

    def __init__(
        self,
        hidden_dim: int = 3840,
        num_layers: int = 49,
        video_inner_dim: int = 4096,
        audio_inner_dim: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        flat_dim = hidden_dim * num_layers
        self.embedding_dim = hidden_dim

        self.video_aggregate_embed = nn.Linear(flat_dim, video_inner_dim, bias=True)
        self.audio_aggregate_embed = nn.Linear(flat_dim, audio_inner_dim, bias=True)

    def extract_from_hidden_states(
        self,
        hidden_states: list,
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> tuple:
        """Extract features from Gemma hidden states.

        Returns:
            Tuple of (video_features, audio_features).
        """
        stacked = mx.stack(hidden_states, axis=-1)  # [B, T, D, L]
        normed = norm_and_concat_per_token_rms(stacked, attention_mask)
        normed = normed.astype(stacked.dtype)

        v_dim = self.video_aggregate_embed.weight.shape[0]
        a_dim = self.audio_aggregate_embed.weight.shape[0]

        video_features = self.video_aggregate_embed(
            normed * math.sqrt(v_dim / self.embedding_dim)
        )
        audio_features = self.audio_aggregate_embed(
            normed * math.sqrt(a_dim / self.embedding_dim)
        )
        return video_features, audio_features
