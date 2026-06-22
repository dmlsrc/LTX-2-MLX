"""Video VAE encoder and decoder (native MLX Conv3d, BFHWC)."""

from .native_decoder import (
    NativeConv3dVideoDecoder,
    load_native_vae_decoder_weights,
)
from .native_encoder import (
    NativeConv3dVideoEncoder,
    load_native_vae_encoder_weights,
)
from .ops import (
    PerChannelStatistics,
    patchify,
    pixel_shuffle_3d,
    pixel_unshuffle_3d,
    unpatchify,
)
from .tiling import (
    SpatialTilingConfig,
    TemporalChunkConfig,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    decode_single_pass,
    decode_streaming,
)

__all__ = [
    # Ops
    "patchify",
    "unpatchify",
    "PerChannelStatistics",
    "pixel_shuffle_3d",
    "pixel_unshuffle_3d",
    # Decoder: single-pass (whole clip, one call) + chunked-streaming
    "decode_single_pass",
    "NativeConv3dVideoDecoder",
    "load_native_vae_decoder_weights",
    # Encoder
    "NativeConv3dVideoEncoder",
    "load_native_vae_encoder_weights",
    # Tiling
    "TilingConfig",
    "SpatialTilingConfig",
    "TemporalChunkConfig",
    "decode_streaming",
    "compute_trapezoidal_mask_1d",
]
