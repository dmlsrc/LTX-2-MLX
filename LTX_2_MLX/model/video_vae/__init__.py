"""Video VAE encoder and decoder (native MLX Conv3d, BFHWC)."""

from .decode_utils import decode_latent
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
    TemporalTilingConfig,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    decode_tiled,
)

__all__ = [
    # Ops
    "patchify",
    "unpatchify",
    "PerChannelStatistics",
    "pixel_shuffle_3d",
    "pixel_unshuffle_3d",
    # Decoder
    "decode_latent",
    "NativeConv3dVideoDecoder",
    "load_native_vae_decoder_weights",
    # Encoder
    "NativeConv3dVideoEncoder",
    "load_native_vae_encoder_weights",
    # Tiling
    "TilingConfig",
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "decode_tiled",
    "compute_trapezoidal_mask_1d",
]
