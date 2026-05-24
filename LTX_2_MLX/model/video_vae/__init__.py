"""Video VAE encoder and decoder (native MLX Conv3d, BFHWC)."""

from .ops import (
    patchify,
    unpatchify,
    PerChannelStatistics,
    pixel_shuffle_3d,
    pixel_unshuffle_3d,
)
from .decode_utils import decode_latent
from .native_decoder import (
    NativeConv3dVideoDecoder,
    load_native_vae_decoder_weights,
)
from .native_encoder import (
    NativeConv3dVideoEncoder,
    load_native_vae_encoder_weights,
)
from .tiling import (
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
    decode_tiled,
    compute_trapezoidal_mask_1d,
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
