"""Video VAE encoder and decoder."""

from .convolution import (
    DualConv3d,
    CausalConv3d,
    Conv3d,
    PointwiseConv3d,
    make_conv_nd,
    make_linear_nd,
    PaddingModeType,
    NormLayerType,
)
from .ops import (
    patchify,
    unpatchify,
    PerChannelStatistics,
    pixel_shuffle_3d,
    pixel_unshuffle_3d,
)
from .resnet import ResnetBlock3D, UNetMidBlock3D, PixelNorm
from .sampling import (
    SpaceToDepthDownsample,
    DepthToSpaceUpsample,
    space_to_depth,
    depth_to_space,
)
from .encoder import VideoEncoder, LogVarianceType
from .decoder import VideoDecoder, decode_video
from .decode_utils import decode_latent
from .native_decoder import (
    NativeConv3dVideoDecoder,
    load_native_vae_decoder_weights,
)
from .simple_encoder import (
    SimpleVideoEncoder,
    load_vae_encoder_weights,
    encode_video,
)
from .tiling import (
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
    decode_tiled,
    compute_trapezoidal_mask_1d,
)

__all__ = [
    # Convolution
    "DualConv3d",
    "CausalConv3d",
    "Conv3d",
    "PointwiseConv3d",
    "make_conv_nd",
    "make_linear_nd",
    "PaddingModeType",
    "NormLayerType",
    # Ops
    "patchify",
    "unpatchify",
    "PerChannelStatistics",
    "pixel_shuffle_3d",
    "pixel_unshuffle_3d",
    # ResNet
    "ResnetBlock3D",
    "UNetMidBlock3D",
    "PixelNorm",
    # Sampling
    "SpaceToDepthDownsample",
    "DepthToSpaceUpsample",
    "space_to_depth",
    "depth_to_space",
    # Encoder/Decoder
    "VideoEncoder",
    "VideoDecoder",
    "LogVarianceType",
    "decode_video",
    # Video VAE decoder (production = NativeConv3dVideoDecoder; legacy
    # SimpleVideoDecoder was archived 2026-05-23, see pipelines/archive/)
    "decode_latent",
    "NativeConv3dVideoDecoder",
    "load_native_vae_decoder_weights",
    # Simple encoder (for weight loading)
    "SimpleVideoEncoder",
    "load_vae_encoder_weights",
    "encode_video",
    # Tiling
    "TilingConfig",
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "decode_tiled",
    "compute_trapezoidal_mask_1d",
]
