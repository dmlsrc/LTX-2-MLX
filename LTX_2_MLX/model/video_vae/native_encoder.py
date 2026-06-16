"""Native MLX Conv3d video VAE encoder.

Mirrors ``NativeConv3dVideoDecoder`` on the shared blocks in
``native_blocks.py``.  Takes BCFHW ``(B, 3, F, H, W)`` in [-1, 1] and
returns the normalized BCFHW latent ``(B, 128, F', H', W')``.
``per_channel_statistics`` is exposed for callers that need to
de/re-normalize between stages.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .native_blocks import (
    NativeConv3dBlock,
    NativeResBlockGroup,
    lookup_weight,
    patchify_spatial_bfhwc,
    pixel_norm_bfhwc,
    to_native_conv3d_layout,
)
from .ops import PerChannelStatistics

_STRIDE_MAP = {
    "compress_all": (2, 2, 2),
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
}

_STAT_KEY_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("mean-of-means", "mean"), "mean_of_means"),
    (("std-of-means", "std"), "std_of_means"),
    (("mean-of-stds",), "mean_of_stds"),
    (
        (
            "mean-of-stds-over-std-of-means",
            "mean-of-stds_over_std-of-means",
        ),
        "mean_of_stds_over_std_of_means",
    ),
    (("channel",), "channel"),
)


class NativeConv3dVideoEncoderStatistics:
    """Small adapter exposing only encoder latent normalization statistics.

    Distilled two-stage upscaling needs ``encoder.per_channel_statistics`` for
    latent de-normalize/re-normalize, but text-only runs do not need any of the
    heavyweight encoder convolution weights.
    """

    def __init__(self, per_channel_statistics: PerChannelStatistics | None = None):
        self.per_channel_statistics = (
            per_channel_statistics
            if per_channel_statistics is not None
            else PerChannelStatistics(latent_channels=128)
        )


def _load_per_channel_statistics(
    stats: PerChannelStatistics,
    weights: dict,
) -> int:
    loaded_count = 0
    for stat_keys, attr_name in _STAT_KEY_ALIASES:
        candidates = []
        for stat_key in stat_keys:
            candidates.extend([
                f"vae.per_channel_statistics.{stat_key}",
                f"vae_encoder.per_channel_statistics.{stat_key}",
                f"per_channel_statistics.{stat_key}",
            ])
        value = lookup_weight(weights, *candidates)
        if value is None:
            continue
        setattr(stats, attr_name, value)
        loaded_count += 1
    return loaded_count


class NativeSpaceToDepthDownsample3d(nn.Module):
    """Conv3d followed by space-to-depth in BFHWC layout.

    Inverse of ``NativeDepthToSpaceUpsample3d`` from ``native_decoder.py``:
    convolves down to a reduced channel count, then packs the spatial
    stride factors into the channel dimension.  Optionally adds a
    residual path that space-to-depth-packs the input and mean-pools
    across the grouped channels to match ``out_channels``.

    Matches the PyTorch reference's ``SpaceToDepthDownsample`` block
    that ``simple_encoder.SpaceToDepthDownsample3d`` was emulating in
    BCFHW.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] = (2, 2, 2),
        residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.residual = residual

        stride_product = math.prod(stride)
        # group_size for residual mean: how many input channels collapse
        # into one output channel via group-mean.
        self.group_size = (in_channels * stride_product) // out_channels
        # Conv emits one channel per output channel divided by the
        # space-to-depth packing factor.  Final out_channels = conv_out
        # * stride_product.
        self.conv_out_channels = out_channels // stride_product
        self.conv = NativeConv3dBlock(in_channels, self.conv_out_channels)

    def _space_to_depth_bfhwc(self, x: mx.array) -> mx.array:
        """Pack stride factors into the channel dim (BFHWC)."""
        b, t, h, w, c = x.shape
        st, sh, sw = self.stride
        # (b, t/st, st, h/sh, sh, w/sw, sw, c)
        x = x.reshape(b, t // st, st, h // sh, sh, w // sw, sw, c)
        # Pack (st, sh, sw) into channel dim: keep b, t/st, h/sh, w/sw; trailing dims c,st,sh,sw
        # axes: (b=0, t/st=1, st=2, h/sh=3, sh=4, w/sw=5, sw=6, c=7)
        # target: (b, t/st, h/sh, w/sw, c, st, sh, sw)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        return x.reshape(b, t // st, h // sh, w // sw, c * st * sh * sw)

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        """Downsample via conv then space-to-depth, optionally with residual."""
        # Causal temporal pad on input (replicate first frame) BEFORE the
        # space-to-depth step so the downsampled output preserves the
        # ``(F - 1) % 8 == 0`` invariant the upstream pipeline expects.
        # SimpleVideoEncoder did this only for stride[0]==2; mirror the
        # same condition exactly.
        st, _, _ = self.stride
        if st == 2:
            x = mx.concatenate([x[:, :1, :, :, :], x], axis=1)

        # Residual path: space-to-depth on input, then mean-pool groups
        # of in_channels*stride_product / out_channels back down to
        # out_channels.
        if self.residual:
            x_in = self._space_to_depth_bfhwc(x)
            b_r, t_r, h_r, w_r, _ = x_in.shape
            x_in = x_in.reshape(b_r, t_r, h_r, w_r, self.out_channels, self.group_size)
            x_in = mx.mean(x_in, axis=-1)
        else:
            x_in = None

        # Main path: conv then space-to-depth.
        x = self.conv(x, causal=causal)
        x = self._space_to_depth_bfhwc(x)

        if x_in is not None:
            x = x + x_in
        return x


class NativeConv3dVideoEncoder(nn.Module):
    """LTX-2.3 video VAE encoder using native MLX Conv3d.

    Architecture (from PyTorch reference; matches SimpleVideoEncoder):
    - patchify: 3 -> 48 channels (patch_size=4)
    - conv_in: 48 -> 128
    - down_blocks.0: 4 res blocks (128 ch)
    - down_blocks.1: SpaceToDepthDownsample (128 -> 256, stride (1,2,2))
    - down_blocks.2: 6 res blocks (256 ch)
    - down_blocks.3: SpaceToDepthDownsample (256 -> 512, stride (2,1,1))
    - down_blocks.4: 6 res blocks (512 ch)
    - down_blocks.5: SpaceToDepthDownsample (512 -> 1024, stride (2,2,2))
    - down_blocks.6: 2 res blocks (1024 ch)
    - down_blocks.7: SpaceToDepthDownsample (1024 -> 1024, stride (2,2,2))
    - down_blocks.8: 2 res blocks (1024 ch)
    - pixel_norm + SiLU
    - conv_out: 1024 -> 129 (128 means + 1 uniform logvar; logvar discarded)
    - per-channel normalize

    Total compression: 1:192 (32x spatial = 4x patchify * 8x d2s; 8x temporal).
    """

    def __init__(self, compute_dtype: mx.Dtype = mx.bfloat16):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.patch_size = 4

        # Per-channel statistics applied at the output (latent space).
        self.per_channel_statistics = PerChannelStatistics(latent_channels=128)

        # Conv in: 48 (patchified RGB) -> 128
        self.conv_in = NativeConv3dBlock(48, 128)

        # Down blocks - exact architecture from the SimpleVideoEncoder reference.
        self.down_blocks_0 = NativeResBlockGroup(128, num_blocks=4)
        self.down_blocks_1 = NativeSpaceToDepthDownsample3d(128, 256, stride=(1, 2, 2))
        self.down_blocks_2 = NativeResBlockGroup(256, num_blocks=6)
        self.down_blocks_3 = NativeSpaceToDepthDownsample3d(256, 512, stride=(2, 1, 1))
        self.down_blocks_4 = NativeResBlockGroup(512, num_blocks=6)
        self.down_blocks_5 = NativeSpaceToDepthDownsample3d(512, 1024, stride=(2, 2, 2))
        self.down_blocks_6 = NativeResBlockGroup(1024, num_blocks=2)
        self.down_blocks_7 = NativeSpaceToDepthDownsample3d(1024, 1024, stride=(2, 2, 2))
        self.down_blocks_8 = NativeResBlockGroup(1024, num_blocks=2)

        # Conv out: 1024 -> 129 (128 means + 1 uniform logvar)
        self.conv_out = NativeConv3dBlock(1024, 129)

    def __call__(self, video: mx.array, show_progress: bool = True) -> mx.array:
        """
        Encode video to normalized latent.

        Args:
            video: Video tensor ``(B, 3, F, H, W)`` in [-1, 1].
                   F must satisfy ``(F - 1) % 8 == 0`` (e.g., 1, 9, 17, 25...).
            show_progress: Whether to show progress bar.

        Returns:
            Normalized latent tensor ``(B, 128, F', H', W')`` (BCFHW) where
            F' = 1 + (F - 1) / 8, H' = H / 32, W' = W / 32.
        """
        pbar = None
        try:
            from tqdm import tqdm
            has_tqdm = show_progress
        except ImportError:
            has_tqdm = False
            tqdm = None

        # Validate frame count
        frames_count = video.shape[2]
        if (frames_count - 1) % 8 != 0:
            raise ValueError(
                f"Invalid number of frames: {frames_count}. "
                f"Encoder input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)."
            )

        # Cast to compute dtype for memory efficiency
        if self.compute_dtype != mx.float32:
            video = video.astype(self.compute_dtype)

        # BCFHW -> BFHWC for native channels-last processing
        x = video.transpose(0, 2, 3, 4, 1)

        def step(x_in, block, desc):
            nonlocal pbar
            x_out = block(x_in, causal=True)
            mx.eval(x_out)
            if has_tqdm and pbar is not None:
                pbar.update(1)
                pbar.set_description(desc)
            return x_out

        if has_tqdm:
            pbar = tqdm(total=12, desc="VAE encode", ncols=80, ascii=True, mininterval=2.0)

        # Patchify: (B, F, H, W, 3) -> (B, F, H/4, W/4, 48) (BFHWC)
        x = patchify_spatial_bfhwc(x, patch_size=self.patch_size)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("patchify done")

        # Conv in
        x = self.conv_in(x, causal=True)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("conv_in done")

        # Down blocks
        x = step(x, self.down_blocks_0, "down_block 0")  # 128ch
        x = step(x, self.down_blocks_1, "down_block 1")  # 128->256, compress_space
        x = step(x, self.down_blocks_2, "down_block 2")  # 256ch
        x = step(x, self.down_blocks_3, "down_block 3")  # 256->512, compress_time
        x = step(x, self.down_blocks_4, "down_block 4")  # 512ch
        x = step(x, self.down_blocks_5, "down_block 5")  # 512->1024, compress_all
        x = step(x, self.down_blocks_6, "down_block 6")  # 1024ch
        x = step(x, self.down_blocks_7, "down_block 7")  # 1024->1024, compress_all
        x = step(x, self.down_blocks_8, "down_block 8")  # 1024ch

        # Final norm and activation
        x = pixel_norm_bfhwc(x)
        x = nn.silu(x)

        # Conv out: 1024 -> 129 (128 means + 1 uniform logvar)
        x = self.conv_out(x, causal=True)
        mx.eval(x)
        if has_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_description("conv_out done")
            pbar.close()

        # BFHWC (B, F', H', W', 129) -> BCFHW (B, 129, F', H', W')
        x = x.transpose(0, 4, 1, 2, 3)

        # Extract means (first 128 channels), discard logvar
        means = x[:, :128, :, :, :]

        # Per-channel normalization
        means = self.per_channel_statistics.normalize(means)

        # Cast output back to float32 (matches SimpleVideoEncoder behavior)
        if self.compute_dtype != mx.float32:
            means = means.astype(mx.float32)

        return means

    def _iter_convs(self):
        """Yield (pytorch_key_prefix, conv_block) pairs for weight loading."""
        yield "conv_in.conv", self.conv_in
        for idx, group in enumerate([
            self.down_blocks_0, self.down_blocks_1, self.down_blocks_2,
            self.down_blocks_3, self.down_blocks_4, self.down_blocks_5,
            self.down_blocks_6, self.down_blocks_7, self.down_blocks_8,
        ]):
            if isinstance(group, NativeResBlockGroup):
                for res_idx, res_block in enumerate(group.res_blocks):
                    yield f"down_blocks.{idx}.res_blocks.{res_idx}.conv1.conv", res_block.conv1
                    yield f"down_blocks.{idx}.res_blocks.{res_idx}.conv2.conv", res_block.conv2
            elif isinstance(group, NativeSpaceToDepthDownsample3d):
                yield f"down_blocks.{idx}.conv.conv", group.conv
            else:
                raise TypeError(f"Unexpected down-block type: {type(group).__name__}")
        yield "conv_out.conv", self.conv_out


def load_native_vae_encoder_weights(
    encoder: NativeConv3dVideoEncoder, weights_path: str
) -> None:
    """Load native VAE encoder weights from stock or split safetensors.

    Idempotent with respect to weight layout: if the safetensors stores
    channels-last (schema_version >= 2 family cache) the loader
    short-circuits; if PyTorch channels-second (raw checkpoint or
    ``--weights-cache off``), it transposes (0, 2, 3, 4, 1) on load.
    """
    print(f"Loading native Conv3d VAE encoder weights from {weights_path}...")
    weights = mx.load(weights_path)
    loaded_count = 0

    # Per-channel statistics (latent normalizer)
    loaded_count += _load_per_channel_statistics(
        encoder.per_channel_statistics,
        weights,
    )

    # Conv weights + biases
    for local_prefix, conv_block in encoder._iter_convs():
        for suffix in ("weight", "bias"):
            local_key = f"{local_prefix}.{suffix}"
            value = lookup_weight(
                weights,
                f"vae.encoder.{local_key}",
                f"vae_encoder.{local_key}",
            )
            if value is None:
                continue
            if suffix == "weight":
                value = to_native_conv3d_layout(value, tuple(conv_block.conv.weight.shape))
            setattr(conv_block.conv, suffix, value)
            loaded_count += 1

    print(f"  Loaded {loaded_count} native Conv3d VAE encoder tensors")


def load_native_vae_encoder_statistics(
    weights_path: str,
) -> NativeConv3dVideoEncoderStatistics:
    """Load only the VAE encoder latent normalization statistics."""
    print(f"Loading native Conv3d VAE encoder statistics from {weights_path}...")
    weights = mx.load(weights_path)
    stats_encoder = NativeConv3dVideoEncoderStatistics()
    loaded_count = _load_per_channel_statistics(
        stats_encoder.per_channel_statistics,
        weights,
    )
    if loaded_count == 0:
        raise ValueError(
            "No VAE encoder per-channel statistics found in "
            f"{weights_path}"
        )
    mx.eval(*(
        getattr(stats_encoder.per_channel_statistics, attr_name)
        for _aliases, attr_name in _STAT_KEY_ALIASES
    ))
    print(f"  Loaded {loaded_count} native Conv3d VAE encoder statistic tensors")
    return stats_encoder
