"""Native MLX Conv3d video VAE decoder.

The older :mod:`simple_decoder` stores PyTorch-layout Conv3d weights and
emulates each 3D convolution with temporal slices of 2D convolutions. This
decoder stores the same weights in MLX Conv3d layout and runs the VAE in
BFHWC/channel-last form. It works with the existing tiled decode wrapper.
"""

from __future__ import annotations

import gc
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


_SPATIAL_PADDING_MODES = {"reflect", "zero"}


def _validate_spatial_padding_mode(mode: str) -> str:
    if mode not in _SPATIAL_PADDING_MODES:
        allowed = ", ".join(sorted(_SPATIAL_PADDING_MODES))
        raise ValueError(f"Unsupported VAE spatial padding mode {mode!r}; expected one of: {allowed}")
    return mode


def _pixel_norm_bfhwc(x: mx.array, eps: float = 1e-8) -> mx.array:
    """PixelNorm over the channel-last dimension."""
    return mx.fast.rms_norm(x, weight=None, eps=eps)


def _unpatchify_spatial_bfhwc(x: mx.array, patch_size: int = 4) -> mx.array:
    """Reverse the VAE's final spatial patchification in BFHWC layout.

    The reference channel order is ``(C, r_width, q_height)`` for spatial
    patches.  This matches the existing BCFHW ``unpatchify`` helper while
    avoiding an extra layout roundtrip.
    """
    b, f, h, w, c_packed = x.shape
    c = c_packed // (patch_size * patch_size)
    x = x.reshape(b, f, h, w, c, patch_size, patch_size)
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    return x.reshape(b, f, h * patch_size, w * patch_size, c)


class NativeConv3dBlock(nn.Module):
    """MLX Conv3d with explicit temporal/spatial padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        causal: bool = False,
        spatial_padding_mode: str = "zero",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.causal = causal
        self.spatial_padding_mode = _validate_spatial_padding_mode(spatial_padding_mode)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=True,
        )

    def set_spatial_padding_mode(self, mode: str) -> None:
        self.spatial_padding_mode = _validate_spatial_padding_mode(mode)

    def __call__(self, x: mx.array, causal: Optional[bool] = None) -> mx.array:
        """Apply convolution to BFHWC input."""
        causal = self.causal if causal is None else causal
        p = self.padding
        k = self.kernel_size

        if k > 1:
            if causal:
                first = mx.repeat(x[:, :1, :, :, :], k - 1, axis=1)
                x = mx.concatenate([first, x], axis=1)
            else:
                pad_size = (k - 1) // 2
                if pad_size > 0:
                    first = mx.repeat(x[:, :1, :, :, :], pad_size, axis=1)
                    last = mx.repeat(x[:, -1:, :, :, :], pad_size, axis=1)
                    x = mx.concatenate([first, x, last], axis=1)

        if p > 0:
            if self.spatial_padding_mode == "reflect":
                h_top = x[:, :, 1 : p + 1, :, :][:, :, ::-1, :, :]
                h_bottom = x[:, :, -(p + 1) : -1, :, :][:, :, ::-1, :, :]
                x = mx.concatenate([h_top, x, h_bottom], axis=2)
                w_left = x[:, :, :, 1 : p + 1, :][:, :, :, ::-1, :]
                w_right = x[:, :, :, -(p + 1) : -1, :][:, :, :, ::-1, :]
                x = mx.concatenate([w_left, x, w_right], axis=3)
            else:
                x = mx.pad(x, [(0, 0), (0, 0), (p, p), (p, p), (0, 0)])

        return self.conv(x)


class NativeResBlock3d(nn.Module):
    """Pre-activation VAE residual block in BFHWC layout."""

    def __init__(self, channels: int, spatial_padding_mode: str = "zero"):
        super().__init__()
        self.conv1 = NativeConv3dBlock(channels, channels, spatial_padding_mode=spatial_padding_mode)
        self.conv2 = NativeConv3dBlock(channels, channels, spatial_padding_mode=spatial_padding_mode)

    def __call__(self, x: mx.array, causal: bool = False) -> mx.array:
        residual = x
        x = self.conv1(nn.silu(_pixel_norm_bfhwc(x)), causal=causal)
        x = self.conv2(nn.silu(_pixel_norm_bfhwc(x)), causal=causal)
        return x + residual


class NativeResBlockGroup(nn.Module):
    """Group of fixed-width residual blocks."""

    def __init__(self, channels: int, num_blocks: int, spatial_padding_mode: str = "zero"):
        super().__init__()
        self.res_blocks = [
            NativeResBlock3d(channels, spatial_padding_mode=spatial_padding_mode)
            for _ in range(num_blocks)
        ]

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep: Optional[mx.array] = None,
    ) -> mx.array:
        del timestep
        for block in self.res_blocks:
            x = block(x, causal=causal)
        return x


class NativeDepthToSpaceUpsample3d(nn.Module):
    """Conv3d followed by depth-to-space in BFHWC layout."""

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: str = "zero",
    ):
        super().__init__()
        self.stride = stride
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        stride_product = math.prod(stride)
        self.final_out_channels = in_channels // out_channels_reduction_factor
        conv_out_channels = stride_product * in_channels // out_channels_reduction_factor
        self.conv = NativeConv3dBlock(
            in_channels,
            conv_out_channels,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _depth_to_space(self, x: mx.array, out_channels: int) -> mx.array:
        b, f, h, w, _c = x.shape
        ft, fh, fw = self.stride
        x = x.reshape(b, f, h, w, out_channels, ft, fh, fw)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(b, f * ft, h * fh, w * fw, out_channels)

    def __call__(self, x: mx.array, causal: bool = False) -> mx.array:
        ft, fh, fw = self.stride
        stride_product = ft * fh * fw

        if self.residual:
            residual_channels = x.shape[-1] // stride_product
            residual = self._depth_to_space(x, residual_channels)
            if ft > 1:
                residual = residual[:, 1:, :, :, :]
            num_repeat = stride_product // self.out_channels_reduction_factor
            residual = mx.tile(residual, (1, 1, 1, 1, num_repeat))

        x = self.conv(x, causal=causal)
        x = self._depth_to_space(x, self.final_out_channels)

        if ft > 1:
            x = x[:, 1:, :, :, :]

        if self.residual:
            x = x + residual
        return x


_STRIDE_MAP = {
    "compress_all": (2, 2, 2),
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
}


class NativeConv3dVideoDecoder(nn.Module):
    """Config-driven LTX video VAE decoder using native MLX Conv3d."""

    def __init__(
        self,
        decoder_blocks: Optional[list] = None,
        base_channels: int = 128,
        timestep_conditioning: bool = False,
        compute_dtype: mx.Dtype = mx.bfloat16,
        spatial_padding_mode: str = "zero",
        causal: bool = False,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.timestep_conditioning = timestep_conditioning
        self.spatial_padding_mode = _validate_spatial_padding_mode(spatial_padding_mode)
        self.causal = causal

        if timestep_conditioning:
            raise ValueError(
                "Native Conv3d VAE decoder currently supports LTX-2.3-style "
                "decoders without timestep conditioning."
            )
        if decoder_blocks is None:
            raise ValueError("Native Conv3d VAE decoder requires decoder_blocks from checkpoint metadata.")

        self.mean_of_means = mx.zeros((128,), dtype=mx.float32)
        self.std_of_means = mx.ones((128,), dtype=mx.float32)

        feature_channels = base_channels * 8
        self.conv_in = NativeConv3dBlock(
            128,
            feature_channels,
            causal=causal,
            spatial_padding_mode=self.spatial_padding_mode,
        )
        self.up_blocks = []
        self.block_types = []

        for block_name, block_params in reversed(decoder_blocks):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params
            if block_name == "res_x":
                block = NativeResBlockGroup(
                    feature_channels,
                    num_blocks=block_config["num_layers"],
                    spatial_padding_mode=self.spatial_padding_mode,
                )
                self.up_blocks.append(block)
                self.block_types.append("res")
            elif block_name in _STRIDE_MAP:
                multiplier = block_config.get("multiplier", 1)
                block = NativeDepthToSpaceUpsample3d(
                    in_channels=feature_channels,
                    stride=_STRIDE_MAP[block_name],
                    residual=block_config.get("residual", False),
                    out_channels_reduction_factor=multiplier,
                    spatial_padding_mode=self.spatial_padding_mode,
                )
                feature_channels = feature_channels // multiplier
                self.up_blocks.append(block)
                self.block_types.append("upsample")
            else:
                raise ValueError(f"Unsupported native Conv3d decoder block: {block_name}")

        self.final_channels = feature_channels
        self.conv_out = NativeConv3dBlock(
            feature_channels,
            48,
            causal=causal,
            spatial_padding_mode=self.spatial_padding_mode,
        )

    def _iter_convs(self):
        yield "conv_in.conv", self.conv_in
        for idx, (block, btype) in enumerate(zip(self.up_blocks, self.block_types)):
            if btype == "res":
                for res_idx, res_block in enumerate(block.res_blocks):
                    yield f"up_blocks.{idx}.res_blocks.{res_idx}.conv1.conv", res_block.conv1
                    yield f"up_blocks.{idx}.res_blocks.{res_idx}.conv2.conv", res_block.conv2
            else:
                yield f"up_blocks.{idx}.conv.conv", block.conv
        yield "conv_out.conv", self.conv_out

    def __call__(
        self,
        latent: mx.array,
        timestep: Optional[float] = 0.05,
        show_progress: bool = True,
        causal: Optional[bool] = None,
    ) -> mx.array:
        del timestep, show_progress
        causal = self.causal if causal is None else causal

        if latent.ndim == 4:
            latent = latent[None]
        if latent.dtype != self.compute_dtype:
            latent = latent.astype(self.compute_dtype)

        x = latent.transpose(0, 2, 3, 4, 1)
        x = x * self.std_of_means.reshape(1, 1, 1, 1, -1)
        x = x + self.mean_of_means.reshape(1, 1, 1, 1, -1)
        if x.dtype != self.compute_dtype:
            x = x.astype(self.compute_dtype)

        x = self.conv_in(x, causal=causal)
        mx.eval(x)

        for block, btype in zip(self.up_blocks, self.block_types):
            if btype == "res":
                x = block(x, causal=causal)
            else:
                x = block(x, causal=causal)
            mx.eval(x)

        x = self.conv_out(nn.silu(_pixel_norm_bfhwc(x)), causal=causal)
        mx.eval(x)
        x = _unpatchify_spatial_bfhwc(x, patch_size=4)
        mx.eval(x)

        return x.transpose(0, 4, 1, 2, 3)


def _lookup_weight(weights: dict, *keys: str):
    for key in keys:
        if key in weights:
            return weights[key]
    return None


def _to_native_conv3d_layout(value: mx.array, expected_shape: tuple[int, ...]) -> mx.array:
    """Convert stock PyTorch Conv3d weights to MLX Conv3d layout when needed."""
    if tuple(value.shape) == expected_shape:
        return value
    if value.ndim == 5:
        converted = value.transpose(0, 2, 3, 4, 1)
        if tuple(converted.shape) == expected_shape:
            return converted
    raise ValueError(f"Cannot load Conv3d weight with shape {value.shape}; expected {expected_shape}")


def load_native_vae_decoder_weights(decoder: NativeConv3dVideoDecoder, weights_path: str) -> None:
    """Load VAE decoder weights from stock or split safetensors."""
    print(f"Loading native Conv3d VAE decoder weights from {weights_path}...")
    weights = mx.load(weights_path)
    loaded_count = 0

    mean = _lookup_weight(
        weights,
        "vae.per_channel_statistics.mean-of-means",
        "vae_decoder.per_channel_statistics.mean",
    )
    std = _lookup_weight(
        weights,
        "vae.per_channel_statistics.std-of-means",
        "vae_decoder.per_channel_statistics.std",
    )
    if mean is not None:
        decoder.mean_of_means = mean
        loaded_count += 1
    if std is not None:
        decoder.std_of_means = std
        loaded_count += 1

    for local_prefix, conv_block in decoder._iter_convs():
        for suffix in ("weight", "bias"):
            local_key = f"{local_prefix}.{suffix}"
            value = _lookup_weight(
                weights,
                f"vae.decoder.{local_key}",
                f"vae_decoder.{local_key}",
            )
            if value is None:
                continue
            if suffix == "weight":
                value = _to_native_conv3d_layout(value, tuple(conv_block.conv.weight.shape))
            setattr(conv_block.conv, suffix, value)
            loaded_count += 1

    del weights
    gc.collect()
    mx.clear_cache()
    print(f"  Loaded {loaded_count} native Conv3d VAE tensors")
