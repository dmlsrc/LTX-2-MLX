"""Native MLX Conv3d video VAE decoder.

Mirrors ``NativeConv3dVideoEncoder`` on the shared blocks in
``native_blocks.py``.  Works with the existing tiled decode wrapper.
"""

from __future__ import annotations

import gc
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .native_blocks import (
    NativeConv3dBlock,
    NativeResBlockGroup,
    lookup_weight,
    pixel_norm_bfhwc,
    to_native_conv3d_layout,
    unpatchify_spatial_bfhwc,
)


class NativeDepthToSpaceUpsample3d(nn.Module):
    """Conv3d followed by depth-to-space in BFHWC layout."""

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: str = "conv",
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
        causal: bool = False,
        spatial_padding_mode: str = "conv",
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.timestep_conditioning = timestep_conditioning
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
            spatial_padding_mode=spatial_padding_mode,
        )
        self.up_blocks = []
        self.block_types = []

        for block_name, block_params in reversed(decoder_blocks):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params
            if block_name == "res_x":
                block = NativeResBlockGroup(
                    feature_channels,
                    num_blocks=block_config["num_layers"],
                    spatial_padding_mode=spatial_padding_mode,
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
                    spatial_padding_mode=spatial_padding_mode,
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
            spatial_padding_mode=spatial_padding_mode,
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

        x = self.conv_out(nn.silu(pixel_norm_bfhwc(x)), causal=causal)
        mx.eval(x)
        x = unpatchify_spatial_bfhwc(x, patch_size=4)
        mx.eval(x)

        return x.transpose(0, 4, 1, 2, 3)


def load_native_vae_decoder_weights(decoder: NativeConv3dVideoDecoder, weights_path: str) -> None:
    """Load VAE decoder weights from stock or split safetensors."""
    print(f"Loading native Conv3d VAE decoder weights from {weights_path}...")
    weights = mx.load(weights_path)
    loaded_count = 0

    mean = lookup_weight(
        weights,
        "vae.per_channel_statistics.mean-of-means",
        "vae_decoder.per_channel_statistics.mean",
    )
    std = lookup_weight(
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
            value = lookup_weight(
                weights,
                f"vae.decoder.{local_key}",
                f"vae_decoder.{local_key}",
            )
            if value is None:
                continue
            if suffix == "weight":
                value = to_native_conv3d_layout(value, tuple(conv_block.conv.weight.shape))
            setattr(conv_block.conv, suffix, value)
            loaded_count += 1

    del weights
    gc.collect()
    mx.clear_cache()
    print(f"  Loaded {loaded_count} native Conv3d VAE tensors")
