"""Shared Conv3d primitives for the BFHWC video VAE encoder + decoder.

Single source of truth for ``NativeConv3dBlock``, ``NativeResBlock3d``,
``NativeResBlockGroup`` and their padding / normalization helpers.
Zero spatial padding only.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def pixel_norm_bfhwc(x: mx.array, eps: float = 1e-8) -> mx.array:
    """PixelNorm over the channel-last dimension."""
    return mx.fast.rms_norm(x, weight=None, eps=eps)


def to_native_conv3d_layout(value: mx.array, expected_shape: tuple[int, ...]) -> mx.array:
    """Convert stock PyTorch Conv3d weights to MLX Conv3d layout when needed.

    Idempotent: when the loaded tensor already matches the target shape
    (cache schema >= 2 stores channels-last directly), returns unchanged.
    """
    if tuple(value.shape) == expected_shape:
        return value
    if value.ndim == 5:
        # PyTorch (O, I, T, H, W) -> MLX (O, T, H, W, I)
        converted = value.transpose(0, 2, 3, 4, 1)
        if tuple(converted.shape) == expected_shape:
            return converted
    raise ValueError(f"Cannot load Conv3d weight with shape {value.shape}; expected {expected_shape}")


def lookup_weight(weights: dict, *keys: str) -> Optional[mx.array]:
    """First-hit lookup across alternative key names (handles both stock
    LTX safetensors layout ``vae.encoder.*`` / ``vae.decoder.*`` and
    pre-split safetensors layout ``vae_encoder.*`` / ``vae_decoder.*``)."""
    for key in keys:
        if key in weights:
            return weights[key]
    return None


class NativeConv3dBlock(nn.Module):
    """MLX Conv3d with explicit temporal/spatial padding (BFHWC input)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.causal = causal
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array, causal: Optional[bool] = None) -> mx.array:
        """Apply convolution to BFHWC input."""
        causal = self.causal if causal is None else causal
        p = self.padding
        k = self.kernel_size

        if k > 1:
            if causal:
                # Causal temporal padding: replicate first frame.
                first = mx.repeat(x[:, :1, :, :, :], k - 1, axis=1)
                x = mx.concatenate([first, x], axis=1)
            else:
                # Non-causal symmetric: replicate first and last frames.
                pad_size = (k - 1) // 2
                if pad_size > 0:
                    first = mx.repeat(x[:, :1, :, :, :], pad_size, axis=1)
                    last = mx.repeat(x[:, -1:, :, :, :], pad_size, axis=1)
                    x = mx.concatenate([first, x, last], axis=1)

        if p > 0:
            # Zero spatial padding (NHWC: pad H and W axes only).
            x = mx.pad(x, [(0, 0), (0, 0), (p, p), (p, p), (0, 0)])

        return self.conv(x)


class NativeResBlock3d(nn.Module):
    """Pre-activation VAE residual block in BFHWC layout.

    Shared between encoder and decoder -- the layer geometry is identical
    (``norm -> silu -> conv -> norm -> silu -> conv -> add residual``);
    decoder ResBlockGroups also pass through this class, just stacked.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = NativeConv3dBlock(channels, channels)
        self.conv2 = NativeConv3dBlock(channels, channels)

    def __call__(self, x: mx.array, causal: bool = False) -> mx.array:
        residual = x
        x = self.conv1(nn.silu(pixel_norm_bfhwc(x)), causal=causal)
        x = self.conv2(nn.silu(pixel_norm_bfhwc(x)), causal=causal)
        return x + residual


class NativeResBlockGroup(nn.Module):
    """Group of fixed-width residual blocks (BFHWC).  Shared encoder/decoder."""

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.res_blocks = [NativeResBlock3d(channels) for _ in range(num_blocks)]

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep: Optional[mx.array] = None,
    ) -> mx.array:
        del timestep  # Unconditional; LTX 2.3 has no per-block timestep injection.
        for block in self.res_blocks:
            x = block(x, causal=causal)
        return x


def unpatchify_spatial_bfhwc(x: mx.array, patch_size: int = 4) -> mx.array:
    """Reverse the VAE's final spatial patchification in BFHWC layout.

    Decoder-side helper; lives here so the matching ``patchify`` in
    ops.py can be paired with it if needed.  Channel order is
    ``(C, r_width, q_height)`` for spatial patches (matches the BCFHW
    ``unpatchify`` in ``ops.py`` while avoiding a layout roundtrip).
    """
    b, f, h, w, c_packed = x.shape
    c = c_packed // (patch_size * patch_size)
    x = x.reshape(b, f, h, w, c, patch_size, patch_size)
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    return x.reshape(b, f, h * patch_size, w * patch_size, c)


def patchify_spatial_bfhwc(x: mx.array, patch_size: int = 4) -> mx.array:
    """Forward spatial patchification in BFHWC layout (encoder side).

    Inverse of ``unpatchify_spatial_bfhwc``: takes ``(B, F, H, W, C)``
    and produces ``(B, F, H/patch, W/patch, C * patch * patch)``
    matching the reference ``(C, r_width, q_height)`` channel order.
    Used by the encoder's input patchify step.
    """
    b, f, h, w, c = x.shape
    p = patch_size
    if h % p != 0 or w % p != 0:
        raise ValueError(
            f"patchify_spatial_bfhwc: H={h} and W={w} must be divisible by patch_size={p}"
        )
    x = x.reshape(b, f, h // p, p, w // p, p, c)
    # Match unpatchify's inverse order: forward perm (0, 1, 2, 4, 6, 5, 3)
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)
    return x.reshape(b, f, h // p, w // p, c * p * p)
