"""Polymorphic ``decode_latent`` -- tiled-decode entry point that
accepts any decoder callable as ``decoder(latent, timestep=...)``.
Typed ``Any`` to keep this module concrete-class-free.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def decode_latent(
    latent: mx.array,
    decoder: Any,
    timestep: float | None = 0.05,
    key: mx.array | None = None,
    temporal_chunk_size: int = 7,
    temporal_overlap: int = 2,
    dtype: Any | None = None,
) -> mx.array:
    """
    Decode latent to video frames.

    Uses temporal chunking for sequences longer than temporal_chunk_size to work
    around a 3D convolution bug in MLX where early frames of long sequences
    produce noise artifacts. Chunks are decoded independently and blended in
    the overlap region using a linear ramp.

    Args:
        latent: Latent tensor (B, 128, T, H, W) or (128, T, H, W).
        decoder: Loaded video decoder instance (NativeConv3dVideoDecoder in
                 production; any decoder exposing ``__call__(latent,
                 timestep=...) -> mx.array`` works).
        timestep: Timestep for conditioning (default 0.05 for denoising).
                  Use 0.0 for no denoising, None to disable timestep conditioning.
        key: Optional random key for deterministic decoding (reserved for future use).
        temporal_chunk_size: Max latent frames per chunk (default 7, proven clean).
        temporal_overlap: Overlap in latent frames between chunks for blending (default 2).
        dtype: Output precision.
               None (default) or mx.uint8: clip + scale to [0, 255], cast to
               uint8, return (T, H, W, 3). The common case for ffmpeg / PNG
               consumers - backward-compatible.
               A float type (mx.float16 / mx.float32 / mx.bfloat16): return
               the raw decoder output (B, C, T, H, W) in [-1, 1] cast to that
               dtype. Lets higher-precision consumers (VSR's RGBAHalf source,
               10-bit HEVC, 16-bit ProRes) quantize at their own destination
               format instead of paying an 8-bit round-trip here.

               TODO: migrate the existing callers to pass dtype explicitly
               where it matters for output fidelity. Specifically the
               higher-bit-depth tiers in scripts/encode_modes_harness.py
               (h265_10bit_*, h265_rgb_lossless, prores_*) and the pipelines
               under LTX_2_MLX/pipelines/ that hand their decoded video into
               ffmpeg_encoder.encode_video_ffmpeg with tier=hq / export / reference -
               those all currently take uint8 here and then promote back to
               10/16-bit downstream, wasting precision the decoder produced.
               Web/default tiers (8-bit YUV 4:2:0) should keep the uint8
               default since their final destination is 8-bit anyway.

    Returns:
        See `dtype`. uint8 returns (T, H, W, 3) in [0, 255]; a float dtype
        returns the raw (B, C, T, H, W) decoder output in [-1, 1].
    """
    # Add batch dim if needed
    if latent.ndim == 4:
        latent = latent[None]

    T = latent.shape[2]  # Temporal dim in latent space

    if T <= temporal_chunk_size:
        # Short enough to decode in one pass
        video = decoder(latent, timestep=timestep)
    else:
        # Tiled temporal decoding: split into overlapping chunks and blend.
        # This works around a 3D convolution bug in MLX where early frames
        # of long temporal sequences produce noise artifacts.

        def latent_t_to_pixel_t(lt):
            """Convert latent temporal dim to pixel frames through 3 upsample stages."""
            pt = lt
            for _ in range(3):
                pt = pt * 2 - 1
            return pt

        # Compute expected total pixel frames
        total_pixel_frames = latent_t_to_pixel_t(T)

        stride = temporal_chunk_size - temporal_overlap

        # Decode chunks
        decoded_chunks = []  # (start_latent, end_latent, decoded_video)
        t = 0
        while t < T:
            end = min(t + temporal_chunk_size, T)
            # Ensure last chunk is at least overlap+1 frames
            if end - t < temporal_overlap + 1 and t > 0:
                t = max(0, end - temporal_chunk_size)
                end = min(t + temporal_chunk_size, T)

            chunk_latent = latent[:, :, t:end, :, :]
            chunk_video = decoder(chunk_latent, timestep=timestep)
            mx.eval(chunk_video)
            decoded_chunks.append((t, end, chunk_video))

            if end >= T:
                break
            t += stride

        if len(decoded_chunks) == 1:
            # Only one chunk - trim to expected length
            video = decoded_chunks[0][2][:, :, :total_pixel_frames, :, :]
        else:
            # Multiple chunks - stitch with overlap blending.
            # Compute overlap in pixel space by decoding the overlap latent count
            # as a standalone chunk to get its exact pixel length.
            overlap_pixel_ref = latent_t_to_pixel_t(temporal_overlap)

            # Start with the first chunk
            video = decoded_chunks[0][2]

            for i in range(1, len(decoded_chunks)):
                curr_start_l, curr_end_l, curr_video = decoded_chunks[i]
                curr_pixel_len = curr_video.shape[2]

                # The overlap in pixel space
                overlap_pixels = min(overlap_pixel_ref, curr_pixel_len, video.shape[2])

                if overlap_pixels <= 1:
                    # No meaningful overlap - just concatenate
                    video = mx.concatenate([video, curr_video], axis=2)
                    continue

                # Get overlap regions from both sides
                prev_overlap = video[:, :, -overlap_pixels:, :, :]
                curr_overlap = curr_video[:, :, :overlap_pixels, :, :]
                curr_tail = curr_video[:, :, overlap_pixels:, :, :]

                # Linear crossfade
                ramp = mx.linspace(0.0, 1.0, overlap_pixels).reshape(1, 1, overlap_pixels, 1, 1)
                blended = prev_overlap * (1.0 - ramp) + curr_overlap * ramp

                # Stitch: keep everything before overlap, add blend, add tail
                video = mx.concatenate([
                    video[:, :, :-overlap_pixels, :, :],
                    blended,
                    curr_tail,
                ], axis=2)

            # Trim to exact expected length
            video = video[:, :, :total_pixel_frames, :, :]

    # Honor the requested output dtype. Default (None / mx.uint8) preserves
    # the legacy ffmpeg-friendly (T, H, W, 3) uint8 format. Float dtypes get
    # the raw (B, C, T, H, W) decoder output untouched so the consumer can
    # quantize at its own destination format.
    if dtype is None or dtype == mx.uint8:
        video = mx.clip((video + 1) / 2, 0, 1) * 255
        video = video.astype(mx.uint8)
        video = video[0]                       # (C, T, H, W)
        video = video.transpose(1, 2, 3, 0)    # (T, H, W, C)
        return video
    if dtype in (mx.float16, mx.float32, mx.bfloat16):
        return video.astype(dtype) if dtype != video.dtype else video
    raise ValueError(
        f"decode_latent dtype must be None / mx.uint8 / mx.float16 / "
        f"mx.float32 / mx.bfloat16; got {dtype!r}"
    )
