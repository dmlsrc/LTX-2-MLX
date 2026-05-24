"""Streaming VAE decode for the pipelines' `decode_video=False` callers.

When a pipeline is called with `decode_video=False`, it returns the
final video latent instead of a decoded tensor.  The caller then runs
the VAE decode through `iter_decoded_chunks` (chunk-granular,
preserving boundaries for progress bookkeeping) or `iter_decoded_frames`
(flat per-frame yields, ready to thread into an iterator-aware encoder
like `LTX_2_MLX.videotoolbox.encode.encode_video_videotoolbox`).

Why this matters
----------------
Pipelines historically concatenated every VAE chunk into one giant
`(B, 3, T, H, W)` float tensor before returning to the caller.  That
makes the peak resident video size a hard ceiling on clip length:

    721 frames @ 576x320 -> ~1.6 GB float32 video tensor
    721 frames @ 1280x768 -> ~8 GB float32 video tensor

Plus a separate uint8 per-frame list materialized in generate.py
before the encode call.

Streaming avoids both peaks: at any moment the resident decoded video
is one tile-chunk worth (e.g. 256 frames), plus whatever buffers the
downstream encoder pools.  Long clips that previously hit decode-time
OOM start working.

Output formats
--------------
- "uint8_rgb"   -> list[(H, W, 3) uint8]  per chunk.  Matches the
                   per-frame list shape generate.py used to build by
                   hand before passing to `encode_video()` (ffmpeg path).
- "fp16_rgba"   -> list[(H, W, 4) float16] per chunk.  Matches VSR's
                   RGBAHalf source format — the encoder pushes these
                   straight into the VSR session's source pool without
                   any further casting.

Both converters force-clear MLX's cache after producing the
Python-owned numpy copies, so the Metal buffers from the chunk decode
don't sit pinned across the downstream iteration.  See
`scripts/vsr_harness.py`'s `chunk_to_rgba_fp16` for the original
rationale and the bench numbers that drove the per-frame-list shape
choice over a single (T,H,W,C) array.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Iterator, List, Optional

import numpy as np

import mlx.core as mx

from ..model.video_vae.decode_utils import decode_latent
from ..model.video_vae.tiling import TilingConfig, decode_tiled


# Per-frame list vs. single (T,H,W,C) ndarray.  Default is the
# per-frame list because the downstream loop can null each entry as
# it's consumed, tapering resident chunk memory as frames are fed to
# the encoder.  Set LTX_STREAM_CHUNK_AS_ARRAY=1 to get a single big
# ndarray per chunk (used in a/b tests; see vsr_harness for the bench
# table).
_CHUNK_AS_ARRAY = os.environ.get("LTX_STREAM_CHUNK_AS_ARRAY", "0") == "1"


def chunk_to_uint8_frames(chunk: Any) -> List[np.ndarray]:
    """(B, C, T, H, W) bf16/fp32 in [-1, 1] -> list[(H, W, 3) uint8].

    Rescale + clip + cast happens in MLX so we never materialize a
    chunk-sized float numpy intermediate.  Each frame becomes an
    independently-allocated Python-owned numpy buffer that the caller
    can free as it iterates.
    """
    rescaled = mx.clip((chunk + 1.0) * 127.5, 0, 255).astype(mx.uint8)
    transposed = mx.transpose(rescaled, (0, 2, 3, 4, 1))  # (B, T, H, W, 3)
    mx.eval(transposed)
    if _CHUNK_AS_ARRAY:
        arr = np.array(transposed, copy=True)
        result = [arr[0, t] for t in range(arr.shape[1])]
    else:
        result = [
            np.array(transposed[0, t], copy=True)
            for t in range(transposed.shape[1])
        ]
    del rescaled, transposed
    try:
        mx.clear_cache()
    except Exception:
        pass
    return result


def chunk_to_rgba_fp16_frames(chunk: Any) -> List[np.ndarray]:
    """(B, C, T, H, W) bf16/fp32 in [-1, 1] -> list[(H, W, 4) float16].

    Direct path for VSR's RGBAHalf source format.  Skips the uint8
    quantization that `chunk_to_uint8_frames` would impose, so the
    VAE's bf16 precision survives into VSR / the encoder.
    """
    B, C, T, H, W = chunk.shape
    rescaled = mx.clip((chunk + 1.0) * 0.5, 0.0, 1.0).astype(mx.float16)
    alpha = mx.ones((B, 1, T, H, W), dtype=mx.float16)
    rgba = mx.concatenate([rescaled, alpha], axis=1)
    transposed = mx.transpose(rgba, (0, 2, 3, 4, 1))  # (B, T, H, W, 4)
    mx.eval(transposed)
    if _CHUNK_AS_ARRAY:
        arr = np.array(transposed, copy=True)
        result = [arr[0, t] for t in range(arr.shape[1])]
    else:
        result = [
            np.array(transposed[0, t], copy=True)
            for t in range(transposed.shape[1])
        ]
    del rescaled, alpha, rgba, transposed
    try:
        mx.clear_cache()
    except Exception:
        pass
    return result


def _converter_for(output_format: str):
    if output_format == "uint8_rgb":
        return chunk_to_uint8_frames
    if output_format == "fp16_rgba":
        return chunk_to_rgba_fp16_frames
    raise ValueError(
        f"Unknown output_format {output_format!r}; expected "
        f"'uint8_rgb' or 'fp16_rgba'."
    )


def iter_decoded_chunks(
    latent: Any,
    decoder: Any,
    *,
    tiling: Optional[TilingConfig],
    output_format: str = "fp16_rgba",
    compute_dtype: Any = mx.bfloat16,
) -> Iterator[List[np.ndarray]]:
    """Yield decoded chunks as per-frame numpy lists.

    Each yield delivers one VAE chunk's worth of frames; the caller
    iterates over the inner list to feed frames into a downstream
    consumer.  Memory bookkeeping (cache clear, gc.collect) runs
    between chunks so the MLX heap stays at one-chunk's worth even on
    long clips.

    `tiling=None` selects the single-shot decode path (used when
    `TilingConfig.auto()` decides no tiling is needed).
    """
    convert = _converter_for(output_format)

    if tiling is None:
        # Single-shot decode.  decode_latent does its own internal
        # temporal chunking + overlap blending; the result is a full
        # (B,C,T,H,W) tensor in `compute_dtype`.  We then convert in
        # one shot and free.
        video = decode_latent(latent, decoder, dtype=compute_dtype)
        try:
            mx.clear_cache()
        except Exception:
            pass
        out = convert(video)
        del video
        try:
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()
        yield out
        return

    for chunk in decode_tiled(latent, decoder, tiling, show_progress=False):
        out = convert(chunk)
        del chunk
        try:
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()
        yield out
        del out


def iter_decoded_frames(
    latent: Any,
    decoder: Any,
    *,
    tiling: Optional[TilingConfig],
    output_format: str = "fp16_rgba",
    compute_dtype: Any = mx.bfloat16,
) -> Iterator[np.ndarray]:
    """Flat per-frame iterator over the chunked decode.

    Wraps `iter_decoded_chunks` and chains the inner lists into a
    single stream of per-frame ndarrays — the shape that
    `encode_video_videotoolbox`'s iterator input expects.  As each
    frame is consumed the caller's reference drops, and at chunk
    boundaries the cache-clear / gc.collect pair runs to keep
    resident memory bounded.

    Use this when feeding a streaming encoder; use
    `iter_decoded_chunks` when the caller needs chunk-level
    granularity (progress bookkeeping, A/B harnesses).
    """
    for chunk_frames in iter_decoded_chunks(
        latent, decoder,
        tiling=tiling,
        output_format=output_format,
        compute_dtype=compute_dtype,
    ):
        # `pop(0)` lets the inner list shrink as frames are yielded,
        # so the per-chunk peak resident frame count drops linearly as
        # the consumer drains the chunk — even though Python's list
        # backing array won't shrink physically until the list is
        # destroyed at end-of-iteration.
        while chunk_frames:
            yield chunk_frames.pop(0)


def latent_dims(latent: Any) -> tuple[int, int, int]:
    """(num_frames, height, width) implied by a video latent's shape.

    Mirrors `vsr_harness.latent_dims` so callers can size progress
    bars and validate against the encoder's spatial expectations
    before any GPU work runs.
    """
    _, _, latent_frames, latent_height, latent_width = latent.shape
    n_frames = 1 + (latent_frames - 1) * 8
    height = latent_height * 32
    width = latent_width * 32
    return n_frames, height, width


def plan_vae_tiling(
    latent: Any, tiling: Optional[TilingConfig],
) -> tuple[int, str]:
    """Describe a (latent, tiling) pair: returns (n_chunks, description).

    Pure CPU arithmetic — no GPU work — so callers can use the result
    to size a progress bar (or print a status line) before any chunk
    decode runs.

    `tiling=None` is the single-shot path; n_chunks=1, description
    notes the frame count.  Spatial-only / temporal-only / both
    spatial+temporal tilings are all handled.

    Mirrors `vsr_harness.plan_vae_tiling` so generate.py's streaming
    output can match the vsr_harness UX line-for-line.
    """
    n_frames, _height, _width = latent_dims(latent)
    if tiling is None:
        return 1, f"off (single-shot decode of {n_frames} frames)"

    sp = tiling.spatial_config
    tp = tiling.temporal_config
    spatial_desc = (
        f"spatial tile={sp.tile_size_in_pixels} overlap={sp.tile_overlap_in_pixels}"
        if sp else "no spatial"
    )
    temporal_desc = (
        f"temporal tile={tp.tile_size_in_frames} overlap={tp.tile_overlap_in_frames}"
        if tp else "no temporal"
    )
    if tp is not None:
        tile = tp.tile_size_in_frames
        overlap = tp.tile_overlap_in_frames
        step = max(1, tile - overlap)
        n_chunks = max(1, -(-(n_frames - overlap) // step))
    else:
        n_chunks = 1
    return n_chunks, f"{spatial_desc}, {temporal_desc}"
