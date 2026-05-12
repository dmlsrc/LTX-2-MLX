"""Tiled VAE decoding for memory-efficient high-resolution video generation.

This implementation keeps spatial and temporal tiling enabled by default. The
video VAE is causal in time, so temporal tiles require extra left context and
careful output mapping. Tile accumulation is done in FP32, and overlapping
regions are updated with slice assignment rather than scatter-style
`.at(...).add(...)`, which can corrupt overlapping tiled output on some MLX
paths.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import mlx.core as mx


DEFAULT_TEMPORAL_SCALE = 8
DEFAULT_SPATIAL_SCALE = 32
_INT32_MAX_ELEMENTS = 2**31 - 1
_NATIVE_CONV3D_FINAL_CHANNELS = 512


def detect_system_memory_gb() -> float | None:
    """Best-effort physical RAM detection without adding a runtime dependency."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return (pages * page_size) / (1000**3)


def default_vae_decode_budget_gb(total_memory_gb: float | None = None) -> float:
    """Choose a moderate VAE decode budget from physical RAM.

    The cap intentionally favors the measured 1024x576 `128/8` native Conv3d
    point over `256/8`: `256/8` was only about a second faster while using much
    more memory.
    """
    if total_memory_gb is None:
        total_memory_gb = detect_system_memory_gb()
    if total_memory_gb is None:
        return 12.0
    return max(6.0, min(16.0, total_memory_gb * 0.5))


def _native_conv3d_safe_frames(height: int, width: int) -> int:
    frame_elements = (
        max(1, (height + 7) // 8)
        * max(1, (width + 7) // 8)
        * _NATIVE_CONV3D_FINAL_CHANNELS
    )
    return max(1, _INT32_MAX_ELEMENTS // frame_elements)


def _estimate_native_conv3d_peak_gb(height: int, width: int, tile_frames: int) -> float:
    """Estimate native Conv3d VAE decode peak from the 1024x576 tile sweep."""
    area_scale = (height * width) / (1024 * 576)
    effective_frames = max(1, tile_frames) * area_scale
    return 4.3 + 0.084 * effective_frames


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> mx.array:
    """Create a 1D blend mask with optional left and right ramps."""
    if length <= 0:
        raise ValueError(f"Mask length must be positive, got {length}")

    ramp_left = max(0, min(int(ramp_left), length))
    ramp_right = max(0, min(int(ramp_right), length))

    values = [1.0] * length

    if ramp_left > 0:
        count = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade = [i / (count - 1) for i in range(count)]
        fade = fade[:-1]
        if not left_starts_from_0:
            fade = fade[1:]
        for i, v in enumerate(fade[:ramp_left]):
            values[i] *= v

    if ramp_right > 0:
        fade = [(ramp_right + 1 - i) / (ramp_right + 1) for i in range(1, ramp_right + 1)]
        start = length - ramp_right
        for i, v in enumerate(fade):
            values[start + i] *= v

    return mx.clip(mx.array(values, dtype=mx.float32), 0.0, 1.0)


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Spatial tile configuration in decoded-pixel coordinates."""

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(
                f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}"
            )
        if self.tile_size_in_pixels % DEFAULT_SPATIAL_SCALE != 0:
            raise ValueError(
                f"tile_size_in_pixels must be divisible by {DEFAULT_SPATIAL_SCALE}, "
                f"got {self.tile_size_in_pixels}"
            )
        if self.tile_overlap_in_pixels % DEFAULT_SPATIAL_SCALE != 0:
            raise ValueError(
                f"tile_overlap_in_pixels must be divisible by {DEFAULT_SPATIAL_SCALE}, "
                f"got {self.tile_overlap_in_pixels}"
            )
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                "Spatial overlap must be smaller than tile size, got "
                f"{self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Temporal tile configuration in decoded-frame coordinates."""

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(
                f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}"
            )
        if self.tile_size_in_frames % DEFAULT_TEMPORAL_SCALE != 0:
            raise ValueError(
                f"tile_size_in_frames must be divisible by {DEFAULT_TEMPORAL_SCALE}, "
                f"got {self.tile_size_in_frames}"
            )
        if self.tile_overlap_in_frames % DEFAULT_TEMPORAL_SCALE != 0:
            raise ValueError(
                f"tile_overlap_in_frames must be divisible by {DEFAULT_TEMPORAL_SCALE}, "
                f"got {self.tile_overlap_in_frames}"
            )
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                "Temporal overlap must be smaller than tile size, got "
                f"{self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for tiled VAE decoding."""

    spatial_config: Optional[SpatialTilingConfig] = None
    temporal_config: Optional[TemporalTilingConfig] = None

    @classmethod
    def default(cls) -> "TilingConfig":
        """Default tiled decode: spatial 512/64 and temporal 64/24."""
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=512,
                tile_overlap_in_pixels=64,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=64,
                tile_overlap_in_frames=24,
            ),
        )

    @classmethod
    def auto(
        cls,
        height: int,
        width: int,
        num_frames: int,
        spatial_threshold: int = 512,
        temporal_threshold: int = 65,
        decoder_backend: str = "simple",
        total_memory_gb: float | None = None,
        memory_budget_gb: float | None = None,
    ) -> Optional["TilingConfig"]:
        """Auto-select VAE tiling for the requested decode shape."""
        if decoder_backend == "native-conv3d":
            return cls.auto_native_conv3d(
                height,
                width,
                num_frames,
                total_memory_gb=total_memory_gb,
                memory_budget_gb=memory_budget_gb,
            )

        # Match mlx-video's legacy auto tiling policy for the simple decoder.
        needs_spatial = height > spatial_threshold or width > spatial_threshold
        needs_temporal = num_frames > temporal_threshold

        if not needs_spatial and not needs_temporal:
            return None

        return cls(
            spatial_config=(
                SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64)
                if needs_spatial
                else None
            ),
            temporal_config=(
                TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24)
                if needs_temporal
                else None
            ),
        )

    @classmethod
    def auto_native_conv3d(
        cls,
        height: int,
        width: int,
        num_frames: int,
        *,
        total_memory_gb: float | None = None,
        memory_budget_gb: float | None = None,
    ) -> Optional["TilingConfig"]:
        """Pick a native Conv3d tile plan from RAM and int32-addressing limits."""
        budget_gb = (
            memory_budget_gb
            if memory_budget_gb is not None
            else default_vae_decode_budget_gb(total_memory_gb)
        )
        safe_frames = _native_conv3d_safe_frames(height, width)

        # Fastest path first. It is allowed only when it fits the int32 Conv3d
        # output-addressing boundary and the estimated memory budget.
        full_peak_gb = _estimate_native_conv3d_peak_gb(height, width, num_frames)
        if num_frames <= safe_frames and full_peak_gb <= budget_gb:
            return None

        temporal_candidates = (256, 128, 64, 40, 32)
        for tile_frames in temporal_candidates:
            if tile_frames >= num_frames:
                continue
            if tile_frames > safe_frames:
                continue
            peak_gb = _estimate_native_conv3d_peak_gb(height, width, tile_frames)
            if peak_gb <= budget_gb:
                return cls.temporal_only(tile_size=tile_frames, overlap=8)

        # If the frame is too large for temporal-only tiling under the budget,
        # add spatial tiling and try again. Keep this as a fallback because it
        # multiplies decode jobs and was much slower in the 1024x576 probes.
        spatial_tile = 512
        spatial_overlap = 64
        effective_h = min(height, spatial_tile)
        effective_w = min(width, spatial_tile)
        safe_spatial_frames = _native_conv3d_safe_frames(effective_h, effective_w)
        for tile_frames in temporal_candidates:
            if tile_frames >= num_frames:
                continue
            if tile_frames > safe_spatial_frames:
                continue
            peak_gb = _estimate_native_conv3d_peak_gb(
                effective_h,
                effective_w,
                tile_frames,
            )
            if peak_gb <= budget_gb:
                return cls(
                    spatial_config=SpatialTilingConfig(
                        tile_size_in_pixels=spatial_tile,
                        tile_overlap_in_pixels=spatial_overlap,
                    ),
                    temporal_config=TemporalTilingConfig(
                        tile_size_in_frames=tile_frames,
                        tile_overlap_in_frames=8,
                    ),
                )

        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=spatial_tile,
                tile_overlap_in_pixels=spatial_overlap,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=32,
                tile_overlap_in_frames=8,
            ),
        )

    @classmethod
    def spatial_only(cls, tile_size: int = 512, overlap: int = 64) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=tile_size,
                tile_overlap_in_pixels=overlap,
            ),
            temporal_config=None,
        )

    @classmethod
    def temporal_only(cls, tile_size: int = 64, overlap: int = 24) -> "TilingConfig":
        return cls(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=tile_size,
                tile_overlap_in_frames=overlap,
            ),
        )

    @classmethod
    def aggressive(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=256,
                tile_overlap_in_pixels=64,
            ),
            temporal_config=None,
        )

    @classmethod
    def conservative(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=768,
                tile_overlap_in_pixels=64,
            ),
            temporal_config=None,
        )

    @classmethod
    def test_small_both(cls) -> "TilingConfig":
        """Small-video test config: forces both spatial and temporal tiling.

        Useful for debugging with 256x256, 4-second generations. Do not use as
        the production default.
        """
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=128,
                tile_overlap_in_pixels=32,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=32,
                tile_overlap_in_frames=8,
            ),
        )


@dataclass(frozen=True)
class AxisTiles:
    starts: List[int]
    ends: List[int]
    left_ramps: List[int]
    right_ramps: List[int]


def _split_axis(length: int, tile_size: int, overlap: int) -> AxisTiles:
    """Split an axis into overlapping tiles in latent coordinates."""
    if length <= 0:
        raise ValueError(f"Axis length must be positive, got {length}")
    if tile_size <= 0:
        raise ValueError(f"Tile size must be positive, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"Invalid overlap {overlap} for tile size {tile_size}")

    if length <= tile_size + overlap:
        return AxisTiles([0], [length], [0], [0])

    stride = tile_size - overlap
    starts: List[int] = []
    ends: List[int] = []

    pos = 0
    while True:
        start = pos
        end = min(start + tile_size, length)

        if end == length:
            start = max(0, length - tile_size)

        if starts and start <= starts[-1]:
            break

        starts.append(start)
        ends.append(end)

        if end >= length:
            break

        pos += stride

    left = [0 if i == 0 else overlap for i in range(len(starts))]
    right = [0 if i == len(starts) - 1 else overlap for i in range(len(starts))]
    return AxisTiles(starts, ends, left, right)


def _split_temporal_axis(length: int, tile_size: int, overlap: int) -> AxisTiles:
    """Split temporal latent axis, adding left context after tile 0.

    The VAE is causal in time. Later temporal tiles need one latent frame of left
    context so their earliest decoded frames are not decoded as if the tile were
    the beginning of the whole video.
    """
    base = _split_axis(length, tile_size, overlap)

    starts = list(base.starts)
    left = list(base.left_ramps)

    for i in range(1, len(starts)):
        if starts[i] > 0:
            starts[i] -= 1
            left[i] += 1

    return AxisTiles(starts, list(base.ends), left, list(base.right_ramps))


def _temporal_output_slice(
    start_latent: int,
    end_latent: int,
    left_ramp_latent: int,
    right_ramp_latent: int,
    scale: int,
) -> Tuple[slice, mx.array]:
    """Map a temporal latent interval to decoded-frame interval and mask."""
    start = 0 if start_latent == 0 else start_latent * scale
    stop = 1 if end_latent <= 1 else 1 + (end_latent - 1) * scale

    left = 0
    if left_ramp_latent > 0:
        left = 1 + (left_ramp_latent - 1) * scale

    right = right_ramp_latent * scale

    mask = compute_trapezoidal_mask_1d(
        stop - start,
        left,
        right,
        left_starts_from_0=True,
    )
    return slice(start, stop), mask


def _spatial_output_slice(
    start_latent: int,
    end_latent: int,
    left_ramp_latent: int,
    right_ramp_latent: int,
    scale: int,
) -> Tuple[slice, mx.array]:
    """Map a spatial latent interval to decoded-pixel interval and mask."""
    start = start_latent * scale
    stop = end_latent * scale
    mask = compute_trapezoidal_mask_1d(
        stop - start,
        left_ramp_latent * scale,
        right_ramp_latent * scale,
        left_starts_from_0=False,
    )
    return slice(start, stop), mask


def _assign_add_5d(
    arr: mx.array,
    update: mx.array,
    t_slice: slice,
    h_slice: slice,
    w_slice: slice,
) -> mx.array:
    """Add update into arr using direct slice assignment.

    This deliberately avoids scatter-style ``.at(...).add(...)`` because
    overlapping multidimensional updates produced corrupted tiled VAE output in
    practice.
    """
    current = arr[:, :, t_slice, h_slice, w_slice]
    arr[:, :, t_slice, h_slice, w_slice] = current + update
    return arr


def _merge_temporal_pending(
    pending: Optional[mx.array],
    pending_weights: Optional[mx.array],
    pending_start: int,
    chunk: mx.array,
    chunk_weights: mx.array,
    chunk_start: int,
) -> Tuple[mx.array, mx.array, int]:
    """Merge a temporal chunk into the rolling output accumulator."""
    if pending is None or pending_weights is None:
        return chunk, chunk_weights, chunk_start

    pending_end = pending_start + pending.shape[2]
    chunk_end = chunk_start + chunk.shape[2]
    merged_start = min(pending_start, chunk_start)
    merged_end = max(pending_end, chunk_end)
    merged_t = merged_end - merged_start

    b, c, _t, h, w = pending.shape
    merged = mx.zeros((b, c, merged_t, h, w), dtype=mx.float32)
    merged_weights = mx.zeros((1, 1, merged_t, h, w), dtype=mx.float32)

    pending_slice = slice(pending_start - merged_start, pending_end - merged_start)
    chunk_slice = slice(chunk_start - merged_start, chunk_end - merged_start)

    merged = _assign_add_5d(merged, pending, pending_slice, slice(0, h), slice(0, w))
    merged_weights = _assign_add_5d(
        merged_weights,
        pending_weights,
        pending_slice,
        slice(0, h),
        slice(0, w),
    )
    merged = _assign_add_5d(merged, chunk, chunk_slice, slice(0, h), slice(0, w))
    merged_weights = _assign_add_5d(
        merged_weights,
        chunk_weights,
        chunk_slice,
        slice(0, h),
        slice(0, w),
    )
    mx.eval(merged, merged_weights)
    return merged, merged_weights, merged_start


def decode_tiled(
    latent: mx.array,
    decoder_fn,
    tiling_config: TilingConfig,
    timestep: Optional[float] = 0.05,
    show_progress: bool = True,
    key: Optional[mx.array] = None,
) -> Iterator[mx.array]:
    """Decode a latent tensor by tiles and blend overlaps."""
    del key  # Reserved for API compatibility.

    b, _c, latent_t, latent_h, latent_w = latent.shape
    scale_t, scale_h, scale_w = (
        DEFAULT_TEMPORAL_SCALE,
        DEFAULT_SPATIAL_SCALE,
        DEFAULT_SPATIAL_SCALE,
    )

    out_t = 1 + (latent_t - 1) * scale_t
    out_h = latent_h * scale_h
    out_w = latent_w * scale_w

    if tiling_config.temporal_config is not None:
        tc = tiling_config.temporal_config
        temporal_tile = tc.tile_size_in_frames // scale_t
        temporal_overlap = tc.tile_overlap_in_frames // scale_t
        t_tiles = _split_temporal_axis(latent_t, temporal_tile, temporal_overlap)
    else:
        t_tiles = AxisTiles([0], [latent_t], [0], [0])

    if tiling_config.spatial_config is not None:
        sc = tiling_config.spatial_config
        spatial_tile = sc.tile_size_in_pixels // scale_h
        spatial_overlap = sc.tile_overlap_in_pixels // scale_h
        h_tiles = _split_axis(latent_h, spatial_tile, spatial_overlap)
        w_tiles = _split_axis(latent_w, spatial_tile, spatial_overlap)
    else:
        h_tiles = AxisTiles([0], [latent_h], [0], [0])
        w_tiles = AxisTiles([0], [latent_w], [0], [0])

    temporal_count = len(t_tiles.starts)
    spatial_count = len(h_tiles.starts) * len(w_tiles.starts)
    total_jobs = temporal_count * spatial_count

    if show_progress:
        print(
            "  Tiled VAE decode: "
            f"temporal={temporal_count}, "
            f"spatial={len(h_tiles.starts)}x{len(w_tiles.starts)}, "
            f"total={total_jobs}"
        )
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
    else:
        tqdm = None

    pending: Optional[mx.array] = None
    pending_weights: Optional[mx.array] = None
    pending_start = 0

    progress = tqdm(total=total_jobs, desc="Tiled decode", ncols=80) if tqdm else None

    try:
        for ti in range(temporal_count):
            t0, t1 = t_tiles.starts[ti], t_tiles.ends[ti]

            out_t_slice, mask_t = _temporal_output_slice(
                t0, t1, t_tiles.left_ramps[ti], t_tiles.right_ramps[ti], scale_t
            )
            chunk_t = out_t_slice.stop - out_t_slice.start
            chunk = mx.zeros((b, 3, chunk_t, out_h, out_w), dtype=mx.float32)
            chunk_weights = mx.zeros((1, 1, chunk_t, out_h, out_w), dtype=mx.float32)
            mx.eval(chunk, chunk_weights)

            for hi in range(len(h_tiles.starts)):
                h0, h1 = h_tiles.starts[hi], h_tiles.ends[hi]
                out_h_slice, mask_h = _spatial_output_slice(
                    h0, h1, h_tiles.left_ramps[hi], h_tiles.right_ramps[hi], scale_h
                )

                for wi in range(len(w_tiles.starts)):
                    w0, w1 = w_tiles.starts[wi], w_tiles.ends[wi]
                    out_w_slice, mask_w = _spatial_output_slice(
                        w0,
                        w1,
                        w_tiles.left_ramps[wi],
                        w_tiles.right_ramps[wi],
                        scale_w,
                    )

                    tile_latent = latent[:, :, t0:t1, h0:h1, w0:w1]
                    decoded = decoder_fn(
                        tile_latent,
                        timestep=timestep,
                        show_progress=False,
                    )
                    mx.eval(decoded)

                    expected_h = out_h_slice.stop - out_h_slice.start
                    expected_w = out_w_slice.stop - out_w_slice.start

                    actual_t = min(decoded.shape[2], chunk_t)
                    actual_h = min(decoded.shape[3], expected_h)
                    actual_w = min(decoded.shape[4], expected_w)

                    decoded = decoded[:, :, :actual_t, :actual_h, :actual_w].astype(mx.float32)

                    mask = (
                        mask_t[:actual_t].reshape(1, 1, actual_t, 1, 1)
                        * mask_h[:actual_h].reshape(1, 1, 1, actual_h, 1)
                        * mask_w[:actual_w].reshape(1, 1, 1, 1, actual_w)
                    ).astype(mx.float32)

                    actual_t_slice = slice(0, actual_t)
                    actual_h_slice = slice(out_h_slice.start, out_h_slice.start + actual_h)
                    actual_w_slice = slice(out_w_slice.start, out_w_slice.start + actual_w)

                    chunk = _assign_add_5d(
                        chunk,
                        decoded * mask,
                        actual_t_slice,
                        actual_h_slice,
                        actual_w_slice,
                    )
                    chunk_weights = _assign_add_5d(
                        chunk_weights,
                        mask,
                        actual_t_slice,
                        actual_h_slice,
                        actual_w_slice,
                    )

                    mx.eval(chunk, chunk_weights)
                    del decoded, mask, tile_latent
                    mx.clear_cache()
                    if progress is not None:
                        progress.update(1)

            if pending is not None and pending_weights is not None:
                pending_end = pending_start + pending.shape[2]
                ready_t = max(0, min(out_t_slice.start, pending_end) - pending_start)
                if ready_t > 0:
                    ready = pending[:, :, :ready_t]
                    ready_weights = pending_weights[:, :, :ready_t]
                    ready = ready / mx.maximum(ready_weights, 1e-8)
                    mx.eval(ready)
                    yield ready
                    del ready, ready_weights
                    pending = pending[:, :, ready_t:]
                    pending_weights = pending_weights[:, :, ready_t:]
                    pending_start += ready_t
                    mx.eval(pending, pending_weights)

            pending, pending_weights, pending_start = _merge_temporal_pending(
                pending,
                pending_weights,
                pending_start,
                chunk,
                chunk_weights,
                out_t_slice.start,
            )
            del chunk, chunk_weights
            gc.collect()
            mx.clear_cache()

        if pending is not None and pending_weights is not None and pending.shape[2] > 0:
            pending = pending / mx.maximum(pending_weights, 1e-8)
            mx.eval(pending)
            yield pending
    finally:
        if progress is not None:
            progress.close()
