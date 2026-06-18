#!/usr/bin/env python3
"""Generate video from text prompt using LTX-2 MLX."""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX import sidecars
from LTX_2_MLX.components import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    VideoLatentPatchifier,
    get_sigma_schedule,
)
from LTX_2_MLX.components.guiders import LegacyStatefulAPGGuider, LtxAPGGuider, STGGuider
from LTX_2_MLX.components.patchifiers import get_pixel_coords
from LTX_2_MLX.components.perturbations import create_batched_stg_config
from LTX_2_MLX.core_utils import to_velocity
from LTX_2_MLX.ffmpeg_encoder import TIERS, encode_video_ffmpeg
from LTX_2_MLX.loader import (
    TRANSFORMER_CACHE_QUANTIZE_MODES,
    TRANSFORMER_CACHE_QUANTIZE_OFF,
    LoRAConfig,
    checkpoint_has_fp8_tensors,
    ensure_weight_family_caches,
    get_transformer_cache_restore_state,
    load_av_transformer_weights,
    load_transformer_weights,
    load_transformer_weights_cached,
    load_transformer_weights_cached_streaming,
    lora_configs_have_stage_strengths,
)
from LTX_2_MLX.loader.lora_loader import fuse_loras_into_model
from LTX_2_MLX.model.audio_vae import (
    AudioDecoder,
    Vocoder,
    VocoderWithBWE,
    load_audio_decoder_weights,
    load_vocoder_weights,
    load_vocoder_with_bwe_weights,
)
from LTX_2_MLX.model.audio_vae.vocoder import MelSTFT
from LTX_2_MLX.model.transformer import (
    LTXAVModel,
    LTXModel,
    LTXModelType,
    LTXRopeType,
    Modality,
    X0Model,
)
from LTX_2_MLX.model.video_vae.decode_utils import decode_latent
from LTX_2_MLX.model.video_vae.native_decoder import (
    NativeConv3dVideoDecoder,
    load_native_vae_decoder_weights,
)
from LTX_2_MLX.model.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from LTX_2_MLX.progress import PhaseBar, StackedPhaseBars
from LTX_2_MLX.types import NATIVE_FPS, SpatioTemporalScaleFactors, VideoLatentShape

# Tiers that map cleanly onto AVAssetWriter's HEVC outputs.  Auto-mode
# routes these through the VideoToolbox backend; everything else stays
# on ffmpeg.  `default` is HEVC Main10 4:2:0, which is exactly what
# AVWriter writes out of the box.  `hq` is HEVC 4:4:4 which AVWriter
# doesn't do; `web`/`export`/`reference` are H.264 / ProRes, also out
# of scope for the VT helper today.
VT_AUTO_TIERS = {"default"}


def _vsr_active(vsr_spatial_mode: str, vsr_target_fps: float | None) -> bool:
    """True if any VSR / VTFRC flag is engaged (forces VT backend)."""
    return vsr_spatial_mode not in (None, "off") or vsr_target_fps is not None


def resolve_output_backend(
    requested: str,
    tier: str,
    vsr_spatial_mode: str = "off",
    vsr_target_fps: float | None = None,
) -> str:
    """Resolve --output-backend {auto,ffmpeg,videotoolbox} -> {ffmpeg,videotoolbox}.

    Auto rules:
      - VSR or target_fps engaged                -> videotoolbox (required)
      - tier in VT_AUTO_TIERS (HEVC Main10)      -> videotoolbox
      - everything else                          -> ffmpeg

    Explicit `videotoolbox` is validated against the tier - non-HEVC
    tiers raise here rather than failing deep inside AVAssetWriter.
    """
    vsr_on = _vsr_active(vsr_spatial_mode, vsr_target_fps)
    if requested == "videotoolbox":
        if tier not in VT_AUTO_TIERS:
            raise SystemExit(
                f"--output-backend videotoolbox supports --encode-tier "
                f"{sorted(VT_AUTO_TIERS)}; got tier={tier!r}. Use the "
                f"ffmpeg backend or change tier."
            )
        return "videotoolbox"
    if requested == "ffmpeg":
        if vsr_on:
            raise SystemExit(
                "--vsr-spatial-mode / --vsr-target-fps require the "
                "videotoolbox backend; --output-backend ffmpeg cannot "
                "run VSR / VTFRC."
            )
        return "ffmpeg"
    # auto
    if vsr_on:
        return "videotoolbox"
    return "videotoolbox" if tier in VT_AUTO_TIERS else "ffmpeg"


def encode_video_dispatch(
    frames,
    output_path,
    *,
    tier: str,
    fps: float,
    audio_waveform=None,
    audio_sample_rate=None,
    save_audio_sidecar: bool = False,
    audio_onset_trim_mode: str = "auto",
    audio_onset_trim_ms: float | None = None,
    output_backend: str = "auto",
    vsr_spatial_mode: str = "off",
    vsr_target_fps: float | None = None,
    vsr_temporal_mode: str = "normal",
    vsr_save_original: bool = False,
    vsr_encode_quality: float = 0.65,
    vsr_audio_codec: str = "alac",
    n_source_frames: int | None = None,
    progress_stack=None,
):
    """Route an encode call to ffmpeg or AVAssetWriter.

    The signature mirrors the ffmpeg `encode_video_ffmpeg(...)` plus the VT-only
    knobs; resolve_output_backend() picks the backend. Returns the
    encoded output path (extension may be normalized).

    `n_source_frames` is the explicit frame count when `frames` is an
    iterator (the streaming-decode path knows its length from the
    latent shape but the iterator itself doesn't expose it).  Used
    only by the videotoolbox backend's progress bar; ffmpeg ignores
    it since the ffmpeg encoder collects frames eagerly into a list
    before piping anyway.

    `audio_onset_trim_mode` / `audio_onset_trim_ms` are forwarded to
    both backends so the sequence-start spike mitigation is uniform
    regardless of which backend serves the encode.

    `vsr_save_original` is a VT-only knob; ignored when the ffmpeg
    backend serves the encode (no VSR/VTFRC available there anyway).
    """
    backend = resolve_output_backend(
        output_backend, tier,
        vsr_spatial_mode=vsr_spatial_mode,
        vsr_target_fps=vsr_target_fps,
    )
    if backend == "videotoolbox":
        from LTX_2_MLX.videotoolbox import encode_video_videotoolbox
        return encode_video_videotoolbox(
            frames, output_path,
            fps=fps,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            save_audio_sidecar=save_audio_sidecar,
            audio_onset_trim_mode=audio_onset_trim_mode,
            audio_onset_trim_ms=audio_onset_trim_ms,
            vsr_spatial_mode=None if vsr_spatial_mode in (None, "off") else vsr_spatial_mode,
            target_fps=vsr_target_fps,
            vsr_temporal_mode=vsr_temporal_mode,
            vsr_save_original=vsr_save_original,
            encode_quality=vsr_encode_quality,
            audio_codec=vsr_audio_codec,
            n_source_frames=n_source_frames,
            progress_stack=progress_stack,
        )
    return encode_video_ffmpeg(
        frames, output_path,
        tier=tier, fps=fps,
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        save_audio_sidecar=save_audio_sidecar,
        audio_onset_trim_mode=audio_onset_trim_mode,
        audio_onset_trim_ms=audio_onset_trim_ms,
    )


SUPPORTED_COMPUTE_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}
FF_QUANTIZE_TARGETS = ("project_in", "project_out")
FF_QUANTIZE_MODES = ("affine", "mxfp4", "mxfp8", "nvfp4")
FF_LAYOUT_SPECS = {
    "project_in": ("pretranspose",),
    "project_out": ("pretranspose",),
}
ATTN_LAYOUT_SPECS = {
    "to_out": ("pretranspose",),
    "to_q": ("pretranspose",),
    "to_k": ("pretranspose",),
    "to_v": ("pretranspose",),
    # to_gate_logits is supported but measured neutral-to-slight-regression
    # (weights too tiny - 4096*32 / 2048*32 - for the implicit transpose to
    # matter).  Opt-in via explicit --video-attn-layout if you want to A/B.
    "to_gate_logits": ("pretranspose",),
}
# Default to ONLY project_out pretranspose for video FF - that's the single
# matmul where pretranspose rescues a kernel-selection cliff (35 % win,
# 5.17 -> 7.95 TFlops/s at (T=8784, K=16384, N=4096) BF16 per
# scripts/bench_ff_microbench.py bf16_layout).  project_in pretranspose was a
# 2.5 % regression in isolated microbench and neutral end-to-end per the older
# PERFORMANCE.md observation - dropped from default but the flag still
# supports it (--video-ff-layout project_in:pretranspose,project_out:pretranspose
# to A/B against this default).  See docs/PERFORMANCE_NOTES.md for the
# microbench data and reasoning.
DEFAULT_VIDEO_FF_LAYOUT_SPECS = (
    ("project_out", "pretranspose"),
)
# Default video-attn pretranspose to OFF.  The four large attention
# projections (to_q/to_k/to_v/to_out) all sit at ~37 ms / call with or
# without pretranspose in the isolated BF16 microbench (within 1 % noise,
# all at ~7.9-8.0 TFlops/s).  Re-enable via
# --video-attn-layout to_out:pretranspose,to_q:pretranspose,...
# for A/B.  Earlier end-to-end PERFORMANCE.md observation called it
# "marginal positive" but microbench evidence suggests that was likely
# measurement noise.  See docs/PERFORMANCE_NOTES.md.
DEFAULT_VIDEO_ATTN_LAYOUT_SPECS: tuple[tuple[str, str], ...] = ()
DEFAULT_TRANSFORMER_LAYOUT_LAYERS = tuple(range(48))
DEFAULT_LTX_REPO_ID = "Lightricks/LTX-2.3"
LEGACY_LTX_REPO_ID = "Lightricks/LTX-2"
DEFAULT_LTX_WEIGHT_FILES = {
    "distilled": "ltx-2.3-22b-distilled-1.1.safetensors",
    "dev": "ltx-2.3-22b-dev.safetensors",
}
DEFAULT_SPATIAL_UPSCALER_FILES = (
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
)
DEFAULT_GEMMA_REPO_ID = "google/gemma-3-12b-it"
FALLBACK_GEMMA_PATH = "weights/gemma-3-12b"


def _hf_hub_cache_candidates() -> list[Path]:
    """Return Hugging Face hub cache dirs in the order users expect."""
    candidates: list[Path] = []

    def add(path: str | Path | None):
        if not path:
            return
        expanded = Path(path).expanduser()
        if expanded not in candidates:
            candidates.append(expanded)

    add(os.environ.get("HF_HUB_CACHE"))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        add(Path(hf_home) / "hub")
    add(Path.home() / ".cache" / "huggingface" / "hub")
    return candidates


def _hf_repo_cache_dir(hub_cache: Path, repo_id: str) -> Path:
    return hub_cache / f"models--{repo_id.replace('/', '--')}"


def _cached_hf_snapshots(repo_id: str) -> list[Path]:
    snapshots: list[Path] = []
    for hub_cache in _hf_hub_cache_candidates():
        snapshot_root = _hf_repo_cache_dir(hub_cache, repo_id) / "snapshots"
        if not snapshot_root.is_dir():
            continue
        snapshots.extend(path for path in snapshot_root.iterdir() if path.is_dir())
    return sorted(
        snapshots,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _find_cached_hf_file(repo_id: str, filename: str) -> str | None:
    for snapshot in _cached_hf_snapshots(repo_id):
        candidate = snapshot / filename
        if candidate.exists():
            return str(candidate)
    return None


def _find_cached_hf_snapshot(repo_id: str, required_file: str = "config.json") -> str | None:
    for snapshot in _cached_hf_snapshots(repo_id):
        if (snapshot / required_file).exists():
            return str(snapshot)
    return None


def resolve_default_ltx_weights(weights_path: str | None, model_variant: str) -> str:
    """Resolve an explicit or HF-cache default LTX checkpoint path."""
    if weights_path:
        return str(Path(weights_path).expanduser())

    filename = DEFAULT_LTX_WEIGHT_FILES.get(model_variant, DEFAULT_LTX_WEIGHT_FILES["distilled"])
    cached = _find_cached_hf_file(DEFAULT_LTX_REPO_ID, filename)
    if cached:
        return cached
    return str(Path("weights") / "ltx-2" / filename)


def resolve_default_spatial_upscaler_weights(
    weights_path: str | None,
    ltx_weights_path: str | None,
) -> str | None:
    """Resolve an explicit or HF-cache default spatial upscaler path."""
    if weights_path:
        return str(Path(weights_path).expanduser())

    if ltx_weights_path:
        checkpoint_dir = Path(ltx_weights_path).expanduser().parent
        for filename in DEFAULT_SPATIAL_UPSCALER_FILES:
            candidate = checkpoint_dir / filename
            if candidate.exists():
                return str(candidate)

    for filename in DEFAULT_SPATIAL_UPSCALER_FILES:
        cached = _find_cached_hf_file(DEFAULT_LTX_REPO_ID, filename)
        if cached:
            return cached

    legacy_cached = _find_cached_hf_file(
        LEGACY_LTX_REPO_ID,
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    )
    if legacy_cached:
        return legacy_cached

    return None


def resolve_default_gemma_path(gemma_path: str | None) -> str:
    """Resolve an explicit or HF-cache default Gemma directory."""
    if gemma_path:
        return str(Path(gemma_path).expanduser())

    cached = _find_cached_hf_snapshot(DEFAULT_GEMMA_REPO_ID, required_file="config.json")
    if cached:
        return cached
    return FALLBACK_GEMMA_PATH


def resolve_optional_weight_path(path: str | None) -> str | None:
    """Expand an optional user-provided weight path."""
    return str(Path(path).expanduser()) if path else None


def resolve_weight_source(override_path: str | None, default_path: str) -> str:
    """Resolve an advanced per-subsystem override against the bundle default."""
    return resolve_optional_weight_path(override_path) or default_path


def maybe_cache_weight_families(
    family_sources: dict[str, str],
    *,
    cache_mode: str,
    cache_root: str | None,
) -> dict[str, str]:
    """Return load paths for named weight families, using split caches when enabled."""
    if cache_mode == "off":
        return dict(family_sources)

    by_source: dict[str, list[str]] = {}
    for family, source in family_sources.items():
        if source:
            by_source.setdefault(source, []).append(family)

    load_paths: dict[str, str] = {}
    for source, families in by_source.items():
        result = ensure_weight_family_caches(
            source,
            families=tuple(families),
            cache_mode=cache_mode,
            cache_root=cache_root,
        )
        for family in families:
            load_paths[family] = str(result.cache_paths[family])
    return load_paths


def describe_weight_sources(
    *,
    bundle: str,
    config: str,
    transformer: str,
    connector: str,
    video_vae: str,
    audio_vae: str,
    vocoder: str,
) -> list[str]:
    """Create concise user-facing weight source lines."""
    sources = {
        "Config": config,
        "Transformer": transformer,
        "Connector": connector,
        "Video VAE": video_vae,
        "Audio VAE": audio_vae,
        "Vocoder": vocoder,
    }
    if all(source == bundle for source in sources.values()):
        return [f"Weights: {bundle}"]

    lines = [f"Weights bundle: {bundle}", "Weight sources:"]
    lines.extend(f"  {label}: {source}" for label, source in sources.items())
    return lines


def parse_compute_dtype(dtype_name: str | mx.Dtype) -> mx.Dtype:
    """Resolve a user-facing dtype name to an MLX dtype."""
    if not isinstance(dtype_name, str):
        return dtype_name
    try:
        return SUPPORTED_COMPUTE_DTYPES[dtype_name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(SUPPORTED_COMPUTE_DTYPES))
        raise ValueError(f"Unsupported compute dtype '{dtype_name}'. Valid values: {valid}") from exc


def parse_non_negative_float(value: str) -> float:
    """Parse a non-negative float for user-facing CLI limits."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a non-negative number, got '{value}'") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative number, got {parsed}")
    return parsed


def parse_non_negative_int(value: str) -> int:
    """Parse a non-negative integer for count-style CLI knobs."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got '{value}'") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {parsed}")
    return parsed


def parse_profile_transformer_steps(value: str | None) -> tuple[int, ...]:
    """Parse a comma-separated 1-based denoise-step list for profiling."""
    if value is None or value.strip() == "":
        return ()

    steps = []
    for raw_step in value.split(","):
        step_text = raw_step.strip()
        if not step_text:
            continue
        try:
            step = int(step_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid denoise step '{step_text}' in --profile-transformer-steps"
            ) from exc
        if step <= 0:
            raise argparse.ArgumentTypeError(
                "--profile-transformer-steps uses 1-based positive step numbers"
            )
        steps.append(step)

    return tuple(sorted(set(steps)))


def parse_profile_transformer_blocks(value: str | None) -> tuple[int, ...]:
    """Parse a comma-separated 0-based transformer-block list for profiling."""
    if value is None or value.strip() == "":
        return ()

    blocks = []
    for raw_block in value.split(","):
        block_text = raw_block.strip()
        if not block_text:
            continue
        try:
            block = int(block_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid transformer block '{block_text}' in --profile-transformer-blocks"
            ) from exc
        if block < 0:
            raise argparse.ArgumentTypeError(
                "--profile-transformer-blocks uses 0-based non-negative block numbers"
            )
        blocks.append(block)

    return tuple(sorted(set(blocks)))


def build_vae_tiling_config(
    mode: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    decoder_backend: str = "legacy",
    force_tiled: bool = False,
    temporal_tile_frames: int | None = None,
    temporal_overlap_frames: int = 24,
    spatial_tile_pixels: int | None = None,
    spatial_overlap_pixels: int = 64,
) -> tuple[TilingConfig | None, bool]:
    """Resolve CLI VAE tiling knobs before handing off to a pipeline."""
    if mode == "off":
        return None, False
    if mode == "auto":
        if force_tiled:
            return TilingConfig.default(), True
        return (
            TilingConfig.auto(
                height,
                width,
                num_frames,
                decoder_backend=decoder_backend,
            ),
            True,
        )
    if mode != "custom":
        raise ValueError(f"Unsupported VAE tiling mode: {mode}")

    if temporal_tile_frames is None:
        temporal_tile_frames = 256

    temporal_config = None
    if temporal_tile_frames > 0:
        temporal_config = TemporalTilingConfig(
            tile_size_in_frames=temporal_tile_frames,
            tile_overlap_in_frames=temporal_overlap_frames,
        )

    spatial_config = None
    if spatial_tile_pixels is not None and spatial_tile_pixels > 0:
        spatial_config = SpatialTilingConfig(
            tile_size_in_pixels=spatial_tile_pixels,
            tile_overlap_in_pixels=spatial_overlap_pixels,
        )

    if temporal_config is None and spatial_config is None:
        raise ValueError("Custom VAE tiling needs at least one temporal or spatial tile size.")

    return TilingConfig(
        spatial_config=spatial_config,
        temporal_config=temporal_config,
    ), False


def describe_vae_tiling_config(
    tiling_config: TilingConfig | None,
    auto_tiling: bool,
) -> str:
    if tiling_config is None:
        return "auto (off)" if auto_tiling else "off"

    parts = []
    temporal = tiling_config.temporal_config
    if temporal is not None:
        parts.append(
            "temporal="
            f"{temporal.tile_size_in_frames}/{temporal.tile_overlap_in_frames} frames"
        )
    else:
        parts.append("temporal=off")

    spatial = tiling_config.spatial_config
    if spatial is not None:
        parts.append(
            "spatial="
            f"{spatial.tile_size_in_pixels}/{spatial.tile_overlap_in_pixels} px"
        )
    else:
        parts.append("spatial=off")

    prefix = "auto-selected" if auto_tiling else "explicit"
    return prefix + " (" + ", ".join(parts) + ")"


def parse_transformer_layer_selection(value: str | None) -> tuple[int, ...]:
    """Parse comma-separated 0-based layers and inclusive ranges."""
    if value is None or value.strip() == "":
        return ()

    layers = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start = int(start_raw.strip())
                end = int(end_raw.strip())
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid layer range '{part}'"
                ) from exc
            if start < 0 or end < 0 or end < start:
                raise argparse.ArgumentTypeError(
                    "--video-ff-quantize-layers uses non-negative ranges like 40-47"
                )
            layers.extend(range(start, end + 1))
            continue

        try:
            layer = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid layer '{part}' in --video-ff-quantize-layers"
            ) from exc
        if layer < 0:
            raise argparse.ArgumentTypeError(
                "--video-ff-quantize-layers uses 0-based non-negative layer numbers"
            )
        layers.append(layer)

    return tuple(sorted(set(layers)))


def normalize_layout_layers(
    specs: tuple[tuple[str, str], ...],
    layers: tuple[int, ...],
) -> tuple[int, ...]:
    """Canonicalize implicit all-layer layouts so cache keys stay stable."""
    if specs and not layers:
        return DEFAULT_TRANSFORMER_LAYOUT_LAYERS
    return tuple(layers)


def describe_transformer_layers(layers: tuple[int, ...]) -> str:
    if not layers or tuple(layers) == DEFAULT_TRANSFORMER_LAYOUT_LAYERS:
        return "all"
    return ",".join(str(layer) for layer in layers)


def parse_video_ff_quantize_specs(value: str | None) -> tuple[tuple[str, str], ...]:
    """Parse comma-separated target:mode video FF quantization specs."""
    if value is None or value.strip() == "":
        return ()

    specs = []
    seen_targets = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            raw_target, raw_mode = part.split(":", 1)
            mode = raw_mode.strip().lower()
        else:
            raw_target = part
            mode = "mxfp8"
        target = raw_target.strip().lower().replace("-", "_")
        if target not in FF_QUANTIZE_TARGETS:
            valid = ", ".join(FF_QUANTIZE_TARGETS)
            raise argparse.ArgumentTypeError(
                f"Invalid video FF quantization target '{raw_target}'. Valid values: {valid}"
            )
        if mode not in FF_QUANTIZE_MODES:
            valid = ", ".join(FF_QUANTIZE_MODES)
            raise argparse.ArgumentTypeError(
                f"Invalid video FF quantization mode '{mode}'. Valid values: {valid}"
            )
        if target in seen_targets:
            raise argparse.ArgumentTypeError(
                f"Duplicate video FF quantization target '{target}'"
            )
        seen_targets.add(target)
        specs.append((target, mode))

    return tuple(specs)


def _ensure_audio_ff_layout_for_dtype(
    layout_specs: tuple[tuple[str, str], ...],
    audio_ff_dtype: str | None,
) -> tuple[tuple[str, str], ...]:
    """Auto-add audio FF pretranspose when FP16, avoiding FP16 x BF16 -> FP32
    mixed-dtype promotion.  Mirror of ``_ensure_ff_layout_for_dtype`` minus
    the kernel-cliff motivation (audio K=8192 doesn't hit it).
    """
    if not audio_ff_dtype or audio_ff_dtype == "bfloat16":
        return layout_specs
    have = {target for target, _layout in layout_specs}
    additions = []
    for tgt in ("project_in", "project_out"):
        if tgt not in have:
            additions.append((tgt, "pretranspose"))
    if additions:
        return tuple(additions) + tuple(layout_specs)
    return layout_specs


def _ensure_ff_layout_for_dtype(
    layout_specs: tuple[tuple[str, str], ...],
    video_ff_dtype: str | None,
) -> tuple[tuple[str, str], ...]:
    """Auto-add ``project_in/project_out:pretranspose`` when --video-ff-dtype
    is FP16.  Mandatory for two reasons: (1) without a pretransposed dtype-
    baked cache, project_in's nn.Linear promotes FP16 x BF16 -> FP32; (2)
    naive FP16 at K=16384 hits a deeper BlockLoader cliff than BF16 (4.95
    vs 9.51 TFlops/s with vs without pretranspose).  See
    PERFORMANCE_NOTES.md "BlockLoader cliff characterization" entry.
    """
    if not video_ff_dtype or video_ff_dtype == "bfloat16":
        return layout_specs
    have = {target for target, _layout in layout_specs}
    additions = []
    for tgt in ("project_in", "project_out"):
        if tgt not in have:
            additions.append((tgt, "pretranspose"))
    if additions:
        return tuple(additions) + tuple(layout_specs)
    return layout_specs


def parse_video_ff_layout_specs(value: str | None) -> tuple[tuple[str, str], ...]:
    """Parse comma-separated target:layout video FF layout specs."""
    if value is None or value.strip() == "":
        return ()
    if value.strip().lower() in ("off", "none"):
        return ()

    specs = []
    seen_targets = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                "Video FF layout specs must use target:layout, "
                "for example project_in:pretranspose or project_out:pretranspose"
            )
        raw_target, raw_layout = part.split(":", 1)
        target = raw_target.strip().lower().replace("-", "_")
        layout = raw_layout.strip().lower().replace("-", "_")
        if target not in FF_LAYOUT_SPECS:
            valid = ", ".join(FF_LAYOUT_SPECS)
            raise argparse.ArgumentTypeError(
                f"Invalid video FF layout target '{raw_target}'. Valid values: {valid}"
            )
        valid_layouts = FF_LAYOUT_SPECS[target]
        if layout not in valid_layouts:
            valid = ", ".join(valid_layouts)
            raise argparse.ArgumentTypeError(
                f"Invalid layout '{layout}' for video FF target '{target}'. "
                f"Valid values: {valid}"
            )
        if target in seen_targets:
            raise argparse.ArgumentTypeError(
                f"Duplicate video FF layout target '{target}'"
            )
        seen_targets.add(target)
        specs.append((target, layout))

    return tuple(specs)


def parse_video_attn_layout_specs(value: str | None) -> tuple[tuple[str, str], ...]:
    """Parse comma-separated target:layout video attention layout specs."""
    if value is None or value.strip() == "":
        return ()
    if value.strip().lower() in ("off", "none"):
        return ()

    specs = []
    seen_targets = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                "Video attention layout specs must use target:layout, "
                "for example to_out:pretranspose"
            )
        raw_target, raw_layout = part.split(":", 1)
        target = raw_target.strip().lower().replace("-", "_")
        layout = raw_layout.strip().lower().replace("-", "_")
        if target not in ATTN_LAYOUT_SPECS:
            valid = ", ".join(ATTN_LAYOUT_SPECS)
            raise argparse.ArgumentTypeError(
                f"Invalid video attention layout target '{raw_target}'. Valid values: {valid}"
            )
        valid_layouts = ATTN_LAYOUT_SPECS[target]
        if layout not in valid_layouts:
            valid = ", ".join(valid_layouts)
            raise argparse.ArgumentTypeError(
                f"Invalid layout '{layout}' for video attention target '{target}'. "
                f"Valid values: {valid}"
            )
        if target in seen_targets:
            raise argparse.ArgumentTypeError(
                f"Duplicate video attention layout target '{target}'"
            )
        seen_targets.add(target)
        specs.append((target, layout))

    return tuple(specs)


def compute_dtype_name(dtype: mx.Dtype) -> str:
    if dtype == mx.bfloat16:
        return "BF16"
    if dtype == mx.float16:
        return "FP16"
    if dtype == mx.float32:
        return "FP32"
    return str(dtype)


def default_output_dir() -> str:
    """Resolve the default output directory without touching the filesystem."""
    return (
        os.environ.get("DIFFUSERS_OUTPUT_DIR")
        or os.environ.get("OUTPUT_DIR")
        or "outputs"
    )


def sanitize_output_prefix(prefix: str | None) -> str:
    """Keep generated filenames shell-friendly while preserving readable prefixes."""
    prefix = (prefix or "ltx").strip()
    if not prefix:
        prefix = "ltx"
    sanitized = []
    for char in prefix:
        if char.isalnum() or char in ("-", "_", "."):
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("._") or "ltx"


def build_default_output_path(
    output_dir: str | None = None,
    output_prefix: str | None = None,
) -> str:
    """Build a timestamped MP4 path for runs where --output is omitted."""
    directory = Path(output_dir or default_output_dir()).expanduser()
    prefix = sanitize_output_prefix(output_prefix)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(directory / f"{prefix}_{timestamp}.mp4")


def latent_sidecar_path(output_path: str) -> str:
    """Use the requested output stem for the final-latents sidecar."""
    return sidecars.sidecar_path(output_path)


def text_sidecar_path(output_path: str) -> str:
    """Use the requested output stem for the text-conditioning sidecar."""
    return sidecars.sidecar_path(output_path, "_text")


def run_log_sidecar_path(output_path: str) -> str:
    """Use the requested output stem for the run metadata sidecar."""
    return os.path.splitext(output_path)[0] + "_run.json"


def save_text_conditioning_sidecar(
    path: str,
    positive_video_encoding: mx.array,
    negative_video_encoding: mx.array | None,
    positive_mask: mx.array,
    negative_mask: mx.array | None,
    positive_audio_encoding: mx.array | None = None,
    negative_audio_encoding: mx.array | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> None:
    """Save text/video/audio conditioning tensors for later A/B diagnostics.

    Negative fields are optional: when cfg_scale=1.0 distilled modes skip
    negative encoding entirely, so the sidecar contains only positive
    conditioning.  The loader and replay paths fall back to zeros when
    negative fields are absent.
    """
    arrays = {
        "schema_version": mx.array(1, dtype=mx.int32),
        "positive_video_encoding": positive_video_encoding,
        "positive_attention_mask": positive_mask,
    }
    if negative_video_encoding is not None:
        arrays["negative_video_encoding"] = negative_video_encoding
    if negative_mask is not None:
        arrays["negative_attention_mask"] = negative_mask
    if positive_audio_encoding is not None:
        arrays["positive_audio_encoding"] = positive_audio_encoding
    if negative_audio_encoding is not None:
        arrays["negative_audio_encoding"] = negative_audio_encoding

    metadata = {
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
    }
    written = sidecars.save_sidecar(path, arrays, metadata)
    print(f"  Saved text conditioning: {written}")


def save_run_log_sidecar(
    path: str,
    payload: dict,
    timings: RunTimings,
    status: str,
    outputs: dict | None = None,
) -> None:
    """Save human-readable run metadata and timing information."""
    log = dict(payload)
    now = datetime.now(UTC).isoformat()
    log["status"] = status
    log["updated_at"] = now
    if status != "started":
        log["finished_at"] = now
    log["timings"] = timings.to_dict()
    if outputs:
        log["outputs"] = outputs

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  Saved run log: {path}")


def format_duration(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds // 60)
        return f"{minutes}m {seconds - minutes * 60:04.1f}s"
    return f"{seconds:.2f}s"


class RunTimings:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.last_mark = self.started_at
        self.sections: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        self.sections.append((label, now - self.last_mark))
        self.last_mark = now

    def extend(self, sections: list[tuple[str, float]]) -> None:
        """Append externally measured sections and advance the timing cursor."""
        for label, seconds in sections:
            self.sections.append((label, float(seconds)))
        self.last_mark = time.perf_counter()

    def print_summary(self) -> None:
        total = time.perf_counter() - self.started_at
        if not self.sections:
            return

        width = max(len(label) for label, _ in self.sections + [("total", total)])
        print("\nTiming summary:")
        for label, seconds in self.sections:
            print(f"  {label:<{width}}  {format_duration(seconds)}")
        print(f"  {'total':<{width}}  {format_duration(total)}")

    def to_dict(self) -> dict:
        total = time.perf_counter() - self.started_at
        return {
            "sections": [
                {"label": label, "seconds": seconds}
                for label, seconds in self.sections
            ],
            "total_seconds": total,
        }


from LTX_2_MLX.model.text_encoder.encoder import (
    create_av_text_encoder,
    create_av_text_encoder_v2_from_checkpoint,
    create_text_encoder,
    load_av_text_encoder_v2_weights,
    load_av_text_encoder_weights,
    load_text_encoder_weights,
)
from LTX_2_MLX.model.text_encoder.gemma3 import (
    Gemma3Config,
    Gemma3Model,
    load_gemma3_weights,
)
from LTX_2_MLX.model.upscaler import (
    SpatialUpscaler,
    TemporalUpscaler,
    load_spatial_upscaler_weights,
    load_temporal_upscaler_weights,
)
from LTX_2_MLX.model.video_vae.native_encoder import (
    NativeConv3dVideoEncoder,
    NativeConv3dVideoEncoderStatistics,
    load_native_vae_encoder_statistics,
    load_native_vae_encoder_weights,
)
from LTX_2_MLX.pipelines.av_pipeline import (
    AVCFGConfig,
    AVPipeline,
)
from LTX_2_MLX.pipelines.common import ImageCondition
from LTX_2_MLX.pipelines.ic_lora import (
    ControlType,
    ICLoraConfig,
    ICLoraPipeline,
    VideoCondition,
)
from LTX_2_MLX.pipelines.keyframe_interpolation import (
    Keyframe,
    KeyframeInterpolationConfig,
    KeyframeInterpolationPipeline,
)
from LTX_2_MLX.pipelines.two_stage import (
    TwoStageCFGConfig,
    TwoStagePipeline,
)


def _read_checkpoint_config(checkpoint_path: str) -> dict:
    """Read the JSON config from checkpoint metadata."""
    try:
        from LTX_2_MLX.safetensors_header import read_safetensors_metadata
        metadata = read_safetensors_metadata(checkpoint_path)
        return json.loads(metadata.get("config", "{}"))
    except Exception:
        return {}


def _read_transformer_config(checkpoint_path: str | None) -> dict:
    """Read the transformer config from checkpoint metadata."""
    if not checkpoint_path:
        return {}
    config = _read_checkpoint_config(checkpoint_path)
    transformer_config = config.get("transformer", {})
    return transformer_config if isinstance(transformer_config, dict) else {}


def _parse_rope_type_from_metadata(value) -> LTXRopeType:
    if isinstance(value, LTXRopeType):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "split":
            return LTXRopeType.SPLIT
        if normalized == "interleaved":
            raise ValueError(
                "Unsupported checkpoint metadata rope_type=interleaved. "
                "Current LTX-2 checkpoints use split RoPE."
            )
    return LTXRopeType.SPLIT


def _transformer_rope_type_from_config(
    transformer_config: dict,
    context: str,
) -> LTXRopeType:
    if "rope_type" in transformer_config:
        return _parse_rope_type_from_metadata(transformer_config["rope_type"])
    if "split_rope" in transformer_config:
        return _parse_rope_type_from_metadata(transformer_config["split_rope"])

    if transformer_config:
        print(f"  WARNING: {context} metadata missing rope_type; defaulting to split RoPE.")
    else:
        print(f"  WARNING: {context} metadata missing transformer config; defaulting to split RoPE.")
    return LTXRopeType.SPLIT


def _parse_int_metadata(value, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _normalize_positional_embedding_max_pos(value) -> list[int] | None:
    if isinstance(value, (list, tuple)) and value:
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError):
            return None
    return None


def create_vocoder_for_checkpoint(
    checkpoint_path: str,
    compute_dtype: mx.Dtype = mx.bfloat16,
) -> tuple:
    """Create the appropriate vocoder (plain or BWE) based on checkpoint config.

    Returns:
        Tuple of (vocoder_or_bwe, is_bwe).
    """
    config = _read_checkpoint_config(checkpoint_path)
    vocoder_cfg = config.get("vocoder", {})

    if "bwe" not in vocoder_cfg:
        # Plain vocoder (LTX-2.0)
        vocoder = Vocoder(compute_dtype=compute_dtype)
        return vocoder, False

    # BWE vocoder (LTX-2.3)
    inner_cfg = vocoder_cfg.get("vocoder", {})
    bwe_cfg = vocoder_cfg["bwe"]

    # Create inner vocoder (AMP1 + snakebeta)
    inner_vocoder = Vocoder(
        resblock_kernel_sizes=inner_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        upsample_rates=inner_cfg.get("upsample_rates", [6, 5, 2, 2, 2]),
        upsample_kernel_sizes=inner_cfg.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
        resblock_dilation_sizes=inner_cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        upsample_initial_channel=inner_cfg.get("upsample_initial_channel", 1024),
        resblock=inner_cfg.get("resblock", "AMP1"),
        output_sample_rate=bwe_cfg.get("input_sampling_rate", 24000),
        activation=inner_cfg.get("activation", "snakebeta"),
        use_tanh_at_final=inner_cfg.get("use_tanh_at_final", True),
        compute_dtype=compute_dtype,
    )

    # Create BWE generator
    bwe_generator = Vocoder(
        resblock_kernel_sizes=bwe_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        upsample_rates=bwe_cfg.get("upsample_rates", [2]),
        upsample_kernel_sizes=bwe_cfg.get("upsample_kernel_sizes", [4]),
        resblock_dilation_sizes=bwe_cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        upsample_initial_channel=bwe_cfg.get("upsample_initial_channel", 256),
        resblock=bwe_cfg.get("resblock", "AMP1"),
        output_sample_rate=bwe_cfg.get("output_sampling_rate", 48000),
        activation=bwe_cfg.get("activation", "snakebeta"),
        apply_final_activation=False,
        use_tanh_at_final=bwe_cfg.get("use_tanh_at_final", True),
        compute_dtype=compute_dtype,
    )

    # Create MelSTFT
    mel_stft = MelSTFT(
        filter_length=bwe_cfg.get("n_fft", 2048),
        hop_length=bwe_cfg.get("hop_length", 240),
        win_length=bwe_cfg.get("n_fft", 2048),
        n_mel_channels=bwe_cfg.get("num_mels", 128),
    )

    vocoder_with_bwe = VocoderWithBWE(
        vocoder=inner_vocoder,
        bwe_generator=bwe_generator,
        mel_stft=mel_stft,
        input_sampling_rate=bwe_cfg.get("input_sampling_rate", 24000),
        output_sampling_rate=bwe_cfg.get("output_sampling_rate", 48000),
        hop_length=bwe_cfg.get("hop_length", 240),
    )
    return vocoder_with_bwe, True


def print_audio_dtype_summary(compute_dtype: mx.Dtype, is_bwe: bool) -> None:
    """Print the actual audio decode precision policy."""
    dtype_name = compute_dtype_name(compute_dtype)
    print(f"  Audio VAE decoder dtype: {dtype_name}")
    if is_bwe:
        print(
            "  Vocoder+BWE dtype: FP32 island "
            "(matches Lightricks BWE autocast caution)"
        )
    else:
        print(f"  Vocoder dtype: {dtype_name}")


def detect_model_version(checkpoint_path: str) -> str:
    """Detect model version from safetensors checkpoint metadata.

    Returns version string (e.g. "2.3.0") or empty string if unknown.
    """
    try:
        from LTX_2_MLX.safetensors_header import read_safetensors_metadata
        return read_safetensors_metadata(checkpoint_path).get("model_version", "")
    except Exception:
        return ""


def is_v2_model(checkpoint_path: str) -> bool:
    """Check if checkpoint is an LTX-2.3 (V2) model."""
    version = detect_model_version(checkpoint_path)
    return version.startswith("2.3")


def get_vae_config(checkpoint_path: str) -> dict:
    """Read VAE config from checkpoint metadata."""
    try:
        from LTX_2_MLX.safetensors_header import read_safetensors_metadata
        metadata = read_safetensors_metadata(checkpoint_path)
        config = json.loads(metadata.get("config", "{}"))
        return config.get("vae", {})
    except Exception:
        return {}


# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


def progress_bar(iterable, desc=None, total=None):
    """Create a progress bar wrapper.

    ``ascii`` and ``mininterval`` keep Terminal.app from re-rasterizing a
    fancy unicode bar at 10 Hz while MLX is using the GPU.
    """
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, ncols=80, ascii=True, mininterval=2.0)
    return _simple_progress(iterable, desc, total)


def _simple_progress(iterable, desc, total):
    """Simple progress fallback when tqdm is not available.

    Updates at most every 2 s (and skips identical lines) so the fallback
    matches the throttled tqdm path.
    """
    items = list(iterable)
    total = len(items) if total is None else total
    last_print = 0.0
    last_text = ""
    for i, item in enumerate(items):
        now = time.perf_counter()
        text = f"{desc}: {i+1}/{total}"
        if (now - last_print) >= 2.0 and text != last_text:
            print(f"\r{text}", end="", flush=True)
            last_print = now
            last_text = text
        yield item
    print()  # newline after completion


# LTX-2 system prompt for video generation (used during encoding)
T2V_SYSTEM_PROMPT = """Describe the video in extreme detail, focusing on the visual content, without any introductory phrases."""

# System prompt for prompt enhancement (used to expand short prompts into detailed descriptions)
ENHANCE_SYSTEM_PROMPT = """You enhance short video descriptions into detailed prompts for a video generation model. You MUST preserve the exact subject, characters, and scene from the original - never replace or reinterpret them.

Write a single flowing paragraph (4-8 sentences, present tense) covering these elements in order:

1. SHOT: Cinematography term (wide shot, close-up, medium shot, tracking shot, etc.)
2. SCENE: Lighting, color palette, textures, atmosphere, mood
3. ACTION: The core motion described as a natural sequence from beginning to end
4. CHARACTER(S): Physical appearance, distinguishing features, emotion through physical cues (not labels)
5. CAMERA: How and when the camera moves, what appears after the movement
6. AUDIO: Ambient sound, music, or speech (put dialogue in quotation marks)

CRITICAL RULES:
- If the original says "frog" you write about a frog. If it says "robot" you write about a robot. NEVER substitute the subject.
- Use present tense verbs for action and movement
- Match detail level to shot scale (close-ups need more detail than wide shots)
- Output ONLY the enhanced paragraph, nothing else."""


def load_tokenizer(model_path: str):
    """Load the Gemma tokenizer from HuggingFace transformers."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer
    except ImportError as e:
        print(f"Error: transformers library required for tokenizer: {e}")
        print("Install with: pip install transformers")
        return None


def enhance_prompt(
    prompt: str,
    gemma_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Enhance prompt is not available - the Gemma QAT model used for encoding
    cannot do text generation. Returns the original prompt unchanged.
    """
    print("  Prompt enhancement not available (Gemma QAT model cannot generate text)")
    print("  Using original prompt as-is")
    return prompt


def encode_with_gemma(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    max_length: int = 1024,
    use_early_layers_only: bool = False,
) -> tuple:
    """
    Encode a text prompt using the full Gemma 3 + LTX-2 text encoder pipeline.

    Args:
        prompt: Text prompt to encode.
        gemma_path: Path to Gemma 3 weights directory.
        ltx_weights_path: Path to LTX-2 weights (for text encoder projection).
        max_length: Maximum token length.
        use_early_layers_only: If True, use only Layer 0 (input embeddings) to
            preserve token differentiation. Gemma's self-attention homogenizes
            representations by Layer 4, making different prompts indistinguishable.
            Layer 0 preserves ~0.4 correlation at differing tokens vs ~0.999+ at Layer 4+.

    Returns:
        Tuple of (embedding, attention_mask) as MLX arrays.
    """
    print(f"  Loading tokenizer from {gemma_path}...")
    tokenizer = load_tokenizer(gemma_path)
    if tokenizer is None:
        return None, None

    # Match PyTorch tokenizer behavior: left padding with EOS as pad token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)

    # Weights load in their native bfloat16 via mx.load() - no dtype conversion needed.
    load_gemma3_weights(gemma, gemma_path)

    print("  Loading text encoder projection...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, ltx_weights_path)

    # Tokenize prompt directly (skip chat template - it dilutes the signal)
    # Chat template adds ~28 shared tokens, diluting the actual content
    # Without template: 0.71 correlation for blue vs red (good)
    # With template: 0.98 correlation (bad - template tokens dominate)
    print("  Tokenizing prompt...")
    # Match stock LTX-2 LTXVGemmaTokenizer: strip whitespace before tokenizing.
    encoding = tokenizer(
        (prompt or "").strip(),  # Use raw prompt, not chat template
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = mx.array(encoding["input_ids"])
    attention_mask = mx.array(encoding["attention_mask"])

    num_tokens = int(attention_mask.sum())
    print(f"  Token count: {num_tokens}/{max_length}")

    # Run through Gemma to get hidden states
    print("  Running Gemma 3 forward pass (48 layers)...")
    last_hidden, all_hidden_states = gemma(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mx.eval(last_hidden)

    if all_hidden_states is None:
        print("  Error: Gemma model did not return hidden states")
        return None, None

    print(f"  Got {len(all_hidden_states)} hidden states")

    # EXPERIMENTAL: Use only early layers to preserve differentiation
    if use_early_layers_only:
        print("  [EXPERIMENTAL] Using only Layer 0 (input embeddings)...")
        # Layer 0 is the input embeddings before any self-attention
        # This preserves ~0.4 correlation at differing tokens instead of ~0.999+
        encoded = all_hidden_states[0]  # [B, T, 3840]

        # Zero out padding positions
        mask_expanded = attention_mask[:, :, None].astype(encoded.dtype)
        encoded = encoded * mask_expanded

        original_mask = attention_mask.astype(mx.int32)
        mx.eval(encoded)
        mx.eval(original_mask)

        print(f"  Output embedding shape: {encoded.shape}")  # [B, T, 3840]

    else:
        # Run through text encoder pipeline
        # Note: We skip caption_projection here because the transformer has its own
        print("  Processing through text encoder pipeline...")

        # Feature extraction (uses Layer 48 only for best differentiation)
        encoded = text_encoder.feature_extractor.extract_from_hidden_states(
            hidden_states=all_hidden_states,
            attention_mask=attention_mask,
            padding_side="left",
        )

        # Use connector (1D transformer with learnable registers)
        # Earlier testing showed connector homogenizes embeddings, but the model
        # may have been trained to expect connector output format
        print("  Processing through connector...")

        # Convert mask to additive format for connector attention
        connector_mask = (attention_mask.astype(encoded.dtype) - 1) * 1e9
        connector_mask = connector_mask.reshape(
            attention_mask.shape[0], 1, 1, attention_mask.shape[-1]
        )

        encoded, output_mask = text_encoder.embeddings_connector(encoded, connector_mask)
        mx.eval(encoded)

        # Convert mask back to binary for cross-attention
        original_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)

        # Zero out padding positions
        encoded = encoded * original_mask[:, :, None]
        mx.eval(encoded)
        mx.eval(original_mask)

        print(f"  Output embedding shape: {encoded.shape}")  # Should be [B, T+registers, 3840]

    # === MEMORY OPTIMIZATION ===
    # Clear Gemma and text encoder from memory after encoding
    # These are large models (~12GB for Gemma FP16) that are no longer needed
    print("  Clearing Gemma from memory...")
    del gemma
    del text_encoder
    del all_hidden_states
    del last_hidden
    del tokenizer
    gc.collect()
    # Force MLX to release memory
    mx.metal.clear_cache()

    return encoded, original_mask


def encode_av_gemma_batch(
    prompts: list,
    gemma_path: str,
    ltx_weights_path: str,
    ltx_config_path: str | None = None,
    max_length: int = 1024,
) -> list:
    """
    Encode multiple text prompts using one Gemma load.

    Loads Gemma once, materializes compact real-token hidden states for each
    prompt, frees Gemma, then loads the AV text encoder. This avoids overlapping
    Gemma 3 12B weights with the AV connector peak while still avoiding one
    Gemma load per prompt.

    Args:
        prompts: List of text prompts to encode.
        gemma_path: Path to Gemma 3 weights directory.
        ltx_weights_path: Path to LTX-2 AudioVideo text-connector weights.
        ltx_config_path: Optional original checkpoint path for metadata/config reads.
        max_length: Maximum token length (applied to first prompt, others match).

    Returns:
        List of (video_encoding, audio_encoding, attention_mask) tuples, or None on failure.
    """
    def prompt_label(index: int, total: int) -> str:
        if total == 1:
            return "prompt"
        if total == 2:
            return "positive prompt" if index == 0 else "negative prompt"
        return f"prompt {index + 1}/{total}"

    print(f"  Loading tokenizer from {gemma_path}...")
    tokenizer = load_tokenizer(gemma_path)
    if tokenizer is None:
        return None

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, gemma_path)

    # Default: pad to max_length=1024 before Gemma, matching stock LTX-2's
    # LTXVGemmaTokenizer behavior.  Real tokens are still trimmed out of the
    # hidden states after the forward pass, but running Gemma on the padded
    # 1024-token sequence reproduces stock numerics exactly.  Set
    # LTX_PAD_PROMPT_TO_MAX=0 to skip padding (faster on short prompts, but
    # introduces small bf16 drift relative to stock - see docs).
    pad_to_max = os.environ.get("LTX_PAD_PROMPT_TO_MAX", "1") != "0"

    gemma_outputs = []
    for i, prompt in enumerate(prompts):
        label = prompt_label(i, len(prompts))
        print(f"  Tokenizing {label}...")
        # Match stock LTX-2 LTXVGemmaTokenizer: strip whitespace before tokenizing.
        prompt_text = (prompt or "").strip()
        if pad_to_max:
            encoding = tokenizer(
                prompt_text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        else:
            encoding = tokenizer(
                prompt_text,
                return_tensors="np",
                padding=False,
                truncation=True,
                max_length=max_length,
            )

        input_ids = mx.array(encoding["input_ids"])
        attention_mask = mx.array(encoding["attention_mask"])

        num_tokens = int(attention_mask.sum())
        print(f"  Token count: {num_tokens}/{max_length}")

        print(f"  Running Gemma 3 forward pass for {label} (48 layers, "
              f"{input_ids.shape[1]} tokens)...")
        last_hidden, all_hidden_states = gemma(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if all_hidden_states is None:
            print(f"  Error: Gemma model did not return hidden states for {label}")
            gemma_outputs.append({
                "label": label,
                "hidden_states": None,
                "attention_mask": None,
            })
            del last_hidden
            continue

        print(f"  Got {len(all_hidden_states)} hidden states for {label}")

        # Strip padding to real tokens only (matching ComfyUI behavior).
        # Left-padded: real tokens are at the END. The embeddings connector
        # appends registers to extend the sequence to 1024+, so no rounding needed.
        real_token_count = int(attention_mask.sum())
        seq_len = all_hidden_states[0].shape[1]
        if real_token_count < seq_len:
            all_hidden_states = [
                mx.contiguous(h[:, -real_token_count:, :])
                for h in all_hidden_states
            ]
            attention_mask = mx.contiguous(attention_mask[:, -real_token_count:])
            print(f"  Trimmed {label} padding: {seq_len} -> {real_token_count} (real tokens only)")

        # Materialize compact hidden states before freeing Gemma weights. Keeping
        # only real-token hidden states avoids overlapping the AV connector with
        # the 12B text model during prompt encoding.
        mx.eval(*all_hidden_states, attention_mask)
        gemma_outputs.append({
            "label": label,
            "hidden_states": all_hidden_states,
            "attention_mask": attention_mask,
        })

        del last_hidden

    print("  Clearing Gemma from memory before AV text encoder...")
    del gemma
    del tokenizer
    gc.collect()
    mx.clear_cache()

    print("  Loading AV text encoder projection...")
    config_path = ltx_config_path or ltx_weights_path
    if is_v2_model(config_path):
        print("  Detected LTX-2.3 (V2) model - using V2 text encoder")
        text_encoder = create_av_text_encoder_v2_from_checkpoint(config_path)
        load_av_text_encoder_v2_weights(text_encoder, ltx_weights_path)
    else:
        text_encoder = create_av_text_encoder()
        load_av_text_encoder_weights(text_encoder, ltx_weights_path)

    results = []
    for gemma_output in gemma_outputs:
        label = gemma_output["label"]
        all_hidden_states = gemma_output["hidden_states"]
        attention_mask = gemma_output["attention_mask"]

        if all_hidden_states is None or attention_mask is None:
            results.append((None, None, None))
            continue

        print(f"  Processing {label} through AV text encoder pipeline...")
        av_output = text_encoder.encode_from_hidden_states(
            hidden_states=all_hidden_states,
            attention_mask=attention_mask,
            padding_side="left",
        )
        mx.eval(av_output.video_encoding)
        mx.eval(av_output.audio_encoding)

        print(f"  {label.capitalize()} video encoding shape: {av_output.video_encoding.shape}")
        print(f"  {label.capitalize()} audio encoding shape: {av_output.audio_encoding.shape}")

        results.append((av_output.video_encoding, av_output.audio_encoding, av_output.attention_mask))

        del all_hidden_states

    print("  Clearing AV text encoder from memory...")
    del text_encoder
    del gemma_outputs
    gc.collect()
    mx.clear_cache()

    return results


def create_dummy_text_encoding(
    prompt: str,
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,  # Pre-projection dimension (transformer does its own projection)
) -> tuple:
    """
    Create dummy text encoding for testing.

    In production, this should be replaced with actual Gemma encoding.
    Note: Output is 3840-dim because the transformer has its own caption_projection.
    """
    # For now, use random but deterministic encoding based on prompt
    mx.random.seed(hash(prompt) % (2**31))

    # Create text embeddings in pre-projection dimension
    text_encoding = mx.random.normal(shape=(batch_size, max_tokens, embed_dim)) * 0.1
    text_mask = mx.ones((batch_size, max_tokens))

    return text_encoding, text_mask


def create_null_text_encoding(
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,
) -> tuple:
    """
    Create null/empty text encoding for CFG unconditional pass.

    WARNING: This creates zero embeddings which is NOT semantically correct
    for CFG. For proper CFG, the unconditional embedding should be the
    encoding of an empty string through the text encoder. Use
    encode_with_gemma("") when the encoder is available.

    Returns:
        Tuple of (null_encoding, null_mask).
    """
    # Zero embeddings - NOTE: not ideal for CFG, but works as a fallback
    # Proper CFG should use encoded empty string from text encoder
    null_encoding = mx.zeros((batch_size, max_tokens, embed_dim))
    null_mask = mx.zeros((batch_size, max_tokens))  # All masked out

    return null_encoding, null_mask


def rescale_noise_cfg(
    noise_cfg: mx.array,
    noise_pred_text: mx.array,
    guidance_rescale: float = 0.7,
) -> mx.array:
    """
    Rescale CFG output to prevent variance explosion.

    Based on Section 3.4 of "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    (https://arxiv.org/abs/2305.08891). This rescales the CFG output to match the
    variance of the conditional prediction, preventing over-saturation.

    Args:
        noise_cfg: The CFG-combined output (uncond + scale * (cond - uncond)).
        noise_pred_text: The conditional prediction (before CFG).
        guidance_rescale: Factor for blending rescaled vs original CFG.
                         0.0 = no rescaling (original CFG), 1.0 = full rescaling.

    Returns:
        Rescaled CFG output.
    """
    # Per-channel rescaling to fix per-channel biases
    # Shape: [B, C, F, H, W] - compute stats per channel
    # This is more aggressive than standard guidance rescale

    # Compute per-channel mean and std for both predictions
    # Using RMS (root mean square) for std to avoid issues with zero mean
    cfg_mean = mx.mean(noise_cfg, axis=(2, 3, 4), keepdims=True)  # [B, C, 1, 1, 1]
    cfg_std = mx.sqrt(mx.mean((noise_cfg - cfg_mean) ** 2, axis=(2, 3, 4), keepdims=True) + 1e-8)

    text_mean = mx.mean(noise_pred_text, axis=(2, 3, 4), keepdims=True)
    text_std = mx.sqrt(mx.mean((noise_pred_text - text_mean) ** 2, axis=(2, 3, 4), keepdims=True) + 1e-8)

    # Normalize CFG to have same per-channel mean and std as conditional
    noise_pred_rescaled = (noise_cfg - cfg_mean) / cfg_std * text_std + text_mean

    # Blend between original and rescaled based on guidance_rescale factor
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

    return noise_cfg


def load_text_embedding(embedding_path: str) -> tuple:
    """
    Load pre-computed text embedding from file.

    Args:
        embedding_path: Path to .npz file with embedding and attention_mask.

    Returns:
        Tuple of (embedding, attention_mask).
    """
    data = np.load(embedding_path)
    embedding = mx.array(data["embedding"])
    mask = mx.array(data["attention_mask"])

    print(f"  Loaded embedding from {embedding_path}")
    print(f"  Shape: {embedding.shape}")

    if "prompt" in data:
        print(f"  Original prompt: {data['prompt']}")

    return embedding, mask


def load_text_conditioning(embedding_path: str, use_av_encoder: bool) -> dict:
    """Load legacy text embeddings or the richer `_text.npz` conditioning sidecar."""
    arrays, metadata = sidecars.load_sidecar(embedding_path)

    if "positive_video_encoding" not in arrays:
        embedding, mask = load_text_embedding(embedding_path)
        loaded = {
            "format": "legacy",
            "positive_video_encoding": embedding,
            "positive_attention_mask": mask,
            "negative_video_encoding": None,
            "negative_attention_mask": None,
            "positive_audio_encoding": None,
            "negative_audio_encoding": None,
        }
        if use_av_encoder:
            print("  WARNING: Legacy pre-computed embeddings do not include audio conditioning.")
        return loaded

    loaded = {
        "format": "text_conditioning",
        "positive_video_encoding": arrays["positive_video_encoding"],
        "positive_attention_mask": arrays["positive_attention_mask"],
        "negative_video_encoding": arrays.get("negative_video_encoding"),
        "negative_attention_mask": arrays.get("negative_attention_mask"),
        "positive_audio_encoding": arrays.get("positive_audio_encoding"),
        "negative_audio_encoding": arrays.get("negative_audio_encoding"),
    }

    print(f"  Loaded text conditioning from {embedding_path}")
    print(f"  Video shape: {loaded['positive_video_encoding'].shape}")
    if loaded["positive_audio_encoding"] is not None:
        print(f"  Audio shape: {loaded['positive_audio_encoding'].shape}")
    if metadata.get("prompt"):
        print(f"  Original prompt: {metadata['prompt']}")
    return loaded


def _cast_bf16_weights_to(model, target_dtype: mx.Dtype) -> int:
    """Cast remaining BF16 weight tensors to the transformer compute dtype.

    The weights cache stores tensors at checkpoint dtype (BF16), plus any
    FF-dtype-baked FP16 entries; constructing the model with a compute
    dtype only affects activation casts at the forward boundary.  Running
    FP16 activations against BF16 weights silently promotes every matmul
    to FP32 (MLX type promotion: bfloat16 x float16 -> float32), which is
    both slower than BF16 and defeats the FP16 request.  So a non-BF16
    transformer must cast its weights once at load.

    Walks every module entry (including underscore-private cached layout
    tensors, which `parameters()` filters out) and casts ONLY bfloat16
    arrays: integer/packed-quant tensors, FP32 islands, and already-FP16
    FF weights pass through unchanged.  Evaluates in chunks so the cast
    streams from the lazy cache load instead of materializing two full
    copies of the model.
    """
    # FP16 is the only target narrower than the BF16 source; an out-of-range
    # weight would silently cast to inf. The distilled DiT maxes at ~9.6, so
    # this never fires today -- it enforces the assumption rather than leaving
    # a silent-NaN hole for a future checkpoint. (Mirrors _cast_for_cache on
    # the cached path; this is the rare cache-off branch.)
    guard_fp16 = target_dtype == mx.float16
    count = 0
    pending: list[mx.array] = []
    for module in model.modules():
        for key, value in module.items():
            if isinstance(value, mx.array) and value.dtype == mx.bfloat16:
                if guard_fp16:
                    max_abs = float(mx.max(mx.abs(value.astype(mx.float32))))
                    if max_abs > 65504.0:
                        raise SystemExit(
                            f"ERROR: {key} max|w|={max_abs:.4g} exceeds float16 "
                            "range; use --transformer-dtype bfloat16 for this "
                            "checkpoint."
                        )
                new_value = value.astype(target_dtype)
                module[key] = new_value
                pending.append(new_value)
                count += 1
                if len(pending) >= 256:
                    mx.eval(*pending)
                    pending.clear()
    if pending:
        mx.eval(*pending)
    return count


def load_transformer(
    weights_path: str,
    num_layers: int = 48,
    compute_dtype: mx.Dtype = mx.bfloat16,
    low_memory: bool = False,
    fast_mode: bool = False,
    profile_transformer_once: bool = False,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_layout_specs: tuple[tuple[str, str], ...] = (),
    video_ff_layout_layers: tuple[int, ...] = (),
    video_attn_layout_specs: tuple[tuple[str, str], ...] = (),
    video_attn_layout_layers: tuple[int, ...] = (),
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    weights_cache_mode: str = "off",
    weights_cache_dir: str | None = None,
    transformer_block_resident_blocks: int = 0,
    transformer_block_compile: bool = False,
    transformer_block_compile_group_size: int = 0,
    video_ff_dtype: mx.Dtype | None = None,
) -> LTXModel:
    """Load transformer with weights.

    Args:
        weights_path: Path to safetensors weights file.
        num_layers: Number of transformer layers.
        compute_dtype: Dtype for computation.
        low_memory: If True, use more frequent eval checkpoints for lower peak memory.
        fast_mode: If True, skip intermediate evaluations.
        profile_transformer_once: If True, print one forced-eval transformer timing trace.
    """
    mem_str = " (low memory)" if low_memory else ""
    fast_str = " (fast mode)" if fast_mode else ""
    profile_str = " (profile first call)" if profile_transformer_once else ""
    print(f"Loading transformer ({compute_dtype_name(compute_dtype)}{mem_str}{fast_str}{profile_str})...")

    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=num_layers,
        cross_attention_dim=4096,
        caption_channels=3840,
        positional_embedding_theta=10000.0,
        compute_dtype=compute_dtype,
        low_memory=low_memory,
        fast_mode=fast_mode,
        profile_transformer_once=profile_transformer_once,
    )

    layouts_loaded_from_cache = False
    # Non-BF16 transformer dtype is baked into the weights cache (hash-keyed
    # like the FF dtypes), so cached loads need no load-time cast.
    bake_dtype = compute_dtype if compute_dtype != mx.bfloat16 else None

    # Load weights
    if weights_path and os.path.exists(weights_path):
        if transformer_block_resident_blocks:
            load_transformer_weights_cached_streaming(
                model,
                weights_path,
                cache_mode=weights_cache_mode,
                cache_root=weights_cache_dir,
                include_audio=False,
                video_ff_layout_specs=video_ff_layout_specs,
                video_ff_layout_layers=video_ff_layout_layers,
                video_attn_layout_specs=video_attn_layout_specs,
                video_attn_layout_layers=video_attn_layout_layers,
                transformer_cache_quantize=transformer_cache_quantize,
                video_ff_quantize_specs=video_ff_quantize_specs,
                video_ff_quantize_layers=video_ff_quantize_layers,
                video_ff_quantize_group_size=video_ff_quantize_group_size,
                video_ff_quantize_bits=video_ff_quantize_bits,
                resident_blocks=transformer_block_resident_blocks,
                transformer_dtype=bake_dtype,
                video_ff_dtype=video_ff_dtype,
            )
            model.transformer_block_compile = transformer_block_compile
            model.transformer_block_compile_group_size = transformer_block_compile_group_size
            if (
                video_ff_quantize_specs
                and video_ff_quantize_layers
                and tuple(video_ff_quantize_layers) != DEFAULT_TRANSFORMER_LAYOUT_LAYERS
            ):
                if transformer_block_compile:
                    print(
                        "  Cached streaming FF quantization: disabling resident-group "
                        "compile for partial-layer quantization"
                    )
                model.transformer_block_compile = False
            layouts_loaded_from_cache = True
        elif weights_cache_mode != "off":
            load_transformer_weights_cached(
                model,
                weights_path,
                cache_mode=weights_cache_mode,
                cache_root=weights_cache_dir,
                include_audio=False,
                video_ff_layout_specs=video_ff_layout_specs,
                video_ff_layout_layers=video_ff_layout_layers,
                video_attn_layout_specs=video_attn_layout_specs,
                video_attn_layout_layers=video_attn_layout_layers,
                transformer_cache_quantize=transformer_cache_quantize,
                video_ff_quantize_specs=video_ff_quantize_specs,
                video_ff_quantize_layers=video_ff_quantize_layers,
                video_ff_quantize_group_size=video_ff_quantize_group_size,
                video_ff_quantize_bits=video_ff_quantize_bits,
                transformer_dtype=bake_dtype,
                video_ff_dtype=video_ff_dtype,
            )
            layouts_loaded_from_cache = True
        else:
            if checkpoint_has_fp8_tensors(weights_path):
                raise SystemExit(
                    "ERROR: FP8 checkpoints require the weights cache "
                    "(--weights-cache auto), which dequantizes them at "
                    "build time."
                )
            load_transformer_weights(model, weights_path)
            if bake_dtype is not None:
                cast_count = _cast_bf16_weights_to(model, bake_dtype)
                print(
                    f"  Cast {cast_count} BF16 weight tensors -> "
                    f"{compute_dtype_name(bake_dtype)} (transformer compute dtype)"
                )
        if bake_dtype is not None and (
            transformer_block_resident_blocks or weights_cache_mode != "off"
        ):
            print(
                f"  Transformer dtype baked into cache: "
                f"{compute_dtype_name(bake_dtype)}"
            )
    else:
        print(f"  Warning: Weights not found at {weights_path}, using random init")
    if transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF:
        print(
            "  Transformer cache quantization: "
            f"{transformer_cache_quantize} (MLX-native heavy block linears)"
        )
    if video_ff_quantize_specs:
        if transformer_block_resident_blocks and model.transformer_block_streamer is not None:
            quant_layers = video_ff_quantize_layers or tuple(range(num_layers))
            count = len(quant_layers) * len(video_ff_quantize_specs)
            quant_label = "Cached streaming video FF quantization"
        else:
            count = model.enable_video_ff_quantization(
                quantization_specs=video_ff_quantize_specs,
                group_size=video_ff_quantize_group_size,
                bits=video_ff_quantize_bits,
                layers=video_ff_quantize_layers,
            )
            quant_label = "Experimental video FF quantization"
        layer_str = (
            ",".join(str(layer) for layer in video_ff_quantize_layers)
            if video_ff_quantize_layers
            else "all"
        )
        spec_str = ",".join(f"{target}:{mode}" for target, mode in video_ff_quantize_specs)
        print(
            f"  {quant_label}: "
            f"{count} projections, specs={spec_str}, "
            f"group_size={video_ff_quantize_group_size or 'default'}, "
            f"bits={video_ff_quantize_bits or 'default'}, "
            f"layers={layer_str}"
        )
    if video_ff_layout_specs and not layouts_loaded_from_cache:
        count = model.apply_video_ff_layout(
            layout_specs=video_ff_layout_specs,
            layers=video_ff_layout_layers,
        )
        layer_str = describe_transformer_layers(video_ff_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_ff_layout_specs)
        print(
            "  Video FF layout: "
            f"{count} projections, specs={spec_str}, layers={layer_str}"
        )
    elif video_ff_layout_specs:
        layer_str = describe_transformer_layers(video_ff_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_ff_layout_specs)
        print(
            "  Video FF layout: "
            f"loaded from transformer cache, specs={spec_str}, layers={layer_str}"
        )
    if video_attn_layout_specs and not layouts_loaded_from_cache:
        count = model.apply_video_attn_layout(
            layout_specs=video_attn_layout_specs,
            layers=video_attn_layout_layers,
        )
        layer_str = describe_transformer_layers(video_attn_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_attn_layout_specs)
        print(
            "  Video attention layout: "
            f"{count} projections, specs={spec_str}, layers={layer_str}"
        )
    elif video_attn_layout_specs:
        layer_str = describe_transformer_layers(video_attn_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_attn_layout_specs)
        print(
            "  Video attention layout: "
            f"loaded from transformer cache, specs={spec_str}, layers={layer_str}"
        )
    return model


def load_av_transformer(
    weights_path: str,
    num_layers: int = 48,
    compute_dtype: mx.Dtype = mx.bfloat16,
    low_memory: bool = False,
    fast_mode: bool = False,
    profile_transformer_once: bool = False,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_layout_specs: tuple[tuple[str, str], ...] = (),
    video_ff_layout_layers: tuple[int, ...] = (),
    video_attn_layout_specs: tuple[tuple[str, str], ...] = (),
    video_attn_layout_layers: tuple[int, ...] = (),
    audio_ff_layout_specs: tuple[tuple[str, str], ...] = (),
    audio_ff_layout_layers: tuple[int, ...] = (),
    audio_attn_layout_specs: tuple[tuple[str, str], ...] = (),
    audio_attn_layout_layers: tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    caption_channels: int | None = 3840,
    cross_attention_adaln: bool = False,
    apply_gated_attention: bool = False,
    weights_cache_mode: str = "off",
    weights_cache_dir: str | None = None,
    transformer_block_resident_blocks: int = 0,
    transformer_block_compile: bool = False,
    transformer_block_compile_group_size: int = 0,
    video_ff_dtype: mx.Dtype | None = None,
    audio_ff_dtype: mx.Dtype | None = None,
    config_weights_path: str | None = None,
    double_precision_rope: bool | None = None,
) -> LTXAVModel:
    """Load AudioVideo transformer with weights.

    Args:
        fast_mode: If True, skip intermediate evaluations.
        profile_transformer_once: If True, print one forced-eval transformer timing trace.
        caption_channels: Caption embedding dim (3840 for V1/2.0, None for V2/2.3
            where the feature extractor projects directly to transformer dims).
        cross_attention_adaln: V2 cross-attention AdaLN (prompt_adaln_single).
        apply_gated_attention: V2 per-head gating in attention.
    """
    transformer_config = _read_transformer_config(config_weights_path or weights_path)
    if double_precision_rope is None:
        double_precision_rope = transformer_config.get("frequencies_precision") == "float64"
    rope_type = _transformer_rope_type_from_config(
        transformer_config,
        "AudioVideo transformer",
    )
    positional_embedding_max_pos = _normalize_positional_embedding_max_pos(
        transformer_config.get("positional_embedding_max_pos")
    )
    av_ca_timestep_scale_multiplier = _parse_int_metadata(
        transformer_config.get("av_ca_timestep_scale_multiplier"),
        1000,
    )

    mem_str = " (low memory)" if low_memory else ""
    fast_str = " (fast mode)" if fast_mode else ""
    profile_str = " (profile first call)" if profile_transformer_once else ""
    v2_str = " (V2)" if cross_attention_adaln else ""
    print(f"Loading AudioVideo transformer ({compute_dtype_name(compute_dtype)}{mem_str}{fast_str}{profile_str}{v2_str})...")
    if transformer_config:
        print(
            "  Transformer config: "
            f"rope={rope_type.value}, "
            f"double_precision_rope={'on' if double_precision_rope else 'off'}, "
            f"av_ca_timestep_scale={av_ca_timestep_scale_multiplier}"
        )

    model = LTXAVModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=num_layers,
        cross_attention_dim=4096,
        caption_channels=caption_channels,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=positional_embedding_max_pos,
        rope_type=rope_type,
        use_double_precision_rope=bool(double_precision_rope),
        compute_dtype=compute_dtype,
        low_memory=low_memory,
        fast_mode=fast_mode,
        profile_transformer_once=profile_transformer_once,
        cross_attention_adaln=cross_attention_adaln,
        apply_gated_attention=apply_gated_attention,
        av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
    )

    layouts_loaded_from_cache = False
    # Non-BF16 transformer dtype is baked into the weights cache (hash-keyed
    # like the FF dtypes), so cached loads need no load-time cast.
    bake_dtype = compute_dtype if compute_dtype != mx.bfloat16 else None

    # Load transformer weights, including the audio-token transformer path.
    if weights_path and os.path.exists(weights_path):
        if transformer_block_resident_blocks:
            load_transformer_weights_cached_streaming(
                model,
                weights_path,
                cache_mode=weights_cache_mode,
                cache_root=weights_cache_dir,
                include_audio=True,
                video_ff_layout_specs=video_ff_layout_specs,
                video_ff_layout_layers=video_ff_layout_layers,
                video_attn_layout_specs=video_attn_layout_specs,
                video_attn_layout_layers=video_attn_layout_layers,
                audio_ff_layout_specs=audio_ff_layout_specs,
                audio_ff_layout_layers=audio_ff_layout_layers,
                audio_attn_layout_specs=audio_attn_layout_specs,
                audio_attn_layout_layers=audio_attn_layout_layers,
                adaln_pretranspose=adaln_pretranspose,
                transformer_cache_quantize=transformer_cache_quantize,
                video_ff_quantize_specs=video_ff_quantize_specs,
                video_ff_quantize_layers=video_ff_quantize_layers,
                video_ff_quantize_group_size=video_ff_quantize_group_size,
                video_ff_quantize_bits=video_ff_quantize_bits,
                resident_blocks=transformer_block_resident_blocks,
                transformer_dtype=bake_dtype,
                video_ff_dtype=video_ff_dtype,
                audio_ff_dtype=audio_ff_dtype,
            )
            model.transformer_block_compile = transformer_block_compile
            model.transformer_block_compile_group_size = transformer_block_compile_group_size
            if (
                video_ff_quantize_specs
                and video_ff_quantize_layers
                and tuple(video_ff_quantize_layers) != DEFAULT_TRANSFORMER_LAYOUT_LAYERS
            ):
                if transformer_block_compile:
                    print(
                        "  Cached streaming FF quantization: disabling resident-group "
                        "compile for partial-layer quantization"
                    )
                model.transformer_block_compile = False
            layouts_loaded_from_cache = True
        elif weights_cache_mode != "off":
            load_transformer_weights_cached(
                model,
                weights_path,
                cache_mode=weights_cache_mode,
                cache_root=weights_cache_dir,
                include_audio=True,
                video_ff_layout_specs=video_ff_layout_specs,
                video_ff_layout_layers=video_ff_layout_layers,
                video_attn_layout_specs=video_attn_layout_specs,
                video_attn_layout_layers=video_attn_layout_layers,
                audio_ff_layout_specs=audio_ff_layout_specs,
                audio_ff_layout_layers=audio_ff_layout_layers,
                audio_attn_layout_specs=audio_attn_layout_specs,
                audio_attn_layout_layers=audio_attn_layout_layers,
                adaln_pretranspose=adaln_pretranspose,
                transformer_cache_quantize=transformer_cache_quantize,
                video_ff_quantize_specs=video_ff_quantize_specs,
                video_ff_quantize_layers=video_ff_quantize_layers,
                video_ff_quantize_group_size=video_ff_quantize_group_size,
                video_ff_quantize_bits=video_ff_quantize_bits,
                transformer_dtype=bake_dtype,
                video_ff_dtype=video_ff_dtype,
                audio_ff_dtype=audio_ff_dtype,
            )
            layouts_loaded_from_cache = True
        else:
            if checkpoint_has_fp8_tensors(weights_path):
                raise SystemExit(
                    "ERROR: FP8 checkpoints require the weights cache "
                    "(--weights-cache auto), which dequantizes them at "
                    "build time."
                )
            load_av_transformer_weights(model, weights_path)
            if bake_dtype is not None:
                cast_count = _cast_bf16_weights_to(model, bake_dtype)
                print(
                    f"  Cast {cast_count} BF16 weight tensors -> "
                    f"{compute_dtype_name(bake_dtype)} (transformer compute dtype)"
                )
        if bake_dtype is not None and (
            transformer_block_resident_blocks or weights_cache_mode != "off"
        ):
            print(
                f"  Transformer dtype baked into cache: "
                f"{compute_dtype_name(bake_dtype)}"
            )
    else:
        print(f"  Warning: Weights not found at {weights_path}, using random init")
    if transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF:
        print(
            "  Transformer cache quantization: "
            f"{transformer_cache_quantize} (MLX-native heavy block linears)"
        )
    if video_ff_quantize_specs:
        if transformer_block_resident_blocks and model.transformer_block_streamer is not None:
            quant_layers = video_ff_quantize_layers or tuple(range(num_layers))
            count = len(quant_layers) * len(video_ff_quantize_specs)
            quant_label = "Cached streaming video FF quantization"
        else:
            count = model.enable_video_ff_quantization(
                quantization_specs=video_ff_quantize_specs,
                group_size=video_ff_quantize_group_size,
                bits=video_ff_quantize_bits,
                layers=video_ff_quantize_layers,
            )
            quant_label = "Experimental video FF quantization"
        layer_str = (
            ",".join(str(layer) for layer in video_ff_quantize_layers)
            if video_ff_quantize_layers
            else "all"
        )
        spec_str = ",".join(f"{target}:{mode}" for target, mode in video_ff_quantize_specs)
        print(
            f"  {quant_label}: "
            f"{count} projections, specs={spec_str}, "
            f"group_size={video_ff_quantize_group_size or 'default'}, "
            f"bits={video_ff_quantize_bits or 'default'}, "
            f"layers={layer_str}"
        )
    if video_ff_layout_specs and not layouts_loaded_from_cache:
        count = model.apply_video_ff_layout(
            layout_specs=video_ff_layout_specs,
            layers=video_ff_layout_layers,
        )
        layer_str = describe_transformer_layers(video_ff_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_ff_layout_specs)
        print(
            "  Video FF layout: "
            f"{count} projections, specs={spec_str}, layers={layer_str}"
        )
    elif video_ff_layout_specs:
        layer_str = describe_transformer_layers(video_ff_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_ff_layout_specs)
        print(
            "  Video FF layout: "
            f"loaded from transformer cache, specs={spec_str}, layers={layer_str}"
        )
    if video_attn_layout_specs and not layouts_loaded_from_cache:
        count = model.apply_video_attn_layout(
            layout_specs=video_attn_layout_specs,
            layers=video_attn_layout_layers,
        )
        layer_str = describe_transformer_layers(video_attn_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_attn_layout_specs)
        print(
            "  Video attention layout: "
            f"{count} projections, specs={spec_str}, layers={layer_str}"
        )
    elif video_attn_layout_specs:
        layer_str = describe_transformer_layers(video_attn_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_attn_layout_specs)
        print(
            "  Video attention layout: "
            f"loaded from transformer cache, specs={spec_str}, layers={layer_str}"
        )

    # Audio-side pretranspose for AV blocks (audio_ff, audio_attn1.to_out,
    # audio_attn2.to_out, video_to_audio_attn.to_out).  Same fp16-safe
    # weight.T contiguity trick as the video side, just targeting modules
    # where Q comes from audio.  Cached when --weights-cache is on.
    if audio_ff_layout_specs and not layouts_loaded_from_cache:
        count = model.apply_audio_ff_layout(
            layout_specs=audio_ff_layout_specs,
            layers=audio_ff_layout_layers,
        )
        layer_str = describe_transformer_layers(audio_ff_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in audio_ff_layout_specs)
        print(
            "  Audio FF layout: "
            f"{count} projections, specs={spec_str}, layers={layer_str}"
        )
    elif audio_ff_layout_specs:
        layer_str = describe_transformer_layers(audio_ff_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in audio_ff_layout_specs)
        print(
            "  Audio FF layout: "
            f"loaded from transformer cache, specs={spec_str}, layers={layer_str}"
        )
    if audio_attn_layout_specs and not layouts_loaded_from_cache:
        count = model.apply_audio_attn_layout(
            layout_specs=audio_attn_layout_specs,
            layers=audio_attn_layout_layers,
        )
        layer_str = describe_transformer_layers(audio_attn_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in audio_attn_layout_specs)
        print(
            "  Audio attention layout: "
            f"{count} projections, specs={spec_str}, layers={layer_str}"
        )
    elif audio_attn_layout_specs:
        layer_str = describe_transformer_layers(audio_attn_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in audio_attn_layout_specs)
        print(
            "  Audio attention layout: "
            f"loaded from transformer cache, specs={spec_str}, layers={layer_str}"
        )
    return model


def euler_step(
    latent: mx.array,
    velocity: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """
    Simple Euler step with direct velocity (for placeholder/testing only).

    NOTE: For proper inference, use euler_step_x0 with denoised prediction.
    """
    dt = sigma_next - sigma
    return latent + dt * velocity


def euler_step_x0(
    sample: mx.array,
    denoised: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """
    Perform one Euler diffusion step using X0 (denoised) prediction.

    This matches the PyTorch EulerDiffusionStep.step() which:
    1. Takes denoised sample (not velocity)
    2. Converts to velocity: v = (sample - denoised) / sigma
    3. Applies Euler: x_next = x + dt * v

    Args:
        sample: Current noisy sample
        denoised: Predicted denoised sample (x0)
        sigma: Current noise level
        sigma_next: Next noise level
    """
    # Convert denoised to velocity (matches PyTorch reference)
    velocity = to_velocity(sample, sigma, denoised)

    # Euler step
    dt = sigma_next - sigma
    return sample.astype(mx.float32) + velocity.astype(mx.float32) * dt


def next_valid_frame_count(frame_count: int) -> int:
    """Round up to the next LTX-valid frame count: 8*k + 1."""
    if frame_count <= 1:
        return 1
    return ((frame_count - 1 + 7) // 8) * 8 + 1


def resolve_num_frames(
    *,
    num_frames: int,
    duration_seconds: float | None,
    fps: float,
) -> int:
    """Resolve frame count from explicit frames or duration.

    If duration is provided, ceil() is used so the generated video covers at
    least the requested duration, then the result is rounded up to 8*k + 1.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    if duration_seconds is None:
        return num_frames

    if duration_seconds <= 0:
        raise ValueError(f"duration must be positive, got {duration_seconds}")

    requested_frames = math.ceil(duration_seconds * fps)
    return next_valid_frame_count(requested_frames)


def generate_video(
    prompt: str,
    height: int = 288,
    width: int = 512,
    num_frames: int = 97,  # ~32s at 24fps (97 latent frames -> 769 pixel frames via 8x VAE temporal compression)
    num_steps: int | None = None,
    cfg_scale: float | None = None,
    guidance_rescale: float = 0.7,  # Rescale CFG output to prevent variance explosion
    # Two-stage pipeline parameters
    steps_stage1: int = 15,
    steps_stage2: int = 3,
    cfg_stage1: float | None = None,  # Defaults to cfg_scale if not specified
    seed: int = 42,
    weights_path: str | None = None,
    transformer_weights_path: str | None = None,
    connector_weights_path: str | None = None,
    video_vae_weights_path: str | None = None,
    audio_vae_weights_path: str | None = None,
    vocoder_weights_path: str | None = None,
    config_weights_path: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    output_prefix: str = "ltx",
    use_placeholder: bool = False,
    skip_vae: bool = False,
    embedding_path: str | None = None,
    gemma_path: str | None = None,
    use_gemma: bool = True,
    dtype: str | mx.Dtype = "bfloat16",
    transformer_dtype: str | mx.Dtype | None = None,
    vae_decoder_backend: str = "native",
    model_variant: str = "distilled",
    upscale_spatial: bool = False,
    spatial_upscaler_weights: str = None,
    upscale_temporal: bool = False,
    temporal_upscaler_weights: str = None,
    generate_audio: bool = False,
    internal_audio: str = "auto",
    low_memory: bool = False,
    fast_mode: bool = False,
    profile_transformer_once: bool = False,
    profile_transformer_steps: tuple[int, ...] = (),
    profile_transformer_blocks: tuple[int, ...] = (),
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_layout_specs: tuple[tuple[str, str], ...] = DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    video_ff_layout_layers: tuple[int, ...] = (),
    video_attn_layout_specs: tuple[tuple[str, str], ...] = DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    video_attn_layout_layers: tuple[int, ...] = (),
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    weights_cache_mode: str = "auto",
    weights_cache_dir: str | None = None,
    mlx_cache_limit_gb: float | None = 1.0,
    stream_transformer: bool = False,
    transformer_block_resident_blocks: int = 0,
    transformer_block_compile: bool = False,
    transformer_block_compile_group_size: int = 0,
    # New parameters
    image_path: str = None,
    image_strength: float = 0.95,
    lora_path: str = None,
    lora_strength: float = 1.0,
    lora_configs: list | None = None,
    lora_allow_partial: bool = False,
    stage2_lora_fuse_mode: str = "fresh-total",
    tiled_vae: bool = False,
    vae_tiling_mode: str = "auto",
    vae_temporal_tile_frames: int | None = None,
    vae_temporal_overlap_frames: int = 24,
    vae_spatial_tile_pixels: int | None = None,
    vae_spatial_overlap_pixels: int = 64,
    pipeline_type: str = "text-to-video",
    early_layers_only: bool = False,
    enhance_prompt_flag: bool = False,
    cross_attn_scale: float = 1.0,
    video_ff_dtype: str | None = None,
    audio_ff_dtype: str | None = None,
    distilled_lora: str = None,
    distilled_lora_scale: float = 1.0,
    stg_scale: float = 0.0,
    stg_mode: str = "video",
    apg_scale: float = 1.0,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    apg_momentum: float = 0.0,
    control_video: str = None,
    control_type: str = "raw",
    canny_low: int = 100,
    canny_high: int = 200,
    control_strength: float = 0.95,
    save_control: bool = False,
    save_latents: bool = False,
    save_text_embeddings: bool = False,
    save_run_log: bool = False,
    save_audio_sidecar: bool = False,
    ge_gamma: float = 0.0,
    output_fps: float = NATIVE_FPS,
    output_tier: str = "default",
    # Output backend / VideoToolbox post-processing
    output_backend: str = "auto",
    vsr_spatial_mode: str = "off",
    vsr_target_fps: float | None = None,
    vsr_temporal_mode: str = "normal",
    vsr_save_original: bool = False,
    vsr_encode_quality: float = 0.65,
    vsr_audio_codec: str = "alac",
    # Audio onset (sequence-start spike) mitigation
    audio_onset_trim_mode: str = "auto",
    audio_onset_trim_ms: float | None = None,
    # IC-LoRA and Keyframe Interpolation
    keyframes: list = None,
    ic_lora_weights: str = None,
    # Audio guidance (LTX-2.3 reference defaults)
    audio_cfg_scale: float = None,  # None = use 7.0 default
    rescale_scale: float = None,    # None = use 0.7 default
    negative_prompt: str = None,    # None = use default negative prompt
):
    """Generate video from text prompt."""

    requested_num_steps = num_steps

    if output_path is None:
        output_path = build_default_output_path(output_dir, output_prefix)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if num_steps is None:
        num_steps = 30 if model_variant == "dev" else 8
    if cfg_scale is None:
        cfg_scale = 5.0 if model_variant == "dev" else 1.0

    weights_path = resolve_default_ltx_weights(weights_path, model_variant)
    transformer_weights_path = resolve_weight_source(transformer_weights_path, weights_path)
    connector_weights_path = resolve_weight_source(connector_weights_path, weights_path)
    video_vae_weights_path = resolve_weight_source(video_vae_weights_path, weights_path)
    audio_vae_weights_path = resolve_weight_source(audio_vae_weights_path, weights_path)
    vocoder_weights_path = resolve_weight_source(vocoder_weights_path, weights_path)
    config_weights_path = resolve_weight_source(config_weights_path, weights_path)
    spatial_upscaler_weights = resolve_default_spatial_upscaler_weights(
        spatial_upscaler_weights,
        config_weights_path,
    )
    gemma_path = resolve_default_gemma_path(gemma_path)

    v2 = config_weights_path and is_v2_model(config_weights_path)
    distilled_two_stage_requested = pipeline_type == "distilled" and v2
    distilled_single_pass_requested = (
        model_variant == "distilled" and pipeline_type in {"text-to-video", "one-stage"} and v2
    )
    if stage2_lora_fuse_mode not in {"delta", "fresh-total"}:
        raise ValueError(
            "stage2_lora_fuse_mode must be 'delta' or 'fresh-total', "
            f"got {stage2_lora_fuse_mode!r}"
        )
    if stage2_lora_fuse_mode != "delta" and not (
        distilled_two_stage_requested or pipeline_type == "two-stage"
    ):
        print(
            "  WARNING: --stage2-lora-fuse-mode only affects two-stage "
            "generation; ignoring it for this pipeline."
        )
    if (
        distilled_two_stage_requested
        and requested_num_steps is not None
        and requested_num_steps != len(DISTILLED_SIGMA_VALUES) - 1
    ):
        print(
            "  WARNING: --pipeline distilled uses the official fixed 8+3 "
            "two-stage sigma schedule; ignoring --steps."
        )
    if (
        distilled_single_pass_requested
        and requested_num_steps is not None
        and requested_num_steps != len(DISTILLED_SIGMA_VALUES) - 1
    ):
        print(
            "  WARNING: distilled one-stage uses the official fixed 8-step "
            "sigma schedule; ignoring --steps."
        )
        num_steps = len(DISTILLED_SIGMA_VALUES) - 1
    if distilled_two_stage_requested and (height % 64 != 0 or width % 64 != 0):
        new_height = ((height + 63) // 64) * 64
        new_width = ((width + 63) // 64) * 64
        print("  WARNING: Distilled two-stage requires resolution divisible by 64.")
        print(f"  Adjusting resolution from {height}x{width} to {new_height}x{new_width}")
        height = new_height
        width = new_width

    if stream_transformer:
        if transformer_block_resident_blocks == 0:
            transformer_block_resident_blocks = 16
        transformer_block_compile = True
        if transformer_block_compile_group_size == 0:
            transformer_block_compile_group_size = min(4, transformer_block_resident_blocks)

    lora_requested = bool(lora_configs or lora_path or distilled_lora or ic_lora_weights)
    if transformer_block_resident_blocks and lora_requested:
        raise ValueError(
            "LoRA fusion is not supported with transformer block streaming yet. "
            "Disable --stream-transformer / --transformer-block-resident-blocks, "
            "or run without --lora / --distilled-lora / --ic-lora-weights."
        )

    video_ff_layout_layers = normalize_layout_layers(
        video_ff_layout_specs,
        video_ff_layout_layers,
    )
    video_attn_layout_layers = normalize_layout_layers(
        video_attn_layout_specs,
        video_attn_layout_layers,
    )
    video_ff_quantize_layers = normalize_layout_layers(
        video_ff_quantize_specs,
        video_ff_quantize_layers,
    )
    if transformer_cache_quantize not in TRANSFORMER_CACHE_QUANTIZE_MODES:
        raise ValueError(
            f"--transformer-cache-quantize must be one of: "
            f"{', '.join(TRANSFORMER_CACHE_QUANTIZE_MODES)}"
        )
    if (
        transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF
        and weights_cache_mode == "off"
    ):
        weights_cache_mode = "auto"
    transformer_cache_quantize_layouts_disabled = False
    if transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF:
        if video_ff_quantize_specs:
            raise ValueError("--transformer-cache-quantize and --video-ff-quantize are separate experiments")
        transformer_cache_quantize_layouts_disabled = bool(
            video_ff_layout_specs or video_attn_layout_specs
        )
        video_ff_layout_specs = ()
        video_ff_layout_layers = ()
        video_attn_layout_specs = ()
        video_attn_layout_layers = ()

    compute_dtype = parse_compute_dtype(dtype)
    # The transformer (denoise) dtype may diverge from the rest of the
    # pipeline: VAE, audio decoder, vocoder, and text encoding keep
    # `compute_dtype` while the DiT runs at `transformer_compute_dtype`.
    # Boundary casts are already in place (the model casts its inputs, the
    # VAE decoder casts its latents), so the split needs no pipeline glue.
    transformer_compute_dtype = (
        parse_compute_dtype(transformer_dtype) if transformer_dtype else compute_dtype
    )
    if not (prompt or "").strip() and not embedding_path:
        raise SystemExit(
            "ERROR: prompt is empty. If you passed a shell variable "
            '(e.g. "$PROMPT"), it did not expand in this shell - check with '
            "'echo \"$PROMPT\"'. Pass --embedding to reuse saved text "
            "conditioning if an empty prompt is intentional."
        )
    if transformer_compute_dtype != mx.bfloat16:
        _t_dtype_str = (
            "float16" if transformer_compute_dtype == mx.float16 else "float32"
        )
        for _ff_flag, _ff_value in (
            ("--video-ff-dtype", video_ff_dtype),
            ("--audio-ff-dtype", audio_ff_dtype),
        ):
            if _ff_value is not None and _ff_value != _t_dtype_str:
                raise SystemExit(
                    f"ERROR: {_ff_flag} {_ff_value} conflicts with transformer "
                    f"dtype {_t_dtype_str}; the FF interior is part of the "
                    "transformer and --transformer-dtype subsumes the FF dtype "
                    "flags."
                )
        # Subsumed: normalize the FF dtype flags away so the weights-cache
        # hash is identical whether or not they were passed. The FP16 FF
        # pretranspose auto-adds key off the transformer dtype instead.
        video_ff_dtype = None
        audio_ff_dtype = None
    requested_profile_steps = set(profile_transformer_steps or ())
    if profile_transformer_once:
        requested_profile_steps.add(1)
    # Distilled two-stage chains stage_1 (8 steps) and stage_2 (3 steps), so
    # callers can profile global steps 1..11 (e.g. 9 = stage_2 step 1).
    profile_step_upper = num_steps
    if distilled_two_stage_requested:
        profile_step_upper = num_steps + (len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1)
    active_profile_steps = tuple(
        step for step in sorted(requested_profile_steps)
        if step <= profile_step_upper
    )
    ignored_profile_steps = tuple(
        step for step in sorted(requested_profile_steps)
        if step > profile_step_upper
    )
    active_profile_blocks = tuple(
        block for block in sorted(set(profile_transformer_blocks or ()))
        if block < 48
    )
    ignored_profile_blocks = tuple(
        block for block in sorted(set(profile_transformer_blocks or ()))
        if block >= 48
    )
    if video_ff_quantize_specs and video_ff_layout_specs:
        raise ValueError("--video-ff-quantize and --video-ff-layout should be tested separately")
    if transformer_block_compile and not transformer_block_resident_blocks:
        raise ValueError("--transformer-block-compile requires --transformer-block-resident-blocks")
    if transformer_block_compile_group_size and not transformer_block_compile:
        raise ValueError("--transformer-block-compile-group-size requires --transformer-block-compile")
    if transformer_block_compile_group_size and not transformer_block_resident_blocks:
        raise ValueError("--transformer-block-compile-group-size requires --transformer-block-resident-blocks")
    if (
        transformer_block_compile_group_size
        and transformer_block_compile_group_size > transformer_block_resident_blocks
    ):
        raise ValueError("--transformer-block-compile-group-size cannot exceed --transformer-block-resident-blocks")
    if transformer_block_resident_blocks and weights_cache_mode == "off":
        weights_cache_mode = "auto"
    if mlx_cache_limit_gb is not None:
        mlx_cache_limit_bytes = int(mlx_cache_limit_gb * (1000**3))
        mx.set_cache_limit(mlx_cache_limit_bytes)
        mx.clear_cache()
    vae_tiling_config, vae_auto_tiling = build_vae_tiling_config(
        vae_tiling_mode,
        height=height,
        width=width,
        num_frames=num_frames,
        decoder_backend=vae_decoder_backend,
        force_tiled=tiled_vae,
        temporal_tile_frames=vae_temporal_tile_frames,
        temporal_overlap_frames=vae_temporal_overlap_frames,
        spatial_tile_pixels=vae_spatial_tile_pixels,
        spatial_overlap_pixels=vae_spatial_overlap_pixels,
    )
    timings = RunTimings()
    run_metadata = None
    if save_run_log:
        run_metadata = {
            "schema_version": 1,
            "started_at": datetime.now(UTC).isoformat(),
            "argv": sys.argv[:],
            "cwd": os.getcwd(),
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "output_path": output_path,
            "sidecars": {
                "run_log": run_log_sidecar_path(output_path),
                "latents": latent_sidecar_path(output_path) if save_latents else None,
                "text_conditioning": text_sidecar_path(output_path) if save_text_embeddings else None,
            },
            "parameters": {
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_steps": num_steps,
                "cfg_scale": cfg_scale,
                "guidance_rescale": guidance_rescale,
                "seed": seed,
                "output_dir": os.path.dirname(output_path),
                "output_prefix": output_prefix,
                "weights_path": weights_path,
                "transformer_weights_path": transformer_weights_path,
                "connector_weights_path": connector_weights_path,
                "video_vae_weights_path": video_vae_weights_path,
                "audio_vae_weights_path": audio_vae_weights_path,
                "vocoder_weights_path": vocoder_weights_path,
                "config_weights_path": config_weights_path,
                "gemma_path": gemma_path,
                "embedding_path": embedding_path,
                "use_gemma": use_gemma,
                "use_placeholder": use_placeholder,
                "skip_vae": skip_vae,
                "dtype": dtype if isinstance(dtype, str) else str(dtype),
                "compute_dtype": compute_dtype_name(compute_dtype),
                "transformer_dtype": compute_dtype_name(transformer_compute_dtype),
                "vae_decoder_backend": vae_decoder_backend,
                "vae_tiling_mode": vae_tiling_mode,
                "tiled_vae": tiled_vae,
                "vae_auto_tiling": vae_auto_tiling,
                "vae_temporal_tile_frames": vae_temporal_tile_frames,
                "vae_temporal_overlap_frames": vae_temporal_overlap_frames,
                "vae_spatial_tile_pixels": vae_spatial_tile_pixels,
                "vae_spatial_overlap_pixels": vae_spatial_overlap_pixels,
                "model_variant": model_variant,
                "pipeline_type": pipeline_type,
                "generate_audio": generate_audio,
                "low_memory": low_memory,
                "fast_mode": fast_mode,
                "profile_transformer_once": profile_transformer_once,
                "profile_transformer_steps": list(profile_transformer_steps or ()),
                "profile_transformer_blocks": list(profile_transformer_blocks or ()),
                "active_profile_transformer_steps": list(active_profile_steps),
                "active_profile_transformer_blocks": list(active_profile_blocks),
                "video_ff_quantize_specs": [
                    {"target": target, "mode": mode}
                    for target, mode in video_ff_quantize_specs
                ],
                "video_ff_quantize_group_size": video_ff_quantize_group_size,
                "video_ff_quantize_bits": video_ff_quantize_bits,
                "video_ff_quantize_layers": list(video_ff_quantize_layers),
                "video_ff_layout_specs": [
                    {"target": target, "layout": layout}
                    for target, layout in video_ff_layout_specs
                ],
                "video_ff_layout_layers": list(video_ff_layout_layers),
                "video_attn_layout_specs": [
                    {"target": target, "layout": layout}
                    for target, layout in video_attn_layout_specs
                ],
                "video_attn_layout_layers": list(video_attn_layout_layers),
                "transformer_cache_quantize": transformer_cache_quantize,
                "transformer_cache_quantize_layouts_disabled": transformer_cache_quantize_layouts_disabled,
                "weights_cache_mode": weights_cache_mode,
                "weights_cache_dir": weights_cache_dir,
                "mlx_cache_limit_gb": mlx_cache_limit_gb,
                "stream_transformer": stream_transformer,
                "transformer_block_resident_blocks": transformer_block_resident_blocks,
                "transformer_block_compile": transformer_block_compile,
                "transformer_block_compile_group_size": transformer_block_compile_group_size,
                "save_latents": save_latents,
                "save_text_embeddings": save_text_embeddings,
                "save_run_log": save_run_log,
                "save_audio_sidecar": save_audio_sidecar,
                "audio_onset_trim_mode": audio_onset_trim_mode,
                "audio_onset_trim_ms": audio_onset_trim_ms,
                "output_fps": output_fps,
                "output_tier": output_tier,
                "image_path": image_path,
                "image_strength": image_strength,
                "lora_path": lora_path,
                "lora_strength": lora_strength,
                "upscale_spatial": upscale_spatial,
                "upscale_temporal": upscale_temporal,
                "stg_scale": stg_scale,
                "stg_mode": stg_mode,
                "apg_scale": apg_scale,
                "apg_eta": apg_eta,
                "apg_norm_threshold": apg_norm_threshold,
                "apg_momentum": apg_momentum,
                "control_video": control_video,
                "control_type": control_type,
                "control_strength": control_strength,
                "ge_gamma": ge_gamma,
                "audio_cfg_scale": audio_cfg_scale,
                "rescale_scale": rescale_scale,
            },
            # Output-affecting env-var flags.  These aren't CLI parameters but
            # they materially change the bits the model produces, so capture
            # them here so post-hoc forensics on a saved run can tell which
            # mode generated the output.
            "env_flags": {
                "LTX_NORMALIZE_AUDIO_NOISE": bool(
                    os.environ.get("LTX_NORMALIZE_AUDIO_NOISE")
                ),
                "LTX_VELOCITY_MODE": bool(os.environ.get("LTX_VELOCITY_MODE")),
                "LTX_PAD_PROMPT_TO_MAX": (
                    os.environ.get("LTX_PAD_PROMPT_TO_MAX", "1") != "0"
                ),
            },
        }

    print(f"\n{'='*50}")
    print("LTX-2 MLX Video Generation")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}, {num_frames} frames")
    steps_display = (
        f"{len(DISTILLED_SIGMA_VALUES) - 1}+{len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1}"
        if distilled_two_stage_requested
        else str(num_steps)
    )
    print(f"Steps: {steps_display}, CFG: {cfg_scale}, Seed: {seed}")
    print(f"Model variant: {model_variant}")
    for source_line in describe_weight_sources(
        bundle=weights_path,
        config=config_weights_path,
        transformer=transformer_weights_path,
        connector=connector_weights_path,
        video_vae=video_vae_weights_path,
        audio_vae=audio_vae_weights_path,
        vocoder=vocoder_weights_path,
    ):
        print(source_line)
    print(f"Compute dtype: {compute_dtype_name(compute_dtype)}")
    if transformer_compute_dtype != compute_dtype:
        print(
            f"Transformer dtype: {compute_dtype_name(transformer_compute_dtype)} "
            "(denoise only; VAE/audio/vocoder keep compute dtype)"
        )
    if not skip_vae:
        print(f"VAE tiling: {describe_vae_tiling_config(vae_tiling_config, vae_auto_tiling)}")
    if skip_vae:
        print("VAE decoding: SKIPPED")
    elif vae_decoder_backend == "native":
        print("VAE decoder: native Conv3d")
    else:
        raise ValueError(f"Unsupported VAE decoder backend: {vae_decoder_backend}")
    if not skip_vae:
        print("VAE spatial padding: zero (boundary-flicker mitigation)")
    if upscale_spatial:
        print(f"Spatial upscaling: 2x (output will be {width*2}x{height*2})")
    if upscale_temporal:
        print(f"Temporal upscaling: 2x (frames will be ~{num_frames*2})")
    if generate_audio:
        print("Audio generation: ENABLED (stereo 24kHz)")

    # Resolve and report internal-audio state.  Validation: --internal-audio off
    # with --generate-audio is incoherent - the audio branch produces what the
    # decoder needs.  Reject it loudly rather than silently turning off audio.
    if internal_audio == "off" and generate_audio:
        raise SystemExit(
            "error: --internal-audio off cannot be combined with --generate-audio "
            "(audio output requires the internal audio branch).  Drop one of the flags."
        )
    _env_disable = bool(os.environ.get("LTX_DISABLE_INTERNAL_AUDIO"))
    if _env_disable:
        _internal_resolved_state = "off"
        _internal_source = "LTX_DISABLE_INTERNAL_AUDIO env"
    elif internal_audio == "on":
        _internal_resolved_state = "on"
        _internal_source = "--internal-audio on"
    elif internal_audio == "off":
        _internal_resolved_state = "off"
        _internal_source = "--internal-audio off"
    else:  # auto
        _internal_resolved_state = "on" if generate_audio else "off"
        _internal_source = f"auto (--generate-audio={generate_audio})"
    print(
        f"Internal audio branch: {_internal_resolved_state.upper()} "
        f"[{_internal_source}]"
    )
    if _internal_resolved_state == "on" and not generate_audio:
        print(
            "  Note: internal audio runs but no audio output will be saved.  "
            "Pass --internal-audio off (or just omit --internal-audio) to skip "
            "the audio branch entirely for a meaningful per-step speedup."
        )
    if save_latents:
        print("Latent sidecar: ENABLED")
    if save_text_embeddings:
        print("Text conditioning sidecar: ENABLED")
    if save_run_log:
        print("Run log sidecar: ENABLED")
        save_run_log_sidecar(
            run_log_sidecar_path(output_path),
            run_metadata,
            timings,
            status="started",
            outputs={
                "video": output_path,
                "audio_wav": os.path.splitext(output_path)[0] + ".wav" if generate_audio else None,
                "latents": latent_sidecar_path(output_path) if save_latents else None,
                "text_conditioning": text_sidecar_path(output_path) if save_text_embeddings else None,
            },
        )
    if low_memory:
        print("Low memory mode: ENABLED (sequential CFG, aggressive eval)")
    if fast_mode:
        print("Fast mode: ENABLED (no intermediate evals)")
    if transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF:
        print(
            "Transformer cache quantization: ENABLED "
            f"(mode={transformer_cache_quantize})"
        )
        if transformer_cache_quantize_layouts_disabled:
            print("  Same-math transformer layouts disabled for quantized cache")
    if video_ff_quantize_specs:
        layer_str = (
            ",".join(str(layer) for layer in video_ff_quantize_layers)
            if video_ff_quantize_layers
            else "all"
        )
        spec_str = ",".join(f"{target}:{mode}" for target, mode in video_ff_quantize_specs)
        print(
            "Experimental video FF quantization: ENABLED "
            f"(specs={spec_str}, "
            f"group_size={video_ff_quantize_group_size or 'default'}, "
            f"bits={video_ff_quantize_bits or 'default'}, "
            f"layers={layer_str})"
        )
    if video_ff_layout_specs:
        layer_str = describe_transformer_layers(video_ff_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_ff_layout_specs)
        print(
            "Video FF layout: ENABLED "
            f"(specs={spec_str}, layers={layer_str})"
        )
    if video_attn_layout_specs:
        layer_str = describe_transformer_layers(video_attn_layout_layers)
        spec_str = ",".join(f"{target}:{layout}" for target, layout in video_attn_layout_specs)
        print(
            "Video attention layout: ENABLED "
            f"(specs={spec_str}, layers={layer_str})"
        )
    if weights_cache_mode != "off":
        cache_dir_str = weights_cache_dir or "default"
        print(
            "Weights cache: ENABLED "
            f"(mode={weights_cache_mode}, dir={cache_dir_str})"
        )
    partial_cached_ff_quant = (
        video_ff_quantize_specs
        and transformer_block_resident_blocks
        and video_ff_quantize_layers
        and tuple(video_ff_quantize_layers) != DEFAULT_TRANSFORMER_LAYOUT_LAYERS
    )
    if stream_transformer:
        group_desc = transformer_block_compile_group_size or transformer_block_resident_blocks
        compile_desc = (
            "compile disabled for partial FF quant"
            if partial_cached_ff_quant
            else f"compile, group {group_desc}"
        )
        print(
            "Transformer streaming preset: ENABLED "
            f"(r{transformer_block_resident_blocks}, {compile_desc})"
        )
    if transformer_block_resident_blocks:
        print(
            "Transformer block streaming: ENABLED "
            f"({transformer_block_resident_blocks} resident blocks)"
        )
        if transformer_block_compile and not partial_cached_ff_quant:
            if transformer_block_compile_group_size:
                print(
                    "Transformer block compile: ENABLED "
                    f"(experimental mx.compile, {transformer_block_compile_group_size}-block groups)"
                )
            else:
                print("Transformer block compile: ENABLED (experimental resident-group mx.compile)")
    if mlx_cache_limit_gb is not None:
        print(f"MLX cache limit: {mlx_cache_limit_gb:g} GB")
    if active_profile_steps:
        steps_str = ", ".join(str(step) for step in active_profile_steps)
        print(f"Transformer profile: ENABLED (denoise steps {steps_str}, forced eval diagnostics)")
        if active_profile_blocks:
            blocks_str = ", ".join(str(block) for block in active_profile_blocks)
            print(f"Transformer block detail: ENABLED (blocks {blocks_str})")
    if ignored_profile_steps:
        steps_str = ", ".join(str(step) for step in ignored_profile_steps)
        print(f"Transformer profile: ignoring out-of-range steps {steps_str} for {num_steps} denoise steps")
    if ignored_profile_blocks:
        blocks_str = ", ".join(str(block) for block in ignored_profile_blocks)
        print(f"Transformer profile: ignoring out-of-range blocks {blocks_str} for 48 transformer blocks")
    if stg_scale > 0:
        print(f"STG guidance: scale={stg_scale}, mode={stg_mode}")
    if apg_scale != 1.0:
        print(f"APG guidance: scale={apg_scale}, eta={apg_eta}, norm_threshold={apg_norm_threshold}")
        if apg_momentum > 0:
            print(f"  Using stateful APG with momentum={apg_momentum}")
    if control_video:
        print(f"Control video: {control_video} (type={control_type}, strength={control_strength})")
        if control_type == "canny":
            print(f"  Canny thresholds: low={canny_low}, high={canny_high}")
    if ge_gamma > 0:
        print(f"GE denoising: gamma={ge_gamma} (velocity correction enabled)")
    if embedding_path:
        print(f"Using pre-computed embedding: {embedding_path}")
    elif use_gemma:
        print(f"Text encoder: Gemma 3 at {gemma_path}")
    else:
        print("Text encoder: DUMMY (testing mode)")

    # Set seed
    mx.random.seed(seed)

    # Compute latent dimensions
    # VAE: 32x spatial, 8x temporal compression
    latent_height = height // 32
    latent_width = width // 32
    latent_frames = (num_frames - 1) // 8 + 1

    print(f"\nLatent shape: {latent_frames}x{latent_height}x{latent_width}")

    # Enhance prompt if requested (expands short prompts to detailed descriptions)
    if enhance_prompt_flag and use_gemma:
        print("\n[0/5] Enhancing prompt...")
        prompt = enhance_prompt(prompt, gemma_path)
        print("  Using enhanced prompt for generation")
    timings.mark("setup")

    # Get text encoding
    # Initialize audio encodings (used only when generate_audio=True or V2.3)
    text_audio_encoding = None
    null_audio_encoding = None

    # V2.3 always uses the AV text encoder (dual video/audio embeddings).
    # Use the explicit config source for architecture/vocoder decisions so
    # transformer-only overrides can still pair with stock auxiliary weights.
    use_av_encoder = generate_audio or v2
    requested_weight_families: dict[str, str] = {}
    if use_gemma and not embedding_path:
        requested_weight_families["connector"] = connector_weights_path
    if (
        not skip_vae
        or image_path
        or pipeline_type in {"distilled", "two-stage", "ic-lora", "keyframe-interpolation"}
    ):
        requested_weight_families["video_vae"] = video_vae_weights_path
    if generate_audio:
        requested_weight_families["audio_vae"] = audio_vae_weights_path
        requested_weight_families["vocoder"] = vocoder_weights_path

    weight_family_load_paths = dict(requested_weight_families)
    if requested_weight_families and weights_cache_mode != "off" and not use_placeholder:
        weight_family_load_paths = maybe_cache_weight_families(
            requested_weight_families,
            cache_mode=weights_cache_mode,
            cache_root=weights_cache_dir,
        )

    connector_load_path = weight_family_load_paths.get("connector", connector_weights_path)
    video_vae_load_path = weight_family_load_paths.get("video_vae", video_vae_weights_path)
    audio_vae_load_path = weight_family_load_paths.get("audio_vae", audio_vae_weights_path)
    vocoder_load_path = weight_family_load_paths.get("vocoder", vocoder_weights_path)

    print("\n[1/5] Encoding prompt...")
    if embedding_path:
        loaded_conditioning = load_text_conditioning(embedding_path, use_av_encoder)
        text_encoding = loaded_conditioning["positive_video_encoding"]
        text_mask = loaded_conditioning["positive_attention_mask"]
        null_encoding = loaded_conditioning["negative_video_encoding"]
        null_mask = loaded_conditioning["negative_attention_mask"]
        text_audio_encoding = loaded_conditioning["positive_audio_encoding"]
        null_audio_encoding = loaded_conditioning["negative_audio_encoding"]

        if null_encoding is None or null_mask is None:
            null_encoding, null_mask = create_null_text_encoding(
                batch_size=1, max_tokens=text_encoding.shape[1], embed_dim=text_encoding.shape[2],
            )
        if use_av_encoder:
            if text_audio_encoding is None:
                print("  WARNING: Pre-computed embeddings don't include positive audio conditioning. Reusing video conditioning.")
                text_audio_encoding = text_encoding
            if null_audio_encoding is None:
                null_audio_encoding = null_encoding
    elif use_gemma:
        # Check if Gemma weights exist
        if not os.path.exists(gemma_path):
            print(f"\n  ERROR: Gemma weights not found at {gemma_path}")
            print("\n  To download Gemma 3 12B:")
            print("    python scripts/download_gemma.py")
            print("\n  Or use --no-gemma flag to use dummy embeddings for testing")
            return

        if use_av_encoder:
            # Encode both prompt AND negative prompt in one Gemma load, then
            # free Gemma before loading the AV connector to reduce peak memory.
            #
            # Skip the negative encoding entirely for any distilled mode
            # (cfg_scale=1.0 means negative is mathematically a no-op).
            # Two-stage doesn't accept a negative at all.  One-stage now
            # short-circuits the CFG branch in AVPipeline.__call__
            # when both cfg scales are 1.0, so the negative isn't needed
            # there either.  Saves ~5-7s/run by avoiding one Gemma + one
            # AV-connector forward pass.
            #
            # We still write the text-conditioning sidecar when requested,
            # just without the negative fields (the loader and replay path
            # both tolerate missing negative - they fall back to zeros).
            # Re-enable with LTX_ENCODE_UNUSED_NEGATIVE=1 for debugging.
            skip_negative = (
                (distilled_two_stage_requested or distilled_single_pass_requested)
                and not os.environ.get("LTX_ENCODE_UNUSED_NEGATIVE")
            )
            if skip_negative:
                results = encode_av_gemma_batch(
                    prompts=[prompt],
                    gemma_path=gemma_path,
                    ltx_weights_path=connector_load_path,
                    ltx_config_path=config_weights_path,
                )
                if results is None or results[0][0] is None:
                    print("  ERROR: Failed to encode prompt with AV encoder")
                    return
                text_encoding, text_audio_encoding, text_mask = results[0]
                # Negative not used: two-stage doesn't pass it; one-stage
                # short-circuits CFG when cfg_scale == 1.0.
                null_encoding, null_audio_encoding, null_mask = None, None, None
                mode_label = "two-stage" if distilled_two_stage_requested else "one-stage (cfg=1.0)"
                print(f"  Encoded positive prompt only with Gemma 3 "
                      f"(distilled {mode_label} doesn't use negative)")
            else:
                neg_prompt = negative_prompt if negative_prompt else ""
                results = encode_av_gemma_batch(
                    prompts=[prompt, neg_prompt],
                    gemma_path=gemma_path,
                    ltx_weights_path=connector_load_path,
                    ltx_config_path=config_weights_path,
                )
                if results is None or results[0][0] is None:
                    print("  ERROR: Failed to encode prompt with AV encoder")
                    return
                text_encoding, text_audio_encoding, text_mask = results[0]
                null_encoding, null_audio_encoding, null_mask = results[1]
                if null_encoding is None:
                    print("  WARNING: Failed to encode negative prompt, using zeros fallback")
                    null_encoding, null_mask = create_null_text_encoding(
                        batch_size=1, max_tokens=text_encoding.shape[1], embed_dim=text_encoding.shape[2],
                    )
                    null_audio_encoding = null_encoding
                print("  Encoded both prompts with Gemma 3 (AudioVideo, single load)")
        else:
            # Use video-only Gemma encoding (V1/V2.0 only)
            text_encoding, text_mask = encode_with_gemma(
                prompt=prompt,
                gemma_path=gemma_path,
                ltx_weights_path=connector_load_path,
                use_early_layers_only=early_layers_only,
            )
            if text_encoding is None:
                print("  ERROR: Failed to encode prompt")
                return
            print("  Encoded with Gemma 3")
            # Null encoding for non-AV path
            neg_prompt = negative_prompt if negative_prompt else ""
            null_encoding, null_mask = encode_with_gemma(
                prompt=neg_prompt,
                gemma_path=gemma_path,
                ltx_weights_path=connector_load_path,
                max_length=text_encoding.shape[1],
                use_early_layers_only=early_layers_only,
            )
            if null_encoding is None:
                null_encoding, null_mask = create_null_text_encoding(
                    batch_size=1, max_tokens=text_encoding.shape[1], embed_dim=text_encoding.shape[2],
                )
    else:
        text_encoding, text_mask = create_dummy_text_encoding(prompt)
        if generate_audio:
            text_audio_encoding = text_encoding
        null_encoding, null_mask = create_null_text_encoding(
            batch_size=1, max_tokens=text_encoding.shape[1], embed_dim=text_encoding.shape[2],
        )
        if use_av_encoder:
            null_audio_encoding = null_encoding
        print("  Using DUMMY encoding (test mode - output will be random)")
    timings.mark("prompt encoding")

    if save_text_embeddings:
        if text_encoding is None or text_mask is None:
            print("  WARNING: Text conditioning sidecar requested, but positive text encoding is unavailable")
        else:
            # null_encoding/null_mask are None when distilled cfg=1.0
            # skipped the negative encoding - sidecar writer handles
            # that by omitting the negative fields.
            save_text_conditioning_sidecar(
                text_sidecar_path(output_path),
                positive_video_encoding=text_encoding,
                negative_video_encoding=null_encoding,
                positive_mask=text_mask,
                negative_mask=null_mask,
                positive_audio_encoding=text_audio_encoding,
                negative_audio_encoding=null_audio_encoding,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
            )
            timings.mark("text sidecar save")

    # Load model
    # V2.3 always uses the AV transformer (dual video/audio cross-attention)
    if use_av_encoder:
        print("\n[2/5] Loading AudioVideo transformer...")
        if not use_placeholder and transformer_weights_path:
            # Audio pretranspose mirrors video by default for AV models - the
            # audio modules see the same per-step dispatch pattern, so the
            # same weight.T contiguity helps.  Disable with
            # LTX_DISABLE_AUDIO_PRETRANSPOSE=1.
            if os.environ.get("LTX_DISABLE_AUDIO_PRETRANSPOSE"):
                _audio_ff_layout_specs = ()
                _audio_attn_layout_specs = ()
            else:
                _audio_ff_layout_specs = video_ff_layout_specs
                _audio_attn_layout_specs = video_attn_layout_specs
            # If --audio-ff-dtype is FP16, enforce both project_in and
            # project_out pretranspose on the audio side too (independent
            # of whether video FF is also FP16).  Otherwise the audio
            # project_in matmul would do FP16 x BF16 -> FP32 promotion.
            _audio_ff_layout_specs = _ensure_audio_ff_layout_for_dtype(
                _audio_ff_layout_specs,
                audio_ff_dtype
                or (
                    "float16"
                    if transformer_compute_dtype == mx.float16
                    else None
                ),
            )
            # AdaLN pretranspose is cache-integrated (no per-load RAM spike),
            # but measured neutral-to-slight-regression at small T because the
            # per-step adaln matmul count is low (8) and batch is tiny.  Off
            # by default; opt-in with LTX_ADALN_PRETRANSPOSE=1.
            _adaln_pretranspose = bool(os.environ.get("LTX_ADALN_PRETRANSPOSE"))
            model = load_av_transformer(
                transformer_weights_path, num_layers=48,
                compute_dtype=transformer_compute_dtype,
                low_memory=low_memory,
                fast_mode=fast_mode,
                video_ff_quantize_specs=video_ff_quantize_specs,
                video_ff_quantize_group_size=video_ff_quantize_group_size,
                video_ff_quantize_bits=video_ff_quantize_bits,
                video_ff_quantize_layers=video_ff_quantize_layers,
                video_ff_layout_specs=video_ff_layout_specs,
                video_ff_layout_layers=video_ff_layout_layers,
                video_attn_layout_specs=video_attn_layout_specs,
                video_attn_layout_layers=video_attn_layout_layers,
                audio_ff_layout_specs=_audio_ff_layout_specs,
                audio_ff_layout_layers=video_ff_layout_layers,
                audio_attn_layout_specs=_audio_attn_layout_specs,
                audio_attn_layout_layers=video_attn_layout_layers,
                adaln_pretranspose=_adaln_pretranspose,
                transformer_cache_quantize=transformer_cache_quantize,
                weights_cache_mode=weights_cache_mode,
                weights_cache_dir=weights_cache_dir,
                transformer_block_resident_blocks=transformer_block_resident_blocks,
                transformer_block_compile=transformer_block_compile,
                transformer_block_compile_group_size=transformer_block_compile_group_size,
                caption_channels=None if v2 else 3840,
                cross_attention_adaln=v2,
                apply_gated_attention=v2,
                video_ff_dtype=(mx.float16 if video_ff_dtype == "float16" else None),
                audio_ff_dtype=(mx.float16 if audio_ff_dtype == "float16" else None),
                config_weights_path=config_weights_path,
            )
            if video_ff_dtype is not None:
                print(f"  Video FF dtype baked into cache: {video_ff_dtype}")
            if audio_ff_dtype is not None:
                print(f"  Audio FF dtype baked into cache: {audio_ff_dtype}")
        else:
            model = None
            print("  Skipping model load (placeholder mode)")
    else:
        print("\n[2/5] Loading transformer...")
        if not use_placeholder and transformer_weights_path:
            velocity_model = load_transformer(
                transformer_weights_path,
                num_layers=48,
                compute_dtype=transformer_compute_dtype,
                low_memory=low_memory,
                fast_mode=fast_mode,
                profile_transformer_once=1 in active_profile_steps,
                video_ff_quantize_specs=video_ff_quantize_specs,
                video_ff_quantize_group_size=video_ff_quantize_group_size,
                video_ff_quantize_bits=video_ff_quantize_bits,
                video_ff_quantize_layers=video_ff_quantize_layers,
                video_ff_layout_specs=video_ff_layout_specs,
                video_ff_layout_layers=video_ff_layout_layers,
                video_attn_layout_specs=video_attn_layout_specs,
                video_attn_layout_layers=video_attn_layout_layers,
                transformer_cache_quantize=transformer_cache_quantize,
                weights_cache_mode=weights_cache_mode,
                weights_cache_dir=weights_cache_dir,
                transformer_block_resident_blocks=transformer_block_resident_blocks,
                transformer_block_compile=transformer_block_compile,
                transformer_block_compile_group_size=transformer_block_compile_group_size,
                video_ff_dtype=(mx.float16 if video_ff_dtype == "float16" else None),
            )
            if video_ff_dtype is not None:
                print(f"  Video FF dtype baked into cache: {video_ff_dtype}")

            # Apply cross-attention scaling if specified (improves text conditioning)
            if cross_attn_scale != 1.0:
                velocity_model.set_cross_attn_scale(cross_attn_scale, start_layer=40)
                print(f"  Applied cross-attention scale {cross_attn_scale}x for layers 40-47")

            # Wrap in X0Model to convert velocity predictions to denoised (X0)
            # The raw LTXModel outputs velocity, but denoising expects X0 predictions
            model = X0Model(velocity_model)
            print("  Wrapped model with X0Model for denoised predictions")
        else:
            model = None
            print("  Skipping model load (placeholder mode)")
    timings.mark("transformer load")

    stage_scoped_loras_requested = lora_configs_have_stage_strengths(lora_configs)
    stage_scoped_loras_active = stage_scoped_loras_requested and (
        pipeline_type == "two-stage" or distilled_two_stage_requested
    )

    # Apply LoRA(s) if provided. Runtime in-place fusion with a key-translation
    # table (raw LoRA names -> MLX weight keys) that is transpose- and
    # pretranspose-aware and never holds a second full copy of the weights.
    # Supports multiple LoRAs at independent strengths.
    if lora_configs is None and lora_path:
        lora_configs = [LoRAConfig(path=lora_path, strength=lora_strength)]
        stage_scoped_loras_requested = False
        stage_scoped_loras_active = False
    if lora_configs and model is not None:
        if stage_scoped_loras_active:
            print(
                f"\n  Deferring {len(lora_configs)} LoRA(s) to per-stage fusion"
            )
        else:
            if stage_scoped_loras_requested:
                print(
                    "  WARNING: LoRA stage strengths were provided outside a "
                    "two-stage mode; using default --lora-strength values."
                )
            print(
                f"\n  Fusing {len(lora_configs)} LoRA(s) into the loaded model:"
            )
            lora_fuse_start = time.perf_counter()
            fuse_loras_into_model(model, lora_configs, allow_partial=lora_allow_partial)
            lora_fuse_elapsed = time.perf_counter() - lora_fuse_start
            timings.extend([("lora fuse", lora_fuse_elapsed)])
            print(f"  LoRA fusion complete in {format_duration(lora_fuse_elapsed)}")

    stage_managed_lora_configs = lora_configs if stage_scoped_loras_active else None
    if stage_scoped_loras_active and distilled_lora:
        # If another LoRA requested per-stage strengths, route the legacy
        # distilled LoRA through the same staged mechanism as a stage-2-only
        # adapter. This preserves the old no-stage-flags behavior while
        # avoiding the legacy stage-2 special case being skipped.
        stage_managed_lora_configs = list(lora_configs or ()) + [
            LoRAConfig(
                path=distilled_lora,
                strength=distilled_lora_scale,
                stage_1_strength=0.0,
                stage_2_strength=distilled_lora_scale,
            )
        ]
        print(
            f"  Distilled LoRA: {distilled_lora} "
            f"(stage 2 scale {distilled_lora_scale})"
        )

    # Whether to use CFG
    # Distilled models (LTX-2 distilled) are trained without CFG and produce artifacts if CFG > 1.0
    # HOWEVER: Two-stage pipeline specifically uses CFG in Stage 1 (at low res), so we allow it there.
    if model_variant == "distilled" and cfg_scale > 1.2 and pipeline_type != "two-stage":
        print(f"  WARNING: Distilled model requires CFG=1.0 (no guidance). You requested {cfg_scale}.")
        print("  Forcing CFG=1.0 to prevent visual artifacts.")
        cfg_scale = 1.0
        # Distilled uses single-pass for both video AND audio (no CFG at all)
        audio_cfg_scale = 1.0
        rescale_scale = 0.0
        # Disable guidance rescale too since it's irrelevant without CFG
        if guidance_rescale > 0:
            guidance_rescale = 0.0

    use_cfg = cfg_scale > 1.0 and model is not None
    if use_cfg:
        print(f"  CFG enabled with scale {cfg_scale}")
        if guidance_rescale > 0:
            print(f"  Guidance rescale: {guidance_rescale}")
    elif distilled_two_stage_requested:
        print(f"  CFG disabled (scale {cfg_scale}) - Running optimized no-CFG inference")
    else:
        print(f"  CFG disabled (scale {cfg_scale}) - Running optimized single-pass inference")

    # Create APG guider if enabled (replaces CFG when active)
    apg_guider = None
    if apg_scale != 1.0:
        if apg_momentum > 0:
            apg_guider = LegacyStatefulAPGGuider(
                scale=apg_scale,
                eta=apg_eta,
                norm_threshold=apg_norm_threshold,
                momentum=apg_momentum,
            )
        else:
            apg_guider = LtxAPGGuider(
                scale=apg_scale,
                eta=apg_eta,
                norm_threshold=apg_norm_threshold,
            )
        print("  APG guidance enabled (replaces standard CFG)")

    # Create STG guider if enabled
    stg_guider = None
    if stg_scale > 0:
        stg_guider = STGGuider(scale=stg_scale)
        print(f"  STG guidance enabled (scale={stg_scale})")
    timings.mark("guidance setup")

    # Load VAE decoder.
    vae_decoder = None
    vae_decoder_loader = None
    defer_av_decoder_load = use_av_encoder and distilled_two_stage_requested and not skip_vae

    def load_video_vae_decoder():
        # Read VAE config from checkpoint to build correct architecture
        vae_config = get_vae_config(config_weights_path) if config_weights_path else {}
        decoder_blocks = vae_config.get("decoder_blocks", None)
        base_channels = vae_config.get("decoder_base_channels", 128)
        timestep_cond = vae_config.get("timestep_conditioning", True)
        if decoder_blocks:
            print(f"  VAE config: {len(decoder_blocks)} blocks, base_ch={base_channels}, timestep={timestep_cond}")

        if vae_decoder_backend == "native":
            decoder = NativeConv3dVideoDecoder(
                decoder_blocks=decoder_blocks,
                base_channels=base_channels,
                timestep_conditioning=timestep_cond,
                compute_dtype=compute_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported VAE decoder backend: {vae_decoder_backend!r}. "
                f"Only 'native' is supported."
            )
        if video_vae_load_path and not use_placeholder:
            load_native_vae_decoder_weights(decoder, video_vae_load_path)
        elif use_placeholder:
            print("  Skipping weights load (placeholder)")
        return decoder

    if not skip_vae:
        if defer_av_decoder_load:
            print("\n[3/5] VAE decoder load deferred until video decode")
            vae_decoder_loader = load_video_vae_decoder
        else:
            print("\n[3/5] Loading VAE decoder...")
            vae_decoder = load_video_vae_decoder()
    else:
        print("\n[3/5] VAE decoder skipped by user")
    timings.mark("vae decoder setup" if defer_av_decoder_load else "vae decoder load")

    # === TWO-STAGE PIPELINE ===
    # Use dedicated two-stage pipeline for higher quality generation
    if pipeline_type == "two-stage":
        # Validate requirements
        if model is None:
            if use_placeholder:
                print("  Creating dummy model for placeholder Two-Stage pipeline...")
                class MockModel:
                    def __init__(self):
                        self.velocity_model = self
                    def parameters(self):
                        return {}
                    def __call__(self, *args, **kwargs):
                        return mx.zeros(1)
                    def load_weights(self, *args):
                        pass
                model = MockModel()
            else:
                raise ValueError("Two-stage pipeline requires a loaded model (cannot use placeholder mode)")

        if vae_decoder is None and not use_placeholder:
            raise ValueError("Two-stage pipeline requires VAE decoder")

        if not spatial_upscaler_weights and not use_placeholder:
            raise ValueError("Two-stage pipeline requires --spatial-upscaler-weights")

        # Two-stage pipeline requires resolution divisible by 64 (for stage 1 half-res to be divisible by 32)
        if height % 64 != 0 or width % 64 != 0:
            new_height = ((height + 63) // 64) * 64
            new_width = ((width + 63) // 64) * 64
            print("  WARNING: Two-stage pipeline requires resolution divisible by 64.")
            print(f"  Adjusting resolution from {height}x{width} to {new_height}x{new_width}")
            height = new_height
            width = new_width

        print("\n=== Using Two-Stage Pipeline ===")
        print(f"  Stage 1: {steps_stage1} steps at {height//2}x{width//2} with CFG {cfg_stage1 or cfg_scale}")
        if guidance_rescale > 0:
            print(f"  Guidance rescale: {guidance_rescale}")
        print(f"  Stage 2: 3 steps at {height}x{width} (distilled refinement)")
        if generate_audio:
            print("  Audio generation: ENABLED")

        # Load spatial upscaler
        print("\n[3.5/5] Loading spatial upscaler...")
        spatial_upscaler = SpatialUpscaler()
        if not use_placeholder:
            load_spatial_upscaler_weights(spatial_upscaler, spatial_upscaler_weights)
        else:
            print("  Skipping weights load (placeholder)")

        print("  Spatial upscaler weight stats:")
        print(
            "    initial_conv: "
            f"mean={float(mx.mean(spatial_upscaler.initial_conv_weight)):.6f}, "
            f"std={float(mx.std(spatial_upscaler.initial_conv_weight.astype(mx.float32))):.6f}"
        )
        print(
            "    final_conv:   "
            f"mean={float(mx.mean(spatial_upscaler.final_conv_weight)):.6f}, "
            f"std={float(mx.std(spatial_upscaler.final_conv_weight.astype(mx.float32))):.6f}"
        )

        # Load video encoder
        print("[3.5/5] Loading VAE encoder...")
        video_encoder = NativeConv3dVideoEncoder(compute_dtype=compute_dtype)
        if not use_placeholder:
            load_native_vae_encoder_weights(video_encoder, video_vae_load_path)
        else:
            print("  Skipping weights load (placeholder)")

        # Load audio VAE and vocoder if audio generation is enabled.
        audio_decoder = None
        vocoder = None
        audio_sample_rate = 24000
        if generate_audio:
            print("  Loading Audio VAE decoder...")
            audio_decoder = AudioDecoder(compute_dtype=compute_dtype)
            if audio_vae_load_path:
                load_audio_decoder_weights(audio_decoder, audio_vae_load_path)

            print("  Loading Vocoder...")
            is_bwe = False
            if vocoder_load_path:
                vocoder, is_bwe = create_vocoder_for_checkpoint(config_weights_path, compute_dtype)
                if is_bwe:
                    print("  Detected BWE vocoder (LTX-2.3)")
                    load_vocoder_with_bwe_weights(vocoder, vocoder_load_path)
                else:
                    load_vocoder_weights(vocoder, vocoder_load_path)
            else:
                vocoder = Vocoder(compute_dtype=compute_dtype)
            print_audio_dtype_summary(compute_dtype, is_bwe)
            audio_sample_rate = vocoder.output_sample_rate if vocoder else 24000

        # Create two-stage pipeline
        print("\n[4/5] Creating two-stage pipeline...")
        pipeline = TwoStagePipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            spatial_upscaler=spatial_upscaler,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
        )

        # Create distilled LoRA config if provided
        distilled_lora_config = None
        if distilled_lora and not stage_scoped_loras_active:
            print(f"  Distilled LoRA: {distilled_lora} (scale {distilled_lora_scale})")
            distilled_lora_config = LoRAConfig(path=distilled_lora, strength=distilled_lora_scale)
        elif pipeline_type == "two-stage" and not stage_scoped_loras_active:
             print("  WARNING: No distilled LoRA provided for two-stage pipeline. Stage 2 quality may be degraded.")

        # Create config
        config = TwoStageCFGConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=output_fps,
            num_inference_steps=steps_stage1,
            cfg_scale=cfg_stage1 if cfg_stage1 is not None else cfg_scale,
            guidance_rescale=guidance_rescale,
            dtype=compute_dtype,
            distilled_lora_config=distilled_lora_config,
            stage_lora_configs=stage_managed_lora_configs,
            stage2_lora_fuse_mode=stage2_lora_fuse_mode,
            tiling_config=vae_tiling_config,
            audio_enabled=generate_audio,
        )

        # Create image conditionings if provided
        images = []
        if image_path:
            print(f"  Image conditioning: {image_path} (strength={image_strength})")
            images = [ImageCondition(
                image_path=image_path,
                frame_index=0,
                strength=image_strength,
            )]

        # Run pipeline
        print("\n[5/5] Running two-stage generation...")
        video, audio_waveform = pipeline(
            positive_encoding=text_encoding,
            negative_encoding=null_encoding,
            config=config,
            images=images,
            positive_audio_encoding=text_audio_encoding,
            negative_audio_encoding=null_audio_encoding,
        )

        # Convert decoded video to per-frame list for encode_video_ffmpeg
        # decode_latent returns (T, H, W, C) in uint8, so just convert to numpy list
        video_np = np.array(video)  # (T, H, W, C)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        if audio_waveform is not None:
            print(f"  Generated audio: {audio_waveform.shape}")

        # Save video
        print(f"\nSaving video to {output_path}...")
        final_path = encode_video_dispatch(
            frames, output_path,
            tier=output_tier, fps=output_fps,
            audio_waveform=audio_waveform if audio_waveform is not None else None,
            audio_sample_rate=audio_sample_rate if audio_waveform is not None else None,
            save_audio_sidecar=save_audio_sidecar,
            audio_onset_trim_mode=audio_onset_trim_mode,
            audio_onset_trim_ms=audio_onset_trim_ms,
            output_backend=output_backend,
            vsr_spatial_mode=vsr_spatial_mode,
            vsr_target_fps=vsr_target_fps,
            vsr_temporal_mode=vsr_temporal_mode,
            vsr_save_original=vsr_save_original,
            vsr_encode_quality=vsr_encode_quality,
            vsr_audio_codec=vsr_audio_codec,
        )
        print(f"Done! Video saved to {final_path}")
        return

    # === IC-LORA PIPELINE ===
    # Use ICLoraPipeline for video-to-video or image-to-video generation with control signals
    if pipeline_type == "ic-lora":
        if not control_video and not image_path:
            raise ValueError("IC-LoRA pipeline requires --control-video or --image argument")

        print("\n=== Using IC-LoRA Pipeline ===")
        if control_video:
            print(f"  Control video: {control_video}")
            print(f"  Control type: {control_type}")
            print(f"  Control strength: {control_strength}")
        if image_path:
            print(f"  Image conditioning: {image_path} (strength={image_strength})")

        if model is None:
            if use_placeholder:
                print("  IC-LoRA requires model - cannot use placeholder mode")
                return
            raise ValueError("IC-LoRA pipeline requires a loaded model")

        if vae_decoder is None and not use_placeholder:
            raise ValueError("IC-LoRA pipeline requires VAE decoder")

        # Load VAE encoder
        print("[3.5/5] Loading VAE encoder...")
        video_encoder = NativeConv3dVideoEncoder(compute_dtype=compute_dtype)
        if video_vae_load_path and not use_placeholder:
            load_native_vae_encoder_weights(video_encoder, video_vae_load_path)
        else:
            print("  Skipping weights load (placeholder)")

        # Load spatial upscaler
        print("[3.6/5] Loading spatial upscaler...")
        spatial_upscaler = SpatialUpscaler()
        upscaler_path = spatial_upscaler_weights or "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        if os.path.exists(upscaler_path):
            load_spatial_upscaler_weights(spatial_upscaler, upscaler_path)
        else:
            print(f"  Warning: Spatial upscaler weights not found at {upscaler_path}")

        # Capture cache metadata for restoring the base transformer after IC-LoRA.
        transformer_cache_restore_state = get_transformer_cache_restore_state(model)

        # Prepare LoRA configs if provided
        lora_configs = None
        if ic_lora_weights:
            print(f"  IC-LoRA weights: {ic_lora_weights}")
            lora_configs = [LoRAConfig(path=ic_lora_weights, strength=1.0)]

        # Create IC-LoRA pipeline
        print("\n[4/5] Creating IC-LoRA pipeline...")
        ic_pipeline = ICLoraPipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            spatial_upscaler=spatial_upscaler,
            transformer_cache_restore_state=transformer_cache_restore_state,
            lora_configs=lora_configs,
        )

        # Create config
        config = ICLoraConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=output_fps,
            stage_1_steps=num_steps,
            tiling_config=vae_tiling_config,
            dtype=compute_dtype,
        )

        # Create video conditioning if control video provided
        video_conditioning = []
        if control_video:
            ctrl_type = ControlType.CANNY if control_type == "canny" else ControlType.RAW
            video_cond = VideoCondition(
                video_path=control_video,
                strength=control_strength,
                control_type=ctrl_type,
                canny_low=canny_low,
                canny_high=canny_high,
                save_control=save_control,
            )
            video_conditioning = [video_cond]

        # Create image conditioning if provided (IC-LoRA supports both video and image conditioning)
        images = []
        if image_path:
            images = [ImageCondition(
                image_path=image_path,
                frame_index=0,
                strength=image_strength,
            )]

        # Run pipeline
        print("\n[5/5] Running IC-LoRA generation...")
        video = ic_pipeline(
            text_encoding=text_encoding,
            text_mask=mx.ones((1, text_encoding.shape[1]), dtype=mx.int32),
            config=config,
            images=images,
            video_conditioning=video_conditioning,
        )

        # Convert to frames
        video_np = np.array(video)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        # Save video
        print(f"\nSaving video to {output_path}...")
        final_path = encode_video_dispatch(
            frames, output_path,
            tier=output_tier, fps=output_fps,
            output_backend=output_backend,
            vsr_spatial_mode=vsr_spatial_mode,
            vsr_target_fps=vsr_target_fps,
            vsr_temporal_mode=vsr_temporal_mode,
            vsr_save_original=vsr_save_original,
            vsr_encode_quality=vsr_encode_quality,
            vsr_audio_codec=vsr_audio_codec,
        )
        print(f"Done! Video saved to {final_path}")
        return

    # === KEYFRAME INTERPOLATION PIPELINE ===
    # Use KeyframeInterpolationPipeline for interpolating between keyframe images
    if pipeline_type == "keyframe-interpolation":
        if not keyframes:
            raise ValueError("Keyframe interpolation pipeline requires --keyframe arguments")

        print("\n=== Using Keyframe Interpolation Pipeline ===")

        # Parse keyframes
        parsed_keyframes = []
        for kf_str in keyframes:
            parts = kf_str.split(":")
            if len(parts) < 2:
                raise ValueError(f"Invalid keyframe format: {kf_str}. Use 'path:frame_index' or 'path:frame_index:strength'")
            path = parts[0]
            frame_idx = int(parts[1])
            strength = float(parts[2]) if len(parts) > 2 else 0.95
            parsed_keyframes.append(Keyframe(image_path=path, frame_index=frame_idx, strength=strength))
            print(f"  Keyframe: {path} at frame {frame_idx} (strength={strength})")

        if model is None:
            if use_placeholder:
                print("  Keyframe interpolation requires model - cannot use placeholder mode")
                return
            raise ValueError("Keyframe interpolation pipeline requires a loaded model")

        if vae_decoder is None and not use_placeholder:
            raise ValueError("Keyframe interpolation pipeline requires VAE decoder")

        # Load VAE encoder
        print("[3.5/5] Loading VAE encoder...")
        video_encoder = NativeConv3dVideoEncoder(compute_dtype=compute_dtype)
        if video_vae_load_path and not use_placeholder:
            load_native_vae_encoder_weights(video_encoder, video_vae_load_path)
        else:
            print("  Skipping weights load (placeholder)")

        # Load spatial upscaler for two-stage
        print("[3.6/5] Loading spatial upscaler...")
        spatial_upscaler = SpatialUpscaler()
        upscaler_path = spatial_upscaler_weights or "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        if os.path.exists(upscaler_path):
            load_spatial_upscaler_weights(spatial_upscaler, upscaler_path)
        else:
            print(f"  Warning: Spatial upscaler weights not found at {upscaler_path}")

        # Create keyframe interpolation pipeline
        print("\n[4/5] Creating keyframe interpolation pipeline...")
        kf_pipeline = KeyframeInterpolationPipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            spatial_upscaler=spatial_upscaler,
        )

        # Create config
        config = KeyframeInterpolationConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=output_fps,
            num_inference_steps=num_steps,
            cfg_scale=cfg_scale,
            tiling_config=vae_tiling_config,
            dtype=compute_dtype,
        )

        # Run pipeline
        print(f"\n[5/5] Running keyframe interpolation ({num_steps} steps)...")
        video = kf_pipeline(
            text_encoding=text_encoding,
            text_mask=mx.ones((1, text_encoding.shape[1]), dtype=mx.int32),
            keyframes=parsed_keyframes,
            config=config,
            negative_text_encoding=null_encoding,
            negative_text_mask=mx.ones((1, null_encoding.shape[1]), dtype=mx.int32),
        )

        # Convert to frames
        video_np = np.array(video)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        # Save video
        print(f"\nSaving video to {output_path}...")
        final_path = encode_video_dispatch(
            frames, output_path,
            tier=output_tier, fps=output_fps,
            output_backend=output_backend,
            vsr_spatial_mode=vsr_spatial_mode,
            vsr_target_fps=vsr_target_fps,
            vsr_temporal_mode=vsr_temporal_mode,
            vsr_save_original=vsr_save_original,
            vsr_encode_quality=vsr_encode_quality,
            vsr_audio_codec=vsr_audio_codec,
        )
        print(f"Done! Video saved to {final_path}")
        return

    # === AUDIO-VIDEO PIPELINE ===
    # Use AVPipeline for joint audio-video generation
    # V2.3 always uses this path (AV transformer) even without audio generation
    if use_av_encoder:
        distilled_two_stage = distilled_two_stage_requested
        print("\n=== Using Audio-Video Pipeline ===")
        if distilled_two_stage:
            print("  Mode: distilled two-stage")
        if save_latents:
            print(f"  Latent sidecar: {latent_sidecar_path(output_path)}")

        if model is None:
            if use_placeholder:
                print("  AV pipeline requires model - cannot use placeholder mode")
                return
            raise ValueError("AV pipeline requires a loaded AudioVideo model")

        if vae_decoder is None and vae_decoder_loader is None and not use_placeholder:
            raise ValueError("AV pipeline requires VAE decoder")

        # Create image conditionings if provided. The VAE encoder is needed only
        # for image conditioning, so text-only/video-audio runs can skip it.
        images = []
        if image_path:
            print(f"  Image conditioning: {image_path} (strength={image_strength})")
            images = [ImageCondition(
                image_path=image_path,
                frame_index=0,
                strength=image_strength,
            )]

        video_encoder = None
        if images:
            print("[3.5/5] Loading VAE encoder...")
            video_encoder = NativeConv3dVideoEncoder(compute_dtype=compute_dtype)
            if video_vae_load_path and not use_placeholder:
                load_native_vae_encoder_weights(video_encoder, video_vae_load_path)
            else:
                print("  Skipping weights load (placeholder)")
        elif distilled_two_stage:
            print("[3.5/5] Loading VAE encoder statistics...")
            if video_vae_load_path and not use_placeholder:
                video_encoder = load_native_vae_encoder_statistics(video_vae_load_path)
            else:
                video_encoder = NativeConv3dVideoEncoderStatistics()
                video_encoder.per_channel_statistics.std_of_means = mx.ones(
                    (128,),
                    dtype=mx.float32,
                )
                print("  Using placeholder VAE encoder statistics")
        else:
            print("[3.5/5] VAE encoder skipped (no image conditioning)")

        spatial_upscaler = None
        spatial_upscaler_loader = None
        if distilled_two_stage:
            if model_variant != "distilled":
                raise ValueError("Distilled two-stage requires --model-variant distilled")
            if not spatial_upscaler_weights:
                raise ValueError("Distilled two-stage requires --spatial-upscaler-weights")
            print("[3.6/5] Spatial upscaler load deferred until inter-stage upscale")

            def spatial_upscaler_loader():
                upscaler = SpatialUpscaler()
                if not use_placeholder:
                    load_spatial_upscaler_weights(upscaler, spatial_upscaler_weights)
                else:
                    print("  Skipping weights load (placeholder)")
                return upscaler

        audio_decoder = None
        vocoder = None
        audio_decoder_loader = None
        audio_sample_rate = 24000

        def load_audio_decode_stack():
            print("  Loading Audio VAE decoder...")
            loaded_audio_decoder = AudioDecoder(compute_dtype=compute_dtype)
            if audio_vae_load_path:
                load_audio_decoder_weights(loaded_audio_decoder, audio_vae_load_path)

            print("  Loading Vocoder...")
            is_bwe = False
            if vocoder_load_path:
                loaded_vocoder, is_bwe = create_vocoder_for_checkpoint(config_weights_path, compute_dtype)
                if is_bwe:
                    print("  Detected BWE vocoder (LTX-2.3)")
                    load_vocoder_with_bwe_weights(loaded_vocoder, vocoder_load_path)
                else:
                    load_vocoder_weights(loaded_vocoder, vocoder_load_path)
            else:
                loaded_vocoder = Vocoder(compute_dtype=compute_dtype)
            print_audio_dtype_summary(compute_dtype, is_bwe)
            loaded_sample_rate = loaded_vocoder.output_sample_rate if loaded_vocoder else 24000
            return loaded_audio_decoder, loaded_vocoder, loaded_sample_rate

        if generate_audio:
            if distilled_two_stage:
                print("  Audio decoder/vocoder load deferred until audio decode")
                audio_decoder_loader = load_audio_decode_stack
            else:
                audio_decoder, vocoder, audio_sample_rate = load_audio_decode_stack()

        # Create one-stage pipeline with audio support
        print("\n[4/5] Creating audio-video pipeline...")
        av_pipeline = AVPipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
            video_decoder_loader=vae_decoder_loader,
            audio_decoder_loader=audio_decoder_loader,
            audio_sample_rate=audio_sample_rate,
        )

        del model  # pipeline now holds the only reference; del self.transformer in av_pipeline.py can actually free it
        model = None
        gc.collect()

        # Create config with audio enabled
        # LTX-2.3 reference: video_cfg=3.0, audio_cfg=7.0, rescale=0.7
        #
        # Internal-audio resolution.  By default V2/AV models always run the
        # internal audio branch (audio self-attn + A2V/V2A cross-modal) even
        # when --generate-audio is off, and discard the result - wasted compute.
        # --internal-audio gives users control:
        #   auto (default): on iff --generate-audio
        #   on            : always on
        #   off           : always off (also disables audio output)
        # LTX_DISABLE_INTERNAL_AUDIO=1 is a legacy env override that forces off.
        _env_disable_internal_audio = bool(os.environ.get("LTX_DISABLE_INTERNAL_AUDIO"))
        if _env_disable_internal_audio:
            _internal_audio_resolved = False
        elif internal_audio == "on":
            _internal_audio_resolved = True
        elif internal_audio == "off":
            _internal_audio_resolved = False
        else:  # "auto"
            _internal_audio_resolved = generate_audio
        _disable_internal_audio = not _internal_audio_resolved
        av_config = AVCFGConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=output_fps,
            num_inference_steps=num_steps,
            use_distilled_sigmas=model_variant == "distilled",
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale if audio_cfg_scale is not None else (1.0 if model_variant == "distilled" else 7.0),
            rescale_scale=rescale_scale if rescale_scale is not None else (0.0 if model_variant == "distilled" else 0.7),
            dtype=compute_dtype,
            tiling_config=vae_tiling_config,
            auto_tiling=False,  # Already resolved by build_vae_tiling_config().
            profile_transformer_steps=active_profile_steps,
            profile_transformer_blocks=active_profile_blocks,
            audio_enabled=generate_audio,
            use_internal_audio_branch=not _disable_internal_audio,
        )

        timings.mark("av pipeline prep")

        # Run pipeline with audio
        if distilled_two_stage:
            print(
                f"\n[5/5] Running distilled two-stage generation "
                f"({len(DISTILLED_SIGMA_VALUES) - 1}+"
                f"{len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1} steps)..."
            )

            stage_labels = {
                "stage_1": "Stage 1 denoising",
                "stage_2": "Stage 2 denoising",
            }

            # Resolve the backend up front so the pipeline can skip the
            # internal VAE decode + concat when we're going to stream
            # chunks straight into the AVAssetWriter sink.  Eager (ffmpeg)
            # path still gets the fully decoded video tensor it expects.
            _resolved_backend = resolve_output_backend(
                output_backend, output_tier,
                vsr_spatial_mode=vsr_spatial_mode,
                vsr_target_fps=vsr_target_fps,
            )
            _stream_decode = _resolved_backend == "videotoolbox"

            # Stage 1 and Stage 2 share a single StackedPhaseBars so
            # their columns (label / count / STEP1 / RUN / ETA / pace)
            # line up; each stage's PhaseBar is lazily added on first
            # callback for that stage.  set_n(step) lets us forward
            # the pipeline's absolute step number directly.
            with StackedPhaseBars() as denoise_bars:
                stage_bars: dict[str, PhaseBar] = {}

                def stage_progress_callback(stage_name: str, step: int, total: int):
                    if stage_name not in stage_bars:
                        stage_bars[stage_name] = denoise_bars.add(
                            total=total,
                            desc=stage_labels.get(stage_name, stage_name),
                            unit="step",
                            show_step1=True,
                        )
                    stage_bars[stage_name].set_n(step)

                def distilled_progress_message(message: str):
                    # Inter-stage messages ("Upsampling latent 2x...",
                    # "Distilled stage 2: ...") land BELOW the finished
                    # Stage 1 bar; Stage 2's bar then slots in below
                    # them when its first callback fires.  Visual order
                    # ends up: Stage 1 bar / messages / Stage 2 bar.
                    denoise_bars.write(message, position="below")

                video, audio_waveform = av_pipeline.generate_distilled_two_stage(
                    positive_encoding=text_encoding,
                    config=av_config,
                    spatial_upscaler=spatial_upscaler,
                    images=images,
                    stage_callback=stage_progress_callback,
                    progress_message=distilled_progress_message,
                    positive_audio_encoding=text_audio_encoding,
                    latent_save_path=latent_sidecar_path(output_path) if save_latents else None,
                    decode_video=not _stream_decode,
                    stage_lora_configs=stage_managed_lora_configs,
                    stage2_lora_fuse_mode=stage2_lora_fuse_mode,
                    spatial_upscaler_loader=spatial_upscaler_loader,
                )
        else:
            print(f"\n[5/5] Running audio-video generation ({num_steps} steps)...")

            _resolved_backend = resolve_output_backend(
                output_backend, output_tier,
                vsr_spatial_mode=vsr_spatial_mode,
                vsr_target_fps=vsr_target_fps,
            )
            _stream_decode = _resolved_backend == "videotoolbox"

            with StackedPhaseBars() as denoise_bars:
                denoise_bar = denoise_bars.add(
                    total=num_steps,
                    desc="Denoising",
                    unit="step",
                    show_step1=True,
                )

                def progress_callback(step: int, total: int):
                    denoise_bar.set_n(step)

                def progress_message(message: str) -> None:
                    # Pipeline status lines ("Saved final latents:",
                    # "VAE decode started", etc.) land BELOW the
                    # denoise bar via position="below" - bar stays
                    # frozen at its row, messages stack permanently
                    # under it as scrollback.
                    denoise_bars.write(message, position="below")

                video, audio_waveform = av_pipeline(
                    positive_encoding=text_encoding,
                    negative_encoding=null_encoding,
                    config=av_config,
                    images=images,
                    callback=progress_callback,
                    progress_message=progress_message,
                    positive_audio_encoding=text_audio_encoding,
                    negative_audio_encoding=null_audio_encoding,
                    latent_save_path=latent_sidecar_path(output_path) if save_latents else None,
                    decode_video=not _stream_decode,
                )
        pipeline_timings = getattr(av_pipeline, "last_timing_sections", None)
        if pipeline_timings:
            timings.extend(pipeline_timings)
        else:
            timings.mark("generation + decode")
        audio_sample_rate = getattr(av_pipeline, "audio_sample_rate", audio_sample_rate)

        if _stream_decode:
            # Streaming path: `video` is the final_video_latent.  Build a
            # chunk-by-chunk frame iterator that runs VAE decode on demand
            # and hand it to the AVAssetWriter sink; no full-decoded video
            # tensor is ever materialized in unified memory.
            #
            # The bar stack is owned here (not by the encoder) so we can
            # stack a "VAE chunks" bar above the encoder's "VT encode"
            # frames bar - same UX as scripts/vsr_harness.py.
            from LTX_2_MLX.pipelines.streaming import (
                iter_decoded_chunks,
                latent_dims,
                plan_vae_tiling,
            )

            final_video_latent = video
            effective_tiling = av_config._get_tiling_config()
            n_total_frames, latent_h, latent_w = latent_dims(final_video_latent)
            n_vae_chunks, tiling_desc = plan_vae_tiling(
                final_video_latent, effective_tiling,
            )
            stream_decoder_loaded_now = False
            if av_pipeline.video_decoder is None:
                if vae_decoder_loader is None:
                    raise ValueError("Streaming VAE decode requires a VAE decoder loader")
                print("  Loading VAE decoder for streaming decode...")
                decoder_load_start = time.perf_counter()
                av_pipeline.video_decoder = vae_decoder_loader()
                decoder_load_elapsed = time.perf_counter() - decoder_load_start
                print(f"  VAE decoder load complete in {decoder_load_elapsed:.1f}s")
                timings.sections.append(("vae decoder load", decoder_load_elapsed))
                timings.last_mark = time.perf_counter()
                stream_decoder_loaded_now = True
            decoder_name = av_pipeline.video_decoder.__class__.__name__
            print(
                f"  Streaming VAE decode -> encoder: latent shape "
                f"{tuple(final_video_latent.shape)} -> {latent_w}x{latent_h}, "
                f"{n_total_frames} frames"
            )
            print(
                f"  VAE decoder: {decoder_name} ({tiling_desc}, "
                f"{n_vae_chunks} chunk{'s' if n_vae_chunks != 1 else ''})"
            )
            if audio_waveform is not None:
                print(f"  Generated audio: {audio_waveform.shape}")
            timings.mark("post-pipeline prep")

            print(f"\nSaving video to {output_path}...")

            with StackedPhaseBars() as _stream_bars:
                vae_pbar = _stream_bars.add(
                    total=n_vae_chunks,
                    desc="VAE chunks",
                    unit="chunk",
                )

                def _chunk_aware_frames():
                    """Yield per-frame ndarrays AND tick the VAE chunks
                    bar on each chunk boundary.  Lets the user watch
                    VAE chunk progress in lockstep with VT encode frame
                    progress (the encoder adds its own bar to the same
                    stack via the progress_stack kwarg below).
                    """
                    for chunk_frames in iter_decoded_chunks(
                        final_video_latent, av_pipeline.video_decoder,
                        tiling=effective_tiling,
                        output_format="fp16_rgba",
                    ):
                        vae_pbar.update(1)
                        while chunk_frames:
                            yield chunk_frames.pop(0)

                final_path = encode_video_dispatch(
                    _chunk_aware_frames(), output_path,
                    tier=output_tier, fps=output_fps,
                    audio_waveform=audio_waveform if audio_waveform is not None else None,
                    audio_sample_rate=audio_sample_rate if audio_waveform is not None else None,
                    save_audio_sidecar=save_audio_sidecar,
                    audio_onset_trim_mode=audio_onset_trim_mode,
                    audio_onset_trim_ms=audio_onset_trim_ms,
                    output_backend=output_backend,
                    vsr_spatial_mode=vsr_spatial_mode,
                    vsr_target_fps=vsr_target_fps,
                    vsr_temporal_mode=vsr_temporal_mode,
                    vsr_save_original=vsr_save_original,
                    vsr_encode_quality=vsr_encode_quality,
                    vsr_audio_codec=vsr_audio_codec,
                    n_source_frames=n_total_frames,
                    progress_stack=_stream_bars,
                )
            print(f"Done! Video saved to {final_path}")
            if stream_decoder_loaded_now:
                av_pipeline.video_decoder = None
                gc.collect()
                mx.clear_cache()
            timings.mark("output save")
        else:
            # Eager path: pipeline produced a fully decoded video tensor.
            # Convert to per-frame uint8 list for the ffmpeg encoder.
            video_np = np.array(video)
            print(f"  Raw video shape: {video_np.shape}, dtype: {video_np.dtype}")
            # Squeeze any singleton dimensions
            video_np = np.squeeze(video_np)
            # Handle (C, T, H, W) format - C=3 is always smallest dim
            if video_np.ndim == 4 and video_np.shape[0] == 3:
                video_np = np.transpose(video_np, (1, 2, 3, 0))  # (T, H, W, C)
            # Convert float32 [-1,1] to uint8 [0,255] (VAE output range is [-1, 1])
            if video_np.dtype != np.uint8:
                video_np = np.clip((video_np + 1) / 2 * 255.0, 0, 255).astype(np.uint8)
            frames = [video_np[t] for t in range(video_np.shape[0])]
            print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

            if audio_waveform is not None:
                print(f"  Generated audio: {audio_waveform.shape}")
            timings.mark("frame conversion")

            # Save video with audio
            print(f"\nSaving video to {output_path}...")
            final_path = encode_video_dispatch(
                frames, output_path,
                tier=output_tier, fps=output_fps,
                audio_waveform=audio_waveform if audio_waveform is not None else None,
                audio_sample_rate=audio_sample_rate if audio_waveform is not None else None,
                save_audio_sidecar=save_audio_sidecar,
                audio_onset_trim_mode=audio_onset_trim_mode,
                audio_onset_trim_ms=audio_onset_trim_ms,
                output_backend=output_backend,
                vsr_spatial_mode=vsr_spatial_mode,
                vsr_target_fps=vsr_target_fps,
                vsr_temporal_mode=vsr_temporal_mode,
                vsr_save_original=vsr_save_original,
                vsr_encode_quality=vsr_encode_quality,
                vsr_audio_codec=vsr_audio_codec,
            )
            print(f"Done! Video saved to {final_path}")
            timings.mark("output save")
        if save_run_log and run_metadata is not None:
            save_run_log_sidecar(
                run_log_sidecar_path(output_path),
                run_metadata,
                timings,
                status="completed",
                outputs={
                    "video": output_path,
                    "audio_wav": os.path.splitext(output_path)[0] + ".wav" if audio_waveform is not None else None,
                    "latents": latent_sidecar_path(output_path) if save_latents else None,
                    "text_conditioning": text_sidecar_path(output_path) if save_text_embeddings else None,
                },
            )
            timings.mark("run log save")
        timings.print_summary()
        return

    # === STANDARD PIPELINE (non-AV fallback, video-only) ===
    # Note: LTX-2.3 distilled one-stage and two-stage modes use the AV path above.
    if save_latents:
        print("  WARNING: --save-latents is currently supported only on the Audio-Video pipeline path")

    # Initialize noise
    print("\n[4/5] Initializing latent noise...")
    latent = mx.random.normal(shape=(1, 128, latent_frames, latent_height, latent_width))

    # Get sigma schedule based on model variant
    # The distilled model was trained with specific sigma values
    use_linear_schedule = False  # Use distilled values for distilled model
    if use_linear_schedule:
        # Linear schedule: evenly spaced from 1.0 to 0.0
        # Better for spatial coherence preservation
        sigmas = mx.linspace(1.0, 0.0, num_steps + 1)
        print(f"  Sigma schedule (linear): {[f'{float(s):.3f}' for s in sigmas]}")
    elif model_variant == "distilled":
        sigmas = mx.array(DISTILLED_SIGMA_VALUES[:num_steps + 1])
        print(f"  Sigma schedule (distilled): {[f'{float(s):.3f}' for s in sigmas]}")
    else:
        # Dev model uses LTX2Scheduler for dynamic schedule
        sigmas = get_sigma_schedule(num_steps=num_steps, distilled=False, latent=latent)
        print(f"  Sigma schedule (dev): {[f'{float(s):.3f}' for s in sigmas]}")

    # Create patchifier
    patchifier = VideoLatentPatchifier(patch_size=1)

    print(f"\n[5/5] Denoising ({num_steps} steps)...")

    # Denoising loop with progress bar
    step_iterator = progress_bar(range(len(sigmas) - 1), desc="Denoising", total=num_steps)

    # GE (Gradient Estimation) velocity tracking
    prev_velocity = None

    for i in step_iterator:
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        if model is not None and not use_placeholder:
            # === Actual model inference ===
            # Patchify video latent: [B, C, F, H, W] -> [B, T, C]
            latent_patchified = patchifier.patchify(latent)

            # Create video position grid with proper pixel-space coordinates
            # (matching the distilled pipeline format)
            output_shape = VideoLatentShape(
                batch=1,
                channels=128,
                frames=latent_frames,
                height=latent_height,
                width=latent_width,
            )
            latent_coords = patchifier.get_patch_grid_bounds(output_shape=output_shape)
            scale_factors = SpatioTemporalScaleFactors.default()  # time=8, height=32, width=32
            positions = get_pixel_coords(
                latent_coords=latent_coords,
                scale_factors=scale_factors,
                causal_fix=True,
            ).astype(mx.float32)
            # Convert temporal positions from frames to seconds
            fps = output_fps
            temporal_positions = positions[:, 0:1, ...] / fps
            other_positions = positions[:, 1:, ...]
            positions = mx.concatenate([temporal_positions, other_positions], axis=1)

            # === Video-only mode with X0 prediction ===
            # Note: Audio mode is handled by the AUDIO-VIDEO PIPELINE section above
            # The distilled LTX-2 model directly outputs X0 (denoised samples),
            # NOT velocity. So we:
            # 1. Get X0 directly from model
            # 2. Apply CFG on X0 samples
            # 3. Euler step with X0
            # (output_shape already defined above for position creation)

            # Apply CFG if enabled
            if use_cfg:
                if low_memory:
                    # MEMORY OPTIMIZATION: Sequential CFG passes
                    # Run unconditional first, eval, then conditional

                    # Unconditional (null text) pass first
                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_uncond = Modality(
                        latent=latent_patchified,
                        context=null_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_uncond_patchified = model(modality_uncond)
                    denoised_uncond = patchifier.unpatchify(x0_uncond_patchified, output_shape=output_shape)
                    mx.eval(denoised_uncond)
                    del x0_uncond_patchified

                    # Conditional (text-guided) pass
                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_cond = Modality(
                        latent=latent_patchified,
                        context=text_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_cond_patchified = model(modality_cond)
                    denoised_cond = patchifier.unpatchify(x0_cond_patchified, output_shape=output_shape)
                    mx.eval(denoised_cond)
                    del x0_cond_patchified

                    # Apply guidance: APG if enabled, otherwise standard CFG
                    if apg_guider is not None and apg_guider.enabled():
                        denoised = apg_guider.guide(denoised_cond, denoised_uncond)
                    else:
                        # CFG formula on X0: x0 = x0_uncond + scale * (x0_cond - x0_uncond)
                        denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

                    # Apply guidance rescale to prevent variance explosion
                    if guidance_rescale > 0:
                        denoised = rescale_noise_cfg(denoised, denoised_cond, guidance_rescale)

                    # Apply STG (Spatio-Temporal Guidance) if enabled
                    if stg_guider is not None and stg_guider.enabled():
                        # Run perturbed forward pass (skip video self-attention)
                        x0_perturbed_patchified = model(modality_cond, perturbations=create_batched_stg_config(batch_size=1))
                        denoised_perturbed = patchifier.unpatchify(x0_perturbed_patchified, output_shape=output_shape)
                        denoised = stg_guider.guide(denoised, denoised_perturbed)
                        del denoised_perturbed, x0_perturbed_patchified

                    del denoised_uncond, denoised_cond
                else:
                    # Standard CFG: Sequential forward passes
                    # NOTE: Batched CFG (stacking cond+uncond) was tested but found SLOWER
                    # for 19B models because GPU is already fully utilized with batch=1.
                    # Doubling batch just doubles compute time with no throughput gain.
                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_cond = Modality(
                        latent=latent_patchified,
                        context=text_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_cond_patchified = model(modality_cond)
                    denoised_cond = patchifier.unpatchify(x0_cond_patchified, output_shape=output_shape)

                    # NOTE: context_mask=None matches PyTorch behavior
                    modality_uncond = Modality(
                        latent=latent_patchified,
                        context=null_encoding,
                        context_mask=None,
                        timesteps=mx.array([sigma]),
                        positions=positions,
                        enabled=True,
                    )
                    x0_uncond_patchified = model(modality_uncond)
                    denoised_uncond = patchifier.unpatchify(x0_uncond_patchified, output_shape=output_shape)

                    # Apply guidance: APG if enabled, otherwise standard CFG
                    if apg_guider is not None and apg_guider.enabled():
                        denoised = apg_guider.guide(denoised_cond, denoised_uncond)
                    else:
                        # CFG formula on X0: x0 = x0_uncond + scale * (x0_cond - x0_uncond)
                        denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

                    # Apply guidance rescale to prevent variance explosion
                    if guidance_rescale > 0:
                        denoised = rescale_noise_cfg(denoised, denoised_cond, guidance_rescale)

                    # Apply STG (Spatio-Temporal Guidance) if enabled
                    if stg_guider is not None and stg_guider.enabled():
                        # Run perturbed forward pass (skip video self-attention)
                        x0_perturbed_patchified = model(modality_cond, perturbations=create_batched_stg_config(batch_size=1))
                        denoised_perturbed = patchifier.unpatchify(x0_perturbed_patchified, output_shape=output_shape)
                        denoised = stg_guider.guide(denoised, denoised_perturbed)
            else:
                # No CFG - just conditional pass
                # NOTE: context_mask=None matches PyTorch behavior
                modality_cond = Modality(
                    latent=latent_patchified,
                    context=text_encoding,
                    context_mask=None,
                    timesteps=mx.array([sigma]),
                    positions=positions,
                    enabled=True,
                )
                x0_cond_patchified = model(modality_cond)
                denoised = patchifier.unpatchify(x0_cond_patchified, output_shape=output_shape)

                # Apply STG (Spatio-Temporal Guidance) if enabled (works without CFG)
                if stg_guider is not None and stg_guider.enabled():
                    # Run perturbed forward pass (skip video self-attention)
                    x0_perturbed_patchified = model(modality_cond, perturbations=create_batched_stg_config(batch_size=1))
                    denoised_perturbed = patchifier.unpatchify(x0_perturbed_patchified, output_shape=output_shape)
                    denoised = stg_guider.guide(denoised, denoised_perturbed)

            # Apply GE (Gradient Estimation) velocity correction if enabled
            if ge_gamma > 0:
                # Compute current velocity: v = (x - x0) / sigma
                current_velocity = (latent - denoised) / sigma

                if prev_velocity is not None:
                    # Apply velocity correction using momentum-like update
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    # Reconstruct corrected denoised: x0 = x - v * sigma
                    denoised = latent - total_velocity * sigma

                # Update velocity for next iteration
                prev_velocity = current_velocity

            # Euler step using X0 (denoised) prediction
            latent = euler_step_x0(latent, denoised, sigma, sigma_next)

            # Force evaluation for memory efficiency
            mx.eval(latent)
        else:
            # Placeholder: random velocity
            velocity = mx.random.normal(shape=latent.shape) * 0.1
            latent = euler_step(latent, velocity, sigma, sigma_next)
            mx.eval(latent)

    # === MEMORY OPTIMIZATION ===
    # Clear transformer from memory - no longer needed after denoising
    print("\n  Clearing transformer from memory...")
    del model
    gc.collect()
    mx.metal.clear_cache()

    # Save denoised latent
    latent_path = output_path.replace('.mp4', '_latent.npz')
    print(f"\nSaving denoised latent to {latent_path}...")
    sidecars.save_sidecar(latent_path, {"latent": latent})
    print(f"  Latent shape: {latent.shape}")

    # Apply spatial upscaling if requested
    if upscale_spatial and spatial_upscaler_weights:
        print("\nApplying 2x spatial upscaling...")
        print(f"  Input latent: {latent.shape}")

        # Load upscaler
        spatial_upscaler = SpatialUpscaler()
        load_spatial_upscaler_weights(spatial_upscaler, spatial_upscaler_weights)

        # CRITICAL: Un-normalize before upsampling, re-normalize after
        # The upsampler model is trained on raw (un-normalized) latents
        # Reference: PyTorch upsample_video() in ltx_core/model/upsampler/model.py
        if vae_decoder is not None:
            # Un-normalize: latent_raw = latent * std + mean
            std = vae_decoder.std_of_means.reshape(1, -1, 1, 1, 1)
            mean = vae_decoder.mean_of_means.reshape(1, -1, 1, 1, 1)
            latent_unnorm = latent * std + mean
            print(f"  Un-normalized: std={float(mx.std(latent_unnorm)):.3f}")

            # Upscale the un-normalized latent
            latent_upscaled = spatial_upscaler(latent_unnorm)
            mx.eval(latent_upscaled)

            # Re-normalize: latent = (latent_raw - mean) / std
            latent = (latent_upscaled - mean) / std
            mx.eval(latent)
            print(f"  Re-normalized: std={float(mx.std(latent)):.3f}")
        else:
            # Fallback: upscale directly (may have incorrect dynamic range)
            print("  WARNING: No VAE decoder for normalization - output may have wrong range")
            latent = spatial_upscaler(latent)
            mx.eval(latent)

        print(f"  Upscaled latent: {latent.shape}")

        # Clear upscaler from memory
        del spatial_upscaler
        gc.collect()
        mx.metal.clear_cache()

    # Apply temporal upscaling if requested
    if upscale_temporal and temporal_upscaler_weights:
        print("\nApplying 2x temporal upscaling...")
        print(f"  Input latent: {latent.shape}")

        # Load upscaler
        temporal_upscaler = TemporalUpscaler()
        load_temporal_upscaler_weights(temporal_upscaler, temporal_upscaler_weights)

        # CRITICAL: Un-normalize before upsampling, re-normalize after
        # The upsampler model is trained on raw (un-normalized) latents
        # Reference: PyTorch upsample_video() in ltx_core/model/upsampler/model.py
        if vae_decoder is not None:
            # Un-normalize: latent_raw = latent * std + mean
            std = vae_decoder.std_of_means.reshape(1, -1, 1, 1, 1)
            mean = vae_decoder.mean_of_means.reshape(1, -1, 1, 1, 1)
            latent_unnorm = latent * std + mean
            print(f"  Un-normalized: std={float(mx.std(latent_unnorm)):.3f}")

            # Upscale the un-normalized latent
            latent_upscaled = temporal_upscaler(latent_unnorm)
            mx.eval(latent_upscaled)

            # Re-normalize: latent = (latent_raw - mean) / std
            latent = (latent_upscaled - mean) / std
            mx.eval(latent)
            print(f"  Re-normalized: std={float(mx.std(latent)):.3f}")
        else:
            # Fallback: upscale directly (may have incorrect dynamic range)
            print("  WARNING: No VAE decoder for normalization - output may have wrong range")
            latent = temporal_upscaler(latent)
            mx.eval(latent)

        print(f"  Upscaled latent: {latent.shape}")

        # Clear upscaler from memory
        del temporal_upscaler
        gc.collect()
        mx.metal.clear_cache()

    # Decode with VAE or create placeholder
    if vae_decoder is not None:
        print("\nDecoding with VAE...")
        print(f"  Input latent: {latent.shape}")

        # VAE decode
        video = decode_latent(latent, vae_decoder)
        mx.eval(video)
        print(f"  Output video: {video.shape}")

        # Convert decoded video to per-frame list for encode_video_ffmpeg
        frames = [np.array(video[f]) for f in range(video.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

    else:
        print("\nCreating placeholder video (VAE not loaded)...")

        # Placeholder output - create simple visualization based on latent statistics
        frames = []
        latent_np = np.array(latent[0])  # (C, F, H, W)
        latent_mean = latent_np.mean(axis=0)  # (F, H, W)
        latent_std = latent_np.std(axis=0)

        frame_iterator = progress_bar(range(num_frames), desc="Creating frames", total=num_frames)

        for f in frame_iterator:
            # Map latent frame to visualization
            lat_f = f * latent_frames // num_frames
            lat_f = min(lat_f, latent_frames - 1)

            # Get latent statistics for this frame
            lat_slice_mean = latent_mean[lat_f]  # (H, W)
            lat_slice_std = latent_std[lat_f]

            # Create frame based on latent (upscale from latent to output resolution)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Simple bilinear-ish upscale of latent visualization
            for y in range(height):
                for x in range(width):
                    lat_y = y * latent_height // height
                    lat_x = x * latent_width // width
                    lat_y = min(lat_y, latent_height - 1)
                    lat_x = min(lat_x, latent_width - 1)

                    # Use latent values to create RGB
                    val_mean = float(lat_slice_mean[lat_y, lat_x])
                    val_std = float(lat_slice_std[lat_y, lat_x])

                    # Normalize and convert to color
                    r = int(np.clip((val_mean + 2) / 4 * 255, 0, 255))
                    g = int(np.clip((val_std) / 2 * 255, 0, 255))
                    b = int(np.clip((val_mean * val_std + 1) / 2 * 128, 0, 255))

                    frame[y, x] = [r, g, b]

            frames.append(frame)

    # Save video
    # Note: Audio generation is handled by the AUDIO-VIDEO PIPELINE section above
    print(f"\nSaving video to {output_path}...")
    encode_video_dispatch(
        frames, output_path,
        tier=output_tier, fps=output_fps,
        output_backend=output_backend,
        vsr_spatial_mode=vsr_spatial_mode,
        vsr_target_fps=vsr_target_fps,
        vsr_temporal_mode=vsr_temporal_mode,
        vsr_save_original=vsr_save_original,
        vsr_encode_quality=vsr_encode_quality,
        vsr_audio_codec=vsr_audio_codec,
    )
    timings.mark("output save")

    print(f"\nDone! Video saved to {output_path}")
    if save_run_log and run_metadata is not None:
        save_run_log_sidecar(
            run_log_sidecar_path(output_path),
            run_metadata,
            timings,
            status="completed",
            outputs={
                "video": output_path,
                "audio_wav": None,
                "latents": None,
                "text_conditioning": text_sidecar_path(output_path) if save_text_embeddings else None,
            },
        )
        timings.mark("run log save")
    timings.print_summary()

    if use_placeholder:
        print("\nNote: This is a placeholder output. Full inference requires:")
        print("  1. Proper weight loading (use --weights flag)")
        print("  2. Gemma text encoder integration")

    if vae_decoder is None and not skip_vae:
        print("\nNote: VAE decoder was not loaded - output is placeholder visualization.")


# save_video / save_video_with_audio moved to LTX_2_MLX.ffmpeg_encoder
# (encode_video_ffmpeg, called above from each pipeline). The legacy bodies took
# a `speed` multiplier and re-encoded via PNG round-tripping; both have
# been dropped - speed adjustment belongs in an editor, and the new
# encoder pipes raw frames directly into ffmpeg.


def main():
    parser = argparse.ArgumentParser(description="Generate video with LTX-2 MLX")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--height", type=int, default=288, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--frames", type=int, default=97, help="Number of frames")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds. Overrides --frames and rounds up to the next valid 8*k+1 frame count.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Denoising steps. Default is 8 for distilled and 30 for dev.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="CFG scale. Default is 1.0 for distilled and 5.0 for dev.",
    )
    parser.add_argument("--guidance-rescale", type=float, default=0.7, help="Guidance rescale factor (0.0=off, 0.7=default, 1.0=full)")
    parser.add_argument("--steps-stage1", type=int, default=15, help="Stage 1 steps for two-stage pipeline")
    parser.add_argument("--steps-stage2", type=int, default=3, help="Stage 2 refinement steps for two-stage pipeline")
    parser.add_argument("--cfg-stage1", type=float, default=None, help="Stage 1 CFG (defaults to --cfg value)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=float, default=NATIVE_FPS, help=f"Generation and output frame rate (default: {NATIVE_FPS}).")
    parser.add_argument(
        "--encode-tier",
        choices=sorted(TIERS),
        default="default",
        help=(
            "Output encode tier, picked by destination. "
            "web: universal browser/player compat (H.264 + AAC). "
            "default: everyday output on Apple / modern browsers (HEVC HW 10-bit 4:2:0 + ALAC). "
            "hq: local viewing with full chroma (HEVC SW 10-bit 4:4:4 + ALAC). "
            "export: editor / colorist hand-off (ProRes 422 HQ + PCM 24-bit, .mov). "
            "reference: canonical highest-fidelity copy (ProRes 4444 + PCM 24-bit + alpha, .mov)."
        ),
    )
    parser.add_argument(
        "--output-backend",
        choices=["auto", "ffmpeg", "videotoolbox"],
        default="auto",
        help=(
            "How to encode the final mp4.  auto (default) routes the "
            "HEVC `default` tier and any --vsr-* run through "
            "AVAssetWriter (no ffmpeg dependency); other tiers stay on "
            "ffmpeg.  Force `videotoolbox` to require the AVWriter path "
            "(valid only for --encode-tier default today).  Force "
            "`ffmpeg` to disable the VT path entirely; incompatible "
            "with any --vsr-* flag."
        ),
    )
    parser.add_argument(
        "--vsr-spatial-mode",
        choices=["off", "fast", "balanced", "image"],
        default="off",
        help=(
            "VideoToolbox spatial upscale, applied after VAE decode and "
            "before AVAssetWriter.  Scale is implied by the mode "
            "(fast=2x VTLowLatency; balanced=4x HQ Video - temporal "
            "feedback; image=4x HQ Image - per-frame deterministic).  "
            "off (default) skips the upscale entirely.  Engaging any "
            "non-off mode forces --output-backend=videotoolbox.  This "
            "is independent from the model-based --upscale-spatial; "
            "the two should not be combined."
        ),
    )
    parser.add_argument(
        "--vsr-target-fps",
        type=float,
        default=None,
        help=(
            "VideoToolbox frame-rate conversion target.  When set and "
            "different from --fps, VTFrameRateConversion interpolates "
            "between source frames at the requested rate (e.g. 24->48 "
            "for 2x slow-mo, 24->60 for high-refresh playback).  "
            "Engaging this forces --output-backend=videotoolbox.  "
            "Independent from --upscale-temporal (model-based)."
        ),
    )
    parser.add_argument(
        "--vsr-temporal-mode",
        choices=["normal", "high"],
        default="normal",
        help=(
            "VTFrameRateConversion quality.  Only meaningful with "
            "--vsr-target-fps.  normal (default) = fast, adequate for "
            "~2x rate-up; high = QualityPrioritizationQuality, slower "
            "but cleaner motion."
        ),
    )
    parser.add_argument(
        "--vsr-save-original",
        action="store_true",
        help=(
            "When --vsr-spatial-mode or --vsr-target-fps is engaged, "
            "also write the un-processed source-resolution source-fps "
            "mp4 alongside the VSR/VTFRC output as `<stem>_orig.mp4`. "
            "The companion writer mirrors the primary's HEVC profile "
            "(RGBAHalf + Main42210 for VSR HQ, NV12 + Main10 for VSR "
            "fast / VTFRC-only) so the A/B is precision-floor matched. "
            "Both files share the same audio track so each is playable "
            "standalone.  Useful for A/B comparisons against the "
            "upscaled / temporally-interpolated version.  Cost per "
            "source frame is one additional source-buffer upload + one "
            "HEVC HW encode pass; the second AVAssetWriter pump runs "
            "on its own GCD queue so wall-time impact is small.  No-op "
            "when no VT post-processing is engaged.  Implied by "
            "--save-all-sidecars."
        ),
    )
    parser.add_argument(
        "--vsr-encode-quality",
        type=float,
        default=0.65,
        help=(
            "AVVideoQualityKey for the AVAssetWriter HEVC encoder when "
            "--output-backend=videotoolbox.  0.65 matches the ffmpeg "
            "default tier's -q:v 65; raise toward 1.0 for higher "
            "bitrate / lower-loss output."
        ),
    )
    parser.add_argument(
        "--audio-onset-trim",
        type=str,
        default="auto",
        help=(
            "Sequence-start audio-spike mitigation.  Some AV generations "
            "produce a loud click at t=0 followed by silence before the "
            "first spoken word (see docs/AUDIO_ISSUES.md -> Sequence-Start "
            "Audio Spike).  auto (default) runs a two-window detector "
            "(first 50 ms > 2x global RMS AND 100-250 ms < 0.1x global "
            "RMS) and, when the click signature is present, zero-fills "
            "the leading 120 ms of audio (sample-count preserved -> AV "
            "sync safe).  off disables the check.  A numeric value (ms) "
            "force-trims that duration unconditionally."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Exact output path. When omitted, uses --output-dir/--output-prefix with a timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for timestamped default outputs. Default resolves "
            "DIFFUSERS_OUTPUT_DIR, then OUTPUT_DIR, then outputs/."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ltx",
        help="Filename prefix for timestamped default outputs.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Path to a full LTX weight bundle. Default resolves the "
            "Lightricks/LTX-2.3 distilled/dev checkpoint from HF_HOME or "
            "HF_HUB_CACHE. Advanced per-subsystem flags override this bundle "
            "for only that subsystem."
        ),
    )
    parser.add_argument(
        "--transformer-weights",
        type=str,
        default=None,
        help="Optional transformer-only weight override. Defaults to --weights.",
    )
    parser.add_argument(
        "--connector-weights",
        type=str,
        default=None,
        help="Optional text connector / AV projection weight override. Defaults to --weights.",
    )
    parser.add_argument(
        "--vae-weights",
        "--video-vae-weights",
        dest="video_vae_weights",
        type=str,
        default=None,
        help="Optional video VAE weight override. Defaults to --weights.",
    )
    parser.add_argument(
        "--audio-vae-weights",
        type=str,
        default=None,
        help="Optional audio VAE weight override. Defaults to --weights.",
    )
    parser.add_argument(
        "--vocoder-weights",
        type=str,
        default=None,
        help="Optional vocoder weight override. Defaults to --weights.",
    )
    parser.add_argument(
        "--config-weights",
        type=str,
        default=None,
        help=(
            "Optional config/metadata source for model version, VAE shape, and "
            "vocoder type. Defaults to --weights."
        ),
    )
    parser.add_argument(
        "--weights-cache",
        choices=["off", "auto", "rebuild"],
        default="auto",
        help=(
            "Disposable converted-weight cache. 'auto' builds on first "
            "use and reuses matching transformer and per-family artifacts; 'rebuild' "
            "forces fresh cache files; 'off' loads stock weights directly."
        ),
    )
    parser.add_argument(
        "--weights-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for --weights-cache artifacts. Defaults to "
            "$LTX_MLX_WEIGHTS_CACHE_DIR, then "
            "$LTX_MLX_TRANSFORMER_CACHE_DIR (legacy alias), then "
            "~/.cache/ltx-2-mlx/weights-cache."
        ),
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=parse_non_negative_float,
        default=1.0,
        help=(
            "Limit MLX's in-memory allocator cache in decimal GB. Default 1. "
            "Use 0 to return "
            "freed buffers immediately. This is separate from --weights-cache and "
            "can reduce system memory pressure."
        ),
    )
    parser.add_argument(
        "--stream-transformer",
        action="store_true",
        help=(
            "Enable the recommended cache-backed transformer streaming preset "
            "(16 resident blocks, mx.compile, 4-block compile groups)."
        ),
    )
    parser.add_argument(
        "--transformer-block-resident-blocks",
        type=parse_non_negative_int,
        default=0,
        help=(
            "Experimental cache-backed transformer block streaming. 0 disables it; "
            "positive values keep that many block modules resident and rotate cached "
            "weights through them. Implies --weights-cache auto when the cache is off."
        ),
    )
    parser.add_argument(
        "--transformer-block-compile",
        action="store_true",
        help=(
            "Experimental: with --transformer-block-resident-blocks, compile each "
            "resident block window with mx.compile(inputs=blocks). Falls back to "
            "eager streaming if MLX rejects the compiled group."
        ),
    )
    parser.add_argument(
        "--transformer-block-compile-group-size",
        type=parse_non_negative_int,
        default=0,
        metavar="BLOCKS",
        help=(
            "Experimental: with --transformer-block-compile, compile/eval at most "
            "this many resident blocks per command-buffer group. 0 uses the full "
            "resident window."
        ),
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Use placeholder inference (skip model loading)"
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip VAE decoding (output latent visualization instead)"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Path to pre-computed text embedding (.npz)"
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default=None,
        help=(
            "Path to Gemma 3 weights directory. Default resolves "
            "google/gemma-3-12b-it from HF_HOME or HF_HUB_CACHE."
        ),
    )
    parser.add_argument(
        "--no-gemma",
        action="store_true",
        help="Use dummy embeddings instead of real Gemma encoding (for testing)"
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(SUPPORTED_COMPUTE_DTYPES),
        default="bfloat16",
        help="Compute dtype for model execution (default: bfloat16)"
    )
    parser.add_argument(
        "--transformer-dtype",
        choices=sorted(SUPPORTED_COMPUTE_DTYPES),
        default=None,
        help=(
            "Compute dtype for the transformer denoise only (attention, FF, "
            "projections); baked into the weights cache. VAE, audio decoder, "
            "vocoder, and text encoding keep --dtype. Subsumes "
            "--video-ff-dtype/--audio-ff-dtype (and keeps their FP16 "
            "pretranspose auto-add). Defaults to --dtype."
        ),
    )
    parser.add_argument(
        "--vae-decoder",
        choices=["native"],
        default="native",
        help="Video VAE decoder backend (MLX-native Conv3d).",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["distilled", "dev"],
        default="distilled",
        help="Model variant: 'distilled' (fast, default 8 steps) or 'dev' (quality, default 30 steps)"
    )
    parser.add_argument(
        "--distilled-lora",
        type=str,
        default=None,
        help="Path to distilled LoRA weights (required for high-quality two-stage generation)"
    )
    parser.add_argument(
        "--distilled-lora-scale",
        type=float,
        default=1.0,
        help="Scale for distilled LoRA (default 1.0)"
    )
    parser.add_argument(
        "--distilled-lora-stage1-scale",
        "--distilled-lora-strength-stage-1",
        dest="distilled_lora_stage1_scale",
        type=float,
        default=None,
        help="Optional stage-1 strength for --distilled-lora in two-stage modes."
    )
    parser.add_argument(
        "--distilled-lora-stage2-scale",
        "--distilled-lora-strength-stage-2",
        dest="distilled_lora_stage2_scale",
        type=float,
        default=None,
        help="Optional stage-2 strength for --distilled-lora in two-stage modes."
    )
    parser.add_argument(
        "--upscale-spatial",
        action="store_true",
        help="Apply 2x spatial upscaling to output (256->512, etc.)"
    )
    parser.add_argument(
        "--spatial-upscaler-weights",
        type=str,
        default=None,
        help=(
            "Path to spatial upscaler weights. Default resolves the cached "
            "LTX-2.3 x2 upscaler beside --weights or from HF_HOME/HF_HUB_CACHE."
        ),
    )
    parser.add_argument(
        "--upscale-temporal",
        action="store_true",
        help="Apply 2x temporal upscaling to output (17->33 frames, etc.)"
    )
    parser.add_argument(
        "--temporal-upscaler-weights",
        type=str,
        default="weights/ltx-2/ltx-2-temporal-upscaler-x2-1.0.safetensors",
        help="Path to temporal upscaler weights"
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate synchronized audio with video (requires AudioVideo model weights)"
    )
    parser.add_argument(
        "--internal-audio",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Control whether the AV transformer runs its internal audio branch "
            "(audio self-attn, audio text-attn, A2V/V2A cross-modal).  This is "
            "independent of audio output: the audio branch runs by default even "
            "when --generate-audio is off, and discarding the result is wasted "
            "compute.\n"
            "  auto  (default): on iff --generate-audio is set.\n"
            "  on            : always run the audio branch.\n"
            "  off           : never run the audio branch.  Disables audio "
            "output regardless of --generate-audio.\n"
            "Effect on a small distilled run (256x256, 25 frames, V2.3): "
            "stage-1 ~26s -> ~11s, stage-2 ~7.5s -> ~5.4s when off vs on.  "
            "(LTX_DISABLE_INTERNAL_AUDIO=1 still works as a legacy override.)"
        ),
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help=(
            "Legacy emergency memory mode. Adds more frequent eval checkpoints "
            "and sequential CFG materialization; usually slower and mostly "
            "redundant with --stream-transformer for distilled runs."
        ),
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Experimental: Skip intermediate evaluations during denoising. "
             "May increase memory usage. Not recommended for 19B models - "
             "the GPU is already fully utilized, so this typically doesn't help."
    )
    parser.add_argument(
        "--video-ff-dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float16"],
        help=(
            "Experimental: cast the video FF interior (project_in / GELU / "
            "project_out) to this dtype.  Residual stream and attention stay "
            "at the loaded checkpoint dtype.  FP16 is ~14-17%% faster than "
            "BF16 on the LTX FF matmul shapes per local microbench; the trade "
            "is FP16's smaller exponent range (max ~65504) vs BF16's "
            "FP32-equivalent range.  Inputs to the FF are AdaLN-normalized so "
            "magnitude overflow is unlikely; output goes straight to the "
            "BF16 residual stream so FP16 drift doesn't propagate beyond one "
            "block.  Default (omit flag): same as the loaded checkpoint dtype "
            "(BF16 for LTX 2.3 distilled).  Validate with --save-latents A/B."
        ),
    )
    parser.add_argument(
        "--audio-ff-dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float16"],
        help=(
            "Experimental: cast the AUDIO FF interior to this dtype, mirror of "
            "--video-ff-dtype for the audio branch.  Microbench predicts "
            "sub-noise per-generation savings (~0.07%% on bakery) because "
            "audio FF per-call wall is small (~2-3 ms vs video's 200-300 ms), "
            "but the per-matmul FP16 win is real (~10-13%%) so exposed for "
            "real-world A/B testing.  No kernel cliff at audio K=8192 (unlike "
            "video K=16384), but auto-pairs with audio FF pretranspose to "
            "avoid the FP16 x BF16 -> FP32 mixed-dtype promotion fallback.  "
            "Default (omit flag): keep audio FF at the loaded checkpoint dtype."
        ),
    )
    parser.add_argument(
        "--profile-transformer-once",
        action="store_true",
        help="Diagnostic: profile the first denoise transformer call with forced eval checkpoints. "
             "This perturbs timing and is intended for hotspot analysis, not final benchmark runs."
    )
    parser.add_argument(
        "--profile-transformer-steps",
        type=parse_profile_transformer_steps,
        default=(),
        metavar="STEPS",
        help="Diagnostic: comma-separated 1-based denoise steps to profile, e.g. '1,2,8'. "
             "Each profiled step inserts forced eval checkpoints and perturbs timing."
    )
    parser.add_argument(
        "--profile-transformer-blocks",
        type=parse_profile_transformer_blocks,
        default=(),
        metavar="BLOCKS",
        help="Diagnostic: comma-separated 0-based transformer blocks to profile in detail "
             "within selected --profile-transformer-steps, e.g. '0,40,47'."
    )
    parser.add_argument(
        "--video-ff-quantize",
        type=parse_video_ff_quantize_specs,
        default=(),
        metavar="TARGET:MODE[,TARGET:MODE]",
        help="Experimental shortcut that enables video FF quantization with per-target "
             "modes, e.g. 'project_in:mxfp8' or 'project_in:mxfp8,project_out:mxfp8'. "
             "Targets without a mode default to mxfp8."
    )
    parser.add_argument(
        "--video-ff-quantize-layers",
        type=parse_transformer_layer_selection,
        default=(),
        metavar="LAYERS",
        help="Optional 0-based layer list/ranges for --video-ff-quantize, "
             "for example '40-47' or '32,40-47'. Leave unset to quantize all video layers."
    )
    parser.add_argument(
        "--video-ff-quantize-group-size",
        type=int,
        default=None,
        help="Optional group size for --video-ff-quantize. "
             "Leave unset to use MLX defaults for the selected mode."
    )
    parser.add_argument(
        "--video-ff-quantize-bits",
        type=int,
        default=None,
        help="Optional bit width for --video-ff-quantize. "
             "Leave unset to use MLX defaults for the selected mode."
    )
    parser.add_argument(
        "--transformer-cache-quantize",
        choices=TRANSFORMER_CACHE_QUANTIZE_MODES,
        default=TRANSFORMER_CACHE_QUANTIZE_OFF,
        help=(
            "Build/load a quantized transformer block cache. "
            "'mxfp8-blocks' mirrors the downloaded MXFP8 block32 policy in "
            "MLX-native QuantizedLinear format; 'mxfp8-blocks-pretranspose' "
            "packs weight.T and uses transpose=False quantized matmul. "
            "Can run resident or with block streaming."
        ),
    )
    parser.add_argument(
        "--video-ff-layout",
        type=parse_video_ff_layout_specs,
        default=DEFAULT_VIDEO_FF_LAYOUT_SPECS,
        metavar="TARGET:LAYOUT[,TARGET:LAYOUT]",
        help=(
            "Same-math video FF layout transform. Default is "
            "project_out:pretranspose (the matmul where pretranspose rescues "
            "a kernel-selection cliff -- 35%% faster per scripts/"
            "bench_ff_microbench.py bf16_layout). Add project_in:pretranspose "
            "if you want the historical-default behavior (microbench showed "
            "+2.5%% regression in isolation, neutral end-to-end). Use 'off' "
            "to disable."
        ),
    )
    parser.add_argument(
        "--video-ff-layout-layers",
        type=parse_transformer_layer_selection,
        default=(),
        metavar="LAYERS",
        help="Optional 0-based layer list/ranges for --video-ff-layout, "
             "for example '0-47'. Leave unset to transform all video layers."
    )
    parser.add_argument(
        "--video-attn-layout",
        type=parse_video_attn_layout_specs,
        default=DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
        metavar="TARGET:LAYOUT[,TARGET:LAYOUT]",
        help=(
            "Same-math video attention layout transform. Default is OFF "
            "(empty). Per scripts/bench_ff_microbench.py bf16_layout, "
            "to_q/to_k/to_v/to_out pretranspose at the 4096x4096 attention "
            "shape is tied within noise with naive BF16 (within +-1%%). "
            "Re-enable per-target: to_out:pretranspose,to_q:pretranspose,... "
            "Use 'off' to keep the new default explicit."
        ),
    )
    parser.add_argument(
        "--video-attn-layout-layers",
        type=parse_transformer_layer_selection,
        default=(),
        metavar="LAYERS",
        help="Optional 0-based layer list/ranges for --video-attn-layout, "
             "for example '0-47'. Leave unset to transform all video layers."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to conditioning image for image-to-video generation"
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=0.95,
        help="Conditioning strength for --image (0.0-1.0, default 0.95)"
    )
    parser.add_argument(
        "--lora",
        type=str,
        action="append",
        default=None,
        help="Path to LoRA weights (.safetensors). Repeat to fuse multiple "
             "LoRAs; pair with repeated --lora-strength for per-LoRA strength."
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        action="append",
        default=None,
        help="LoRA strength (-2.0 to 2.0, default 1.0). Repeat to match the "
             "order of --lora; a single value applies to all."
    )
    parser.add_argument(
        "--lora-stage1-strength",
        "--lora-strength-stage-1",
        dest="lora_stage1_strength",
        type=float,
        action="append",
        default=None,
        help="Optional stage-1 LoRA strength. Repeat to match --lora; a single value applies to all."
    )
    parser.add_argument(
        "--lora-stage2-strength",
        "--lora-strength-stage-2",
        dest="lora_stage2_strength",
        type=float,
        action="append",
        default=None,
        help="Optional stage-2 LoRA strength. Repeat to match --lora; a single value applies to all."
    )
    parser.add_argument(
        "--stage2-lora-fuse-mode",
        "--lora-stage2-fuse-mode",
        choices=("delta", "fresh-total"),
        default="fresh-total",
        help=(
            "Two-stage LoRA transition strategy. fresh-total (default) "
            "restores the base transformer from the weight cache before stage "
            "2 and fuses the full stage-2 LoRA totals. delta applies "
            "stage2-stage1 strength changes over stage-1 fused weights."
        ),
    )
    parser.add_argument(
        "--lora-allow-partial",
        action="store_true",
        help="Allow a LoRA fuse that places <50%% of its resolved targets "
             "(otherwise a hard error flags a likely format/model mismatch). "
             "Use for LoRAs that intentionally modify only a few weights."
    )
    parser.add_argument(
        "--lora-exclude",
        type=str,
        action="append",
        default=None,
        help="Comma-separated LoRA target categories to drop from the fuse. "
             "Coarse: branches video/audio/cross, types attn/gate/ff. Fine "
             "(exact module): attn1, attn2, audio_attn1, audio_attn2, "
             "video_to_audio_attn, audio_to_video_attn, ff, audio_ff. Fine "
             "(projection): to_q, to_k, to_v, to_out, to_gate_logits, "
             "project_in, project_out. Control-path: adaln, prompt_adaln, "
             "scale_shift, prompt_scale_shift, gate_adaln, av_ca, "
             "cross_control, distill_control. Example: 'audio,cross' keeps a "
             "video-only "
             "style; 'video_to_audio_attn' drops only the sound-follows-motion "
             "bridge while keeping lip-sync; 'cross_control' drops "
             "prompt/audio-video "
             "cross-conditioning; 'distill_control' drops the "
             "official-distillation-style control-path set. Repeat to match "
             "the order of --lora; a single value applies to all; use 'none' "
             "to skip filtering for one LoRA in a per-LoRA list."
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=0.0,
        help="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. (EXPERIMENTAL)"
    )
    parser.add_argument(
        "--stg-mode",
        type=str,
        choices=["video", "audio", "both"],
        default="video",
        help="STG perturbation mode: video, audio, or both (EXPERIMENTAL)"
    )
    # APG (Adaptive Projected Guidance) arguments
    parser.add_argument(
        "--apg-scale",
        type=float,
        default=1.0,
        help="APG (Adaptive Projected Guidance) scale. 1.0 disables APG, use values like 3.0-7.0"
    )
    parser.add_argument(
        "--apg-eta",
        type=float,
        default=1.0,
        help="APG parallel component weight (default 1.0)"
    )
    parser.add_argument(
        "--apg-norm-threshold",
        type=float,
        default=0.0,
        help="APG norm threshold for guidance clipping (0 = disabled)"
    )
    parser.add_argument(
        "--apg-momentum",
        type=float,
        default=0.0,
        help="APG momentum for stateful guidance (0 = disabled, try 0.5-0.9)"
    )
    # GE (Gradient Estimation) denoising argument
    parser.add_argument(
        "--ge-gamma",
        type=float,
        default=0.0,
        help="GE (Gradient Estimation) gamma. 0.0 disables GE, try 2.0 to reduce steps"
    )
    # IC-LoRA control signal arguments
    parser.add_argument(
        "--control-video",
        type=str,
        default=None,
        help="Path to control video for IC-LoRA conditioning (depth, pose, canny)"
    )
    parser.add_argument(
        "--control-type",
        type=str,
        choices=["canny", "raw"],
        default="raw",
        help="Control signal type: 'canny' applies edge detection, 'raw' uses video as-is"
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=100,
        help="Canny edge detection low threshold (0-255)"
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=200,
        help="Canny edge detection high threshold (0-255)"
    )
    parser.add_argument(
        "--control-strength",
        type=float,
        default=0.95,
        help="Control signal strength (0.0-1.0, default 0.95)"
    )
    parser.add_argument(
        "--save-control",
        action="store_true",
        help="Save the preprocessed control signal video for debugging"
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help=(
            "Save video/audio latents as an NPZ sidecar; distilled two-stage "
            "runs include stage-1, stage-2, and final aliases"
        ),
    )
    parser.add_argument(
        "--save-text-embeddings",
        "--save-text-conditioning",
        dest="save_text_embeddings",
        action="store_true",
        help=(
            "Save positive/negative video/audio text conditioning tensors as an "
            "_text.npz sidecar next to the requested output"
        ),
    )
    parser.add_argument(
        "--save-run-log",
        "--save-metadata",
        dest="save_run_log",
        action="store_true",
        help="Save generation parameters, argv, output paths, and timings as an _run.json sidecar",
    )
    parser.add_argument(
        "--save-audio-sidecar",
        action="store_true",
        help=(
            "Write a sidecar .wav next to the output video carrying the "
            "vocoder output before audio-codec compression. Useful for A/B "
            "comparison against ALAC/AAC. Implied by --save-all-sidecars."
        ),
    )
    parser.add_argument(
        "--save-all-sidecars",
        "--save-debug-sidecars",
        dest="save_all_sidecars",
        action="store_true",
        help=(
            "Enable all reproducibility/debug sidecars: latents, text "
            "conditioning, run metadata, audio WAV, and (when VSR/VTFRC "
            "is engaged) the pre-VSR original mp4."
        ),
    )
    parser.add_argument(
        "--tiled-vae",
        action="store_true",
        help="Legacy alias: force default tiled VAE decoding for lower memory usage"
    )
    parser.add_argument(
        "--vae-tiling",
        choices=["auto", "off", "custom"],
        default="auto",
        help=(
            "VAE decode tiling policy. auto preserves the pipeline default, off decodes "
            "the full latent volume, and custom uses the tile sizes below."
        ),
    )
    parser.add_argument(
        "--vae-temporal-tile-frames",
        type=parse_non_negative_int,
        default=None,
        help="Custom VAE temporal tile size in decoded frames; must be divisible by 8, or 0 to disable temporal tiling",
    )
    parser.add_argument(
        "--vae-temporal-overlap-frames",
        type=parse_non_negative_int,
        default=24,
        help="Custom VAE temporal overlap in decoded frames; must be divisible by 8",
    )
    parser.add_argument(
        "--vae-spatial-tile-pixels",
        type=parse_non_negative_int,
        default=None,
        help="Custom VAE spatial tile size in decoded pixels; must be divisible by 32, or 0 to disable spatial tiling",
    )
    parser.add_argument(
        "--vae-spatial-overlap-pixels",
        type=parse_non_negative_int,
        default=64,
        help="Custom VAE spatial overlap in decoded pixels; must be divisible by 32",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["text-to-video", "distilled", "one-stage", "two-stage", "ic-lora", "keyframe-interpolation"],
        default="text-to-video",
        help="Pipeline type (default: text-to-video)"
    )
    # Keyframe interpolation arguments
    parser.add_argument(
        "--keyframe",
        type=str,
        action="append",
        default=None,
        help="Keyframe image in format 'path:frame_index' or 'path:frame_index:strength'. Can be specified multiple times."
    )
    # IC-LoRA arguments
    parser.add_argument(
        "--ic-lora-weights",
        type=str,
        default=None,
        help="Path to IC-LoRA weights for video-to-video generation"
    )
    parser.add_argument(
        "--early-layers-only",
        action="store_true",
        help="[EXPERIMENTAL] Use only Layer 0 (input embeddings) from Gemma. "
             "Preserves text differentiation (~0.4 corr vs ~0.999+ with full pipeline)."
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Use Gemma to expand short prompts into detailed descriptions before encoding. "
             "This matches the official LTX-2 pipeline behavior and improves text differentiation."
    )
    parser.add_argument(
        "--cross-attn-scale",
        type=float,
        default=1.0,
        help="Scale factor for cross-attention in late transformer layers (40-47). "
             "Values 5-10 improve text conditioning for semantic content generation. "
             "Default 1.0 preserves original behavior."
    )

    args = parser.parse_args()

    if args.save_all_sidecars:
        args.save_latents = True
        args.save_text_embeddings = True
        args.save_run_log = True
        args.save_audio_sidecar = True
        # The pre-VSR original mp4 is a sidecar in spirit - it lives next
        # to the requested output and helps reproduce / compare runs.
        # Implicit no-op when no VT post-processing is engaged (the
        # companion writer only fires when vsr or vtfrc is alive).
        args.vsr_save_original = True

    # Resolve --audio-onset-trim {auto, off, <ms>} into (mode, trim_ms).
    from LTX_2_MLX.audio import parse_trim_mode as _parse_onset_trim_mode

    try:
        args.audio_onset_trim_mode, args.audio_onset_trim_ms = _parse_onset_trim_mode(
            args.audio_onset_trim
        )
    except ValueError as exc:
        parser.error(str(exc))

    resolved_num_frames = resolve_num_frames(
        num_frames=args.frames,
        duration_seconds=args.duration,
        fps=args.fps,
    )
    if args.duration is not None:
        print(
            f"Resolved duration {args.duration}s at {args.fps}fps "
            f"to {resolved_num_frames} frames"
        )

    def _expand_optional_lora_values(values, count: int, option: str):
        if not values:
            return [None] * count
        if len(values) == 1:
            return values * count
        if len(values) == count:
            return values
        raise SystemExit(
            f"ERROR: got {count} --lora but {len(values)} {option}; pass one "
            "value or one per LoRA."
        )

    # Build LoRA configs from (possibly repeated) --lora / --lora-strength /
    # --lora-exclude. A single strength/exclude applies to all LoRAs; N values
    # pair with N LoRAs in order. Optional per-stage strengths are ignored in
    # one-stage modes and activate stage-managed fusion in two-stage modes.
    cli_lora_configs = None
    if not args.lora and (args.lora_stage1_strength or args.lora_stage2_strength):
        raise SystemExit(
            "ERROR: --lora-stage1-strength/--lora-stage2-strength require at least one --lora."
        )
    if args.lora:
        strengths = args.lora_strength or [1.0]
        if len(strengths) == 1:
            strengths = strengths * len(args.lora)
        if len(strengths) != len(args.lora):
            raise SystemExit(
                f"ERROR: got {len(args.lora)} --lora but "
                f"{len(strengths)} --lora-strength; pass one strength or one "
                "per LoRA."
            )

        def _parse_excl(spec: str) -> tuple:
            # "none"/"off"/empty -> no knockout for this LoRA (lets a per-LoRA
            # list opt one adapter out of filtering without an awkward "").
            if spec.strip().lower() in ("", "none", "off"):
                return ()
            return tuple(t.strip() for t in spec.split(",") if t.strip())

        raw_excl = args.lora_exclude or []
        if len(raw_excl) == 0:
            excludes = [()] * len(args.lora)
        elif len(raw_excl) == 1:
            excludes = [_parse_excl(raw_excl[0])] * len(args.lora)
        elif len(raw_excl) == len(args.lora):
            excludes = [_parse_excl(e) for e in raw_excl]
        else:
            raise SystemExit(
                f"ERROR: got {len(args.lora)} --lora but "
                f"{len(raw_excl)} --lora-exclude; pass one or one per LoRA."
            )

        try:
            stage1_strengths = _expand_optional_lora_values(
                args.lora_stage1_strength,
                len(args.lora),
                "--lora-stage1-strength",
            )
            stage2_strengths = _expand_optional_lora_values(
                args.lora_stage2_strength,
                len(args.lora),
                "--lora-stage2-strength",
            )
            cli_lora_configs = [
                LoRAConfig(
                    path=p,
                    strength=s,
                    stage_1_strength=s1,
                    stage_2_strength=s2,
                    exclude=ex,
                )
                for p, s, s1, s2, ex in zip(
                    args.lora,
                    strengths,
                    stage1_strengths,
                    stage2_strengths,
                    excludes,
                    strict=True,
                )
            ]
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from None

    distilled_stage_strength_requested = (
        args.distilled_lora_stage1_scale is not None
        or args.distilled_lora_stage2_scale is not None
    )
    if distilled_stage_strength_requested and not args.distilled_lora:
        raise SystemExit(
            "ERROR: --distilled-lora-stage1-scale/--distilled-lora-stage2-scale require --distilled-lora."
        )
    legacy_distilled_lora = args.distilled_lora
    if args.distilled_lora and distilled_stage_strength_requested:
        try:
            distilled_config = LoRAConfig(
                path=args.distilled_lora,
                strength=args.distilled_lora_scale,
                stage_1_strength=args.distilled_lora_stage1_scale,
                stage_2_strength=args.distilled_lora_stage2_scale,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        cli_lora_configs = (cli_lora_configs or []) + [distilled_config]
        legacy_distilled_lora = None

    generate_video(
        distilled_lora=legacy_distilled_lora,
        distilled_lora_scale=args.distilled_lora_scale,

        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=resolved_num_frames,
        num_steps=args.steps,
        cfg_scale=args.cfg,
        guidance_rescale=getattr(args, 'guidance_rescale', 0.7),
        seed=args.seed,
        weights_path=args.weights,
        transformer_weights_path=args.transformer_weights,
        connector_weights_path=args.connector_weights,
        video_vae_weights_path=args.video_vae_weights,
        audio_vae_weights_path=args.audio_vae_weights,
        vocoder_weights_path=args.vocoder_weights,
        config_weights_path=args.config_weights,
        output_path=args.output,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        use_placeholder=args.placeholder,
        skip_vae=args.skip_vae,
        embedding_path=args.embedding,
        gemma_path=args.gemma_path,
        use_gemma=not args.no_gemma,
        dtype=args.dtype,
        transformer_dtype=args.transformer_dtype,
        vae_decoder_backend=args.vae_decoder,
        model_variant=args.model_variant,
        upscale_spatial=args.upscale_spatial,
        spatial_upscaler_weights=args.spatial_upscaler_weights,
        upscale_temporal=args.upscale_temporal,
        temporal_upscaler_weights=args.temporal_upscaler_weights,
        generate_audio=args.generate_audio,
        internal_audio=args.internal_audio,
        low_memory=args.low_memory,
        fast_mode=args.fast_mode,
        profile_transformer_once=args.profile_transformer_once,
        profile_transformer_steps=args.profile_transformer_steps,
        profile_transformer_blocks=args.profile_transformer_blocks,
        video_ff_quantize_specs=args.video_ff_quantize,
        video_ff_quantize_group_size=args.video_ff_quantize_group_size,
        video_ff_quantize_bits=args.video_ff_quantize_bits,
        video_ff_quantize_layers=args.video_ff_quantize_layers,
        video_ff_layout_specs=_ensure_ff_layout_for_dtype(
            args.video_ff_layout,
            # An FP16 transformer makes the FF interior FP16 regardless of the
            # FF flags, so the pretranspose auto-add (BlockLoader-cliff dodge)
            # must fire for it too.
            args.video_ff_dtype
            or ("float16" if args.transformer_dtype == "float16" else None),
        ),
        video_ff_layout_layers=args.video_ff_layout_layers,
        video_attn_layout_specs=args.video_attn_layout,
        video_attn_layout_layers=args.video_attn_layout_layers,
        transformer_cache_quantize=args.transformer_cache_quantize,
        weights_cache_mode=args.weights_cache,
        weights_cache_dir=args.weights_cache_dir,
        mlx_cache_limit_gb=args.mlx_cache_limit_gb,
        stream_transformer=args.stream_transformer,
        transformer_block_resident_blocks=args.transformer_block_resident_blocks,
        transformer_block_compile=args.transformer_block_compile,
        transformer_block_compile_group_size=args.transformer_block_compile_group_size,
        # New parameters
        image_path=args.image,
        image_strength=args.image_strength,
        lora_configs=cli_lora_configs,
        lora_allow_partial=args.lora_allow_partial,
        stage2_lora_fuse_mode=args.stage2_lora_fuse_mode,
        tiled_vae=args.tiled_vae,
        vae_tiling_mode=args.vae_tiling,
        vae_temporal_tile_frames=args.vae_temporal_tile_frames,
        vae_temporal_overlap_frames=args.vae_temporal_overlap_frames,
        vae_spatial_tile_pixels=args.vae_spatial_tile_pixels,
        vae_spatial_overlap_pixels=args.vae_spatial_overlap_pixels,
        pipeline_type=args.pipeline,
        early_layers_only=args.early_layers_only,
        enhance_prompt_flag=args.enhance_prompt,
        cross_attn_scale=args.cross_attn_scale,
        video_ff_dtype=args.video_ff_dtype,
        audio_ff_dtype=args.audio_ff_dtype,
        # Two-stage pipeline parameters
        steps_stage1=args.steps_stage1,
        steps_stage2=args.steps_stage2,
        cfg_stage1=args.cfg_stage1,
        # STG parameters
        stg_scale=args.stg_scale,
        stg_mode=args.stg_mode,
        # APG parameters
        apg_scale=args.apg_scale,
        apg_eta=args.apg_eta,
        apg_norm_threshold=args.apg_norm_threshold,
        apg_momentum=args.apg_momentum,
        # IC-LoRA control parameters
        control_video=args.control_video,
        control_type=args.control_type,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        control_strength=args.control_strength,
        save_control=args.save_control,
        save_latents=args.save_latents,
        save_text_embeddings=args.save_text_embeddings,
        save_run_log=args.save_run_log,
        save_audio_sidecar=args.save_audio_sidecar,
        # GE (Gradient Estimation) parameter
        ge_gamma=args.ge_gamma,
        # Output FPS + encode tier
        output_fps=args.fps,
        output_tier=args.encode_tier,
        # Output backend / VideoToolbox post-processing
        output_backend=args.output_backend,
        vsr_spatial_mode=args.vsr_spatial_mode,
        vsr_target_fps=args.vsr_target_fps,
        vsr_temporal_mode=args.vsr_temporal_mode,
        vsr_save_original=args.vsr_save_original,
        vsr_encode_quality=args.vsr_encode_quality,
        # Audio onset (sequence-start spike) mitigation
        audio_onset_trim_mode=args.audio_onset_trim_mode,
        audio_onset_trim_ms=args.audio_onset_trim_ms,
        # IC-LoRA and Keyframe Interpolation
        keyframes=args.keyframe,
        ic_lora_weights=args.ic_lora_weights,
    )


if __name__ == "__main__":
    main()
