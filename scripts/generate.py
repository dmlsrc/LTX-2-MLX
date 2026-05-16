#!/usr/bin/env python3
"""Generate video from text prompt using LTX-2 MLX."""

import argparse
import gc
import json
import os
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.transformer import (
    LTXAVModel,
    LTXModel,
    LTXModelType,
    Modality,
    MultiModalTransformerArgsPreprocessor,
    X0Model,
)
from LTX_2_MLX.components.patchifiers import get_pixel_coords
from LTX_2_MLX.types import SpatioTemporalScaleFactors
from LTX_2_MLX.model.audio_vae import (
    AudioDecoder,
    Vocoder,
    VocoderWithBWE,
    load_audio_decoder_weights,
    load_vocoder_weights,
    load_vocoder_with_bwe_weights,
)
from LTX_2_MLX.model.audio_vae.vocoder import MelSTFT
from LTX_2_MLX.model.video_vae import VideoDecoder, NormLayerType
from LTX_2_MLX.components import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    VideoLatentPatchifier,
    get_sigma_schedule,
)
from LTX_2_MLX.components.guiders import LtxAPGGuider, LegacyStatefulAPGGuider, STGGuider
from LTX_2_MLX.components.perturbations import create_batched_stg_config
from LTX_2_MLX.types import VideoLatentShape, NATIVE_FPS
from LTX_2_MLX.loader import (
    LoRAConfig,
    TRANSFORMER_CACHE_QUANTIZE_MODES,
    TRANSFORMER_CACHE_QUANTIZE_OFF,
    ensure_weight_family_caches,
    load_av_transformer_weights,
    load_transformer_weights,
    load_transformer_weights_cached,
    load_transformer_weights_cached_streaming,
)
from LTX_2_MLX.loader.lora_loader import fuse_lora_into_weights
from mlx.utils import tree_flatten
from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    decode_latent,
)
from LTX_2_MLX.model.video_vae.native_decoder import (
    NativeConv3dVideoDecoder,
    load_native_vae_decoder_weights,
)
from LTX_2_MLX.model.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from LTX_2_MLX.core_utils import to_velocity


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
    # (weights too tiny — 4096*32 / 2048*32 — for the implicit transpose to
    # matter).  Opt-in via explicit --video-attn-layout if you want to A/B.
    "to_gate_logits": ("pretranspose",),
}
DEFAULT_VIDEO_FF_LAYOUT_SPECS = (
    ("project_in", "pretranspose"),
    ("project_out", "pretranspose"),
)
# Pretranspose all four large attention projections by default.  to_gate_logits
# is intentionally excluded — its tiny weight shape gives no measurable benefit
# and adds cache-load + dispatch overhead.
DEFAULT_VIDEO_ATTN_LAYOUT_SPECS = (
    ("to_out", "pretranspose"),
    ("to_q",   "pretranspose"),
    ("to_k",   "pretranspose"),
    ("to_v",   "pretranspose"),
)
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
    add("/Users/Shared/huggingface/hub")
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


def parse_positive_float(value: str) -> float:
    """Parse a strictly positive float for user-facing CLI limits."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive number, got '{value}'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive number, got {parsed}")
    return parsed


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
        raw_step = raw_step.strip()
        if not raw_step:
            continue
        try:
            step = int(raw_step)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid denoise step '{raw_step}' in --profile-transformer-steps"
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
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        try:
            block = int(raw_block)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid transformer block '{raw_block}' in --profile-transformer-blocks"
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
    decoder_backend: str = "simple",
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
    return os.path.splitext(output_path)[0] + ".npz"


def text_sidecar_path(output_path: str) -> str:
    """Use the requested output stem for the text-conditioning sidecar."""
    return os.path.splitext(output_path)[0] + "_text.npz"


def run_log_sidecar_path(output_path: str) -> str:
    """Use the requested output stem for the run metadata sidecar."""
    return os.path.splitext(output_path)[0] + "_run.json"


def mlx_array_to_numpy(array: mx.array) -> np.ndarray:
    """Convert an MLX array to NumPy, preserving dtype where NumPy supports it."""
    mx.eval(array)
    try:
        return np.array(array)
    except (TypeError, RuntimeError):
        return np.array(array.astype(mx.float32))


def save_text_conditioning_sidecar(
    path: str,
    positive_video_encoding: mx.array,
    negative_video_encoding: mx.array,
    positive_mask: mx.array,
    negative_mask: mx.array,
    positive_audio_encoding: mx.array | None = None,
    negative_audio_encoding: mx.array | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> None:
    """Save text/video/audio conditioning tensors for later A/B diagnostics."""
    arrays = {
        "schema_version": np.array(1, dtype=np.int32),
        "prompt": np.array(prompt or ""),
        "negative_prompt": np.array(negative_prompt or ""),
        "positive_video_encoding": mlx_array_to_numpy(positive_video_encoding),
        "positive_video_encoding_mlx_dtype": str(positive_video_encoding.dtype),
        "negative_video_encoding": mlx_array_to_numpy(negative_video_encoding),
        "negative_video_encoding_mlx_dtype": str(negative_video_encoding.dtype),
        "positive_attention_mask": mlx_array_to_numpy(positive_mask),
        "positive_attention_mask_mlx_dtype": str(positive_mask.dtype),
        "negative_attention_mask": mlx_array_to_numpy(negative_mask),
        "negative_attention_mask_mlx_dtype": str(negative_mask.dtype),
    }
    if positive_audio_encoding is not None:
        arrays["positive_audio_encoding"] = mlx_array_to_numpy(positive_audio_encoding)
        arrays["positive_audio_encoding_mlx_dtype"] = str(positive_audio_encoding.dtype)
    if negative_audio_encoding is not None:
        arrays["negative_audio_encoding"] = mlx_array_to_numpy(negative_audio_encoding)
        arrays["negative_audio_encoding_mlx_dtype"] = str(negative_audio_encoding.dtype)

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(path, **arrays)
    print(f"  Saved text conditioning: {path}")


def save_run_log_sidecar(
    path: str,
    payload: dict,
    timings: "RunTimings",
    status: str,
    outputs: dict | None = None,
) -> None:
    """Save human-readable run metadata and timing information."""
    log = dict(payload)
    now = datetime.now(timezone.utc).isoformat()
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


def format_progress_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    whole_seconds = int(round(seconds - minutes * 60))
    if whole_seconds == 60:
        minutes += 1
        whole_seconds = 0
    return f"{minutes}m {whole_seconds:02d}s"


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


class DenoiseProgress:
    """Step-driven progress line for the denoise loop.  See PERFORMANCE.md
    section "Throttle terminal redraws" for the design rationale."""

    def __init__(self, label: str = "Denoising", width: int = 28, total: int | None = None):
        self.label = label
        self.width = width
        self.started_at = time.perf_counter()
        self.denoise_started_at: float | None = None
        self._last_step_at: float | None = None
        self._step_durations: list[float] = []
        self.step = 0
        self.total = total or 0
        self._lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._rendered = False
        self._finished = False
        self._newline_printed = False
        self._last_line = ""
        self._last_line_len = 0

    def start(self) -> None:
        self.started_at = time.perf_counter()
        self.denoise_started_at = None
        self._last_step_at = None
        self._step_durations = []
        self.step = 0
        self._render()

    def update(self, step: int, total: int) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._finished:
                return
            self.total = total
            step = max(0, min(step, total))
            if step == 0:
                self.denoise_started_at = now
                self._last_step_at = now
                self._step_durations = []
            elif step > self.step:
                if self.denoise_started_at is None:
                    # Older callers may not send the step-0 "denoise started" marker.
                    self.denoise_started_at = self.started_at
                    self._last_step_at = self.started_at
                previous = self._last_step_at or self.denoise_started_at
                self._step_durations.append(max(0.0, now - previous))
                self._last_step_at = now
            self.step = step
        self._render()

    def _render(self, final: bool = False) -> None:
        with self._lock:
            if self._finished and not final:
                return
            step = self.step
            total = self.total
            denoise_started_at = self.denoise_started_at
            step_durations = list(self._step_durations)

        now = time.perf_counter()
        run_start = denoise_started_at or now
        run_elapsed = max(0.0, now - run_start) if denoise_started_at is not None else 0.0
        progress = step / total if total else 1.0
        filled = min(self.width, int(round(progress * self.width)))
        bar = "#" * filled + "-" * (self.width - filled)
        first_step_text = (
            format_progress_duration(step_durations[0])
            if step_durations
            else "--"
        )
        if step_durations and total:
            seconds_per_step = sum(step_durations) / len(step_durations)
            eta_text = format_progress_duration(seconds_per_step * max(0, total - step))
            if seconds_per_step >= 1.0:
                pace = f"{seconds_per_step:.1f}s/it"
            else:
                pace = f"{1.0 / seconds_per_step:.2f}it/s"
        else:
            eta_text = "--"
            pace = "warming up" if denoise_started_at is None else "measuring"

        line = (
            f"  {self.label} [{bar}] {step:>3}/{total:<3} "
            f"{progress * 100:5.1f}% | STEP1 {first_step_text} "
            f"| RUN {format_progress_duration(run_elapsed)} "
            f"| ETA {eta_text} | {pace}"
        )
        with self._output_lock:
            # Skip the syscall + GPU repaint if nothing visible changed.
            if line == self._last_line and not final:
                return
            pad = " " * max(0, self._last_line_len - len(line))
            print("\r" + line + pad, end="", flush=True)
            self._last_line = line
            self._last_line_len = len(line)
            self._rendered = True

    def log(self, message: str) -> None:
        """Print a status message without merging it into the progress row."""
        with self._output_lock:
            if self._finished or not self._rendered or self._newline_printed:
                print(message, flush=True)
                return
            pad = " " * max(0, self._last_line_len - len(message))
            print("\r" + message + pad, flush=True)
            self._last_line = ""
            self._last_line_len = 0
            self._rendered = False

    def finish(self) -> None:
        with self._lock:
            already_finished = self._finished
            self._finished = True
            needs_newline = self._rendered and not self._newline_printed

        if not already_finished:
            self._render(final=True)

        if needs_newline:
            with self._output_lock:
                print(flush=True)
            with self._lock:
                self._newline_printed = True


def batched_cfg_forward(
    model,
    latent_patchified: mx.array,
    text_encoding: mx.array,
    text_mask: mx.array,
    null_encoding: mx.array,
    null_mask: mx.array,
    sigma: float,
    positions: mx.array,
) -> tuple:
    """
    Run CFG forward pass with batched inputs (2x speedup).

    Instead of two separate forward passes for cond and uncond,
    we stack them along the batch dimension and do a single pass.

    Returns:
        Tuple of (cond_output, uncond_output) both shape [1, T, C]
    """
    # Stack along batch dimension: [1, T, C] -> [2, T, C]
    batched_latent = mx.concatenate([latent_patchified, latent_patchified], axis=0)
    batched_context = mx.concatenate([text_encoding, null_encoding], axis=0)
    batched_mask = mx.concatenate([text_mask, null_mask], axis=0)
    batched_positions = mx.concatenate([positions, positions], axis=0)
    batched_timesteps = mx.array([sigma, sigma])

    # Single batched modality
    # NOTE: context_mask=None matches PyTorch behavior - they don't use text masks
    batched_modality = Modality(
        latent=batched_latent,
        context=batched_context,
        context_mask=None,
        timesteps=batched_timesteps,
        positions=batched_positions,
        enabled=True,
    )

    # Single forward pass (2x faster than two separate passes)
    batched_output = model(batched_modality)

    # Split back: [2, T, C] -> two [1, T, C]
    cond_output = batched_output[0:1]
    uncond_output = batched_output[1:2]

    return cond_output, uncond_output
from LTX_2_MLX.model.text_encoder.gemma3 import (
    Gemma3Config,
    Gemma3Model,
    load_gemma3_weights,
)
from LTX_2_MLX.model.text_encoder.encoder import (
    create_text_encoder,
    load_text_encoder_weights,
    create_av_text_encoder,
    load_av_text_encoder_weights,
    create_av_text_encoder_v2_from_checkpoint,
    load_av_text_encoder_v2_weights,
)
from LTX_2_MLX.model.upscaler import (
    SpatialUpscaler,
    load_spatial_upscaler_weights,
    TemporalUpscaler,
    load_temporal_upscaler_weights,
)
from LTX_2_MLX.pipelines.two_stage import (
    TwoStagePipeline,
    TwoStageCFGConfig,
)
from LTX_2_MLX.pipelines.one_stage import (
    OneStagePipeline,
    OneStageCFGConfig,
)
from LTX_2_MLX.pipelines.common import ImageCondition
from LTX_2_MLX.pipelines.ic_lora import (
    ControlType,
    VideoCondition,
    ICLoraPipeline,
    ICLoraConfig,
    preprocess_control_signal,
    load_control_signal_tensor,
)
from LTX_2_MLX.pipelines.keyframe_interpolation import (
    KeyframeInterpolationPipeline,
    KeyframeInterpolationConfig,
    Keyframe,
)
from LTX_2_MLX.model.video_vae.simple_encoder import (
    SimpleVideoEncoder,
    load_vae_encoder_weights,
)

def _read_checkpoint_config(checkpoint_path: str) -> dict:
    """Read the JSON config from checkpoint metadata."""
    import json
    try:
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="numpy") as f:
            metadata = f.metadata() or {}
        config_str = metadata.get("config", "{}")
        return json.loads(config_str)
    except Exception:
        return {}


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
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="numpy") as f:
            metadata = f.metadata() or {}
        return metadata.get("model_version", "")
    except Exception:
        return ""


def is_v2_model(checkpoint_path: str) -> bool:
    """Check if checkpoint is an LTX-2.3 (V2) model."""
    version = detect_model_version(checkpoint_path)
    return version.startswith("2.3")


def get_vae_config(checkpoint_path: str) -> dict:
    """Read VAE config from checkpoint metadata."""
    try:
        import json
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="numpy") as f:
            metadata = f.metadata() or {}
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
    else:
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
ENHANCE_SYSTEM_PROMPT = """You enhance short video descriptions into detailed prompts for a video generation model. You MUST preserve the exact subject, characters, and scene from the original — never replace or reinterpret them.

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


def create_chat_prompt(user_prompt: str) -> str:
    """Create a chat-format prompt for Gemma 3."""
    # Gemma 3 instruction-tuned format
    chat = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    return chat


def enhance_prompt(
    prompt: str,
    gemma_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Enhance prompt is not available — the Gemma QAT model used for encoding
    cannot do text generation. Returns the original prompt unchanged.
    """
    print(f"  Prompt enhancement not available (Gemma QAT model cannot generate text)")
    print(f"  Using original prompt as-is")
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

    print(f"  Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    
    # Weights load in their native bfloat16 via mx.load() - no dtype conversion needed.
    load_gemma3_weights(gemma, gemma_path)

    print(f"  Loading text encoder projection...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, ltx_weights_path)

    # Tokenize prompt directly (skip chat template - it dilutes the signal)
    # Chat template adds ~28 shared tokens, diluting the actual content
    # Without template: 0.71 correlation for blue vs red (good)
    # With template: 0.98 correlation (bad - template tokens dominate)
    print(f"  Tokenizing prompt...")
    encoding = tokenizer(
        prompt,  # Use raw prompt, not chat template
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
    print(f"  Running Gemma 3 forward pass (48 layers)...")
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
        print(f"  Processing through text encoder pipeline...")

        # Feature extraction (uses Layer 48 only for best differentiation)
        encoded = text_encoder.feature_extractor.extract_from_hidden_states(
            hidden_states=all_hidden_states,
            attention_mask=attention_mask,
            padding_side="left",
        )

        # Use connector (1D transformer with learnable registers)
        # Earlier testing showed connector homogenizes embeddings, but the model
        # may have been trained to expect connector output format
        print(f"  Processing through connector...")

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
    print(f"  Clearing Gemma from memory...")
    del gemma
    del text_encoder
    del all_hidden_states
    del last_hidden
    del tokenizer
    gc.collect()
    # Force MLX to release memory
    mx.metal.clear_cache()

    return encoded, original_mask


def encode_with_av_gemma(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    max_length: int = 1024,
) -> tuple:
    """
    Encode a single text prompt using the AudioVideo Gemma 3 + LTX-2 text encoder.

    Returns:
        Tuple of (video_encoding, audio_encoding, attention_mask) as MLX arrays.
    """
    results = encode_av_gemma_batch(
        prompts=[prompt],
        gemma_path=gemma_path,
        ltx_weights_path=ltx_weights_path,
        max_length=max_length,
    )
    if results is None:
        return None, None, None
    return results[0]


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

    print(f"  Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, gemma_path)

    # LTX_PAD_PROMPT_TO_MAX=1 restores the legacy padding-to-max behavior for
    # debugging.  Default skips it — running Gemma on a 1024-token padded
    # sequence when the real prompt is only a few tokens wastes O(N^2) attention
    # work that's discarded by the post-forward trim below.
    pad_to_max = bool(os.environ.get("LTX_PAD_PROMPT_TO_MAX"))

    gemma_outputs = []
    for i, prompt in enumerate(prompts):
        label = prompt_label(i, len(prompts))
        print(f"  Tokenizing {label}...")
        if pad_to_max:
            encoding = tokenizer(
                prompt,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        else:
            encoding = tokenizer(
                prompt,
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
                "real_token_count": 0,
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
            "real_token_count": real_token_count,
        })

        del last_hidden

    print(f"  Clearing Gemma from memory before AV text encoder...")
    del gemma
    del tokenizer
    gc.collect()
    mx.clear_cache()

    print(f"  Loading AV text encoder projection...")
    config_path = ltx_config_path or ltx_weights_path
    if is_v2_model(config_path):
        print(f"  Detected LTX-2.3 (V2) model — using V2 text encoder")
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
        real_token_count = gemma_output["real_token_count"]

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

        # Diagnostic: check encoding statistics for anomalies
        import numpy as _np
        for name, enc in [("video", av_output.video_encoding), ("audio", av_output.audio_encoding)]:
            arr = _np.array(enc[0].astype(mx.float32))
            real_part = arr[:real_token_count]
            reg_part = arr[real_token_count:]
            print(f"    {label} {name} real[:{real_token_count}]: mean={real_part.mean():.4f} std={real_part.std():.4f} "
                  f"min={real_part.min():.4f} max={real_part.max():.4f} nan={_np.isnan(real_part).sum()}")
            if reg_part.shape[0] > 0:
                print(f"    {label} {name} regs[{real_token_count}:]: mean={reg_part.mean():.4f} std={reg_part.std():.4f} "
                      f"min={reg_part.min():.4f} max={reg_part.max():.4f} nan={_np.isnan(reg_part).sum()}")

        results.append((av_output.video_encoding, av_output.audio_encoding, av_output.attention_mask))

        del all_hidden_states

    print(f"  Clearing AV text encoder from memory...")
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


def _npz_scalar_str(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _mlx_dtype_from_metadata(dtype_name: str | None) -> mx.Dtype | None:
    if not dtype_name:
        return None
    normalized = dtype_name.lower()
    if "bfloat16" in normalized:
        return mx.bfloat16
    if "float16" in normalized:
        return mx.float16
    if "float32" in normalized:
        return mx.float32
    if "int32" in normalized:
        return mx.int32
    if "int64" in normalized:
        return mx.int64
    if "bool" in normalized:
        return mx.bool_
    return None


def _load_mlx_npz_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    dtype_key: str | None = None,
) -> mx.array:
    array = mx.array(data[key])
    dtype = _mlx_dtype_from_metadata(_npz_scalar_str(data, dtype_key) if dtype_key else None)
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return array


def load_text_conditioning(embedding_path: str, use_av_encoder: bool) -> dict:
    """Load legacy text embeddings or the richer `_text.npz` conditioning sidecar."""
    data = np.load(embedding_path)

    if "positive_video_encoding" not in data.files:
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
        "positive_video_encoding": _load_mlx_npz_array(
            data,
            "positive_video_encoding",
            "positive_video_encoding_mlx_dtype",
        ),
        "positive_attention_mask": _load_mlx_npz_array(
            data,
            "positive_attention_mask",
            "positive_attention_mask_mlx_dtype",
        ),
        "negative_video_encoding": _load_mlx_npz_array(
            data,
            "negative_video_encoding",
            "negative_video_encoding_mlx_dtype",
        ) if "negative_video_encoding" in data.files else None,
        "negative_attention_mask": _load_mlx_npz_array(
            data,
            "negative_attention_mask",
            "negative_attention_mask_mlx_dtype",
        ) if "negative_attention_mask" in data.files else None,
        "positive_audio_encoding": _load_mlx_npz_array(
            data,
            "positive_audio_encoding",
            "positive_audio_encoding_mlx_dtype",
        ) if "positive_audio_encoding" in data.files else None,
        "negative_audio_encoding": _load_mlx_npz_array(
            data,
            "negative_audio_encoding",
            "negative_audio_encoding_mlx_dtype",
        ) if "negative_audio_encoding" in data.files else None,
    }

    print(f"  Loaded text conditioning from {embedding_path}")
    print(f"  Video shape: {loaded['positive_video_encoding'].shape}")
    if loaded["positive_audio_encoding"] is not None:
        print(f"  Audio shape: {loaded['positive_audio_encoding'].shape}")
    if _npz_scalar_str(data, "prompt"):
        print(f"  Original prompt: {_npz_scalar_str(data, 'prompt')}")
    return loaded


def load_vae_decoder(weights_path: str) -> VideoDecoder:
    """Load VAE decoder with weights."""
    print("Loading VAE decoder...")

    # LTX-2 decoder configuration
    decoder_blocks = [
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 4}),
        ("compress_time", {}),
        ("res_x", {"num_layers": 4}),
        ("compress_space", {}),
        ("res_x", {"num_layers": 4}),
        ("compress_space", {}),
    ]

    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
    )

    # TODO: Load VAE weights from file
    print("  VAE decoder created (weights not loaded yet)")

    return decoder


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
            )
            layouts_loaded_from_cache = True
        else:
            load_transformer_weights(model, weights_path)
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
    mem_str = " (low memory)" if low_memory else ""
    fast_str = " (fast mode)" if fast_mode else ""
    profile_str = " (profile first call)" if profile_transformer_once else ""
    v2_str = " (V2)" if cross_attention_adaln else ""
    print(f"Loading AudioVideo transformer ({compute_dtype_name(compute_dtype)}{mem_str}{fast_str}{profile_str}{v2_str})...")

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
        compute_dtype=compute_dtype,
        low_memory=low_memory,
        fast_mode=fast_mode,
        profile_transformer_once=profile_transformer_once,
        cross_attention_adaln=cross_attention_adaln,
        apply_gated_attention=apply_gated_attention,
        av_ca_timestep_scale_multiplier=1000,
    )

    layouts_loaded_from_cache = False

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
            )
            layouts_loaded_from_cache = True
        else:
            load_av_transformer_weights(model, weights_path)
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
    num_frames: int = 97,  # ~32s at 24fps (97 latent frames → 769 pixel frames via 8x VAE temporal compression)
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
    vae_decoder_backend: str = "native-conv3d",
    vae_spatial_padding: str = "zero",
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
    ge_gamma: float = 0.0,
    output_fps: float = NATIVE_FPS,
    output_speed: float = 1.0,
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
    distilled_one_stage_requested = (
        model_variant == "distilled" and pipeline_type in {"text-to-video", "one-stage"} and v2
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
        distilled_one_stage_requested
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
            "started_at": datetime.now(timezone.utc).isoformat(),
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
                "vae_decoder_backend": vae_decoder_backend,
                "vae_spatial_padding": vae_spatial_padding,
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
                "output_fps": output_fps,
                "output_speed": output_speed,
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
        }

    print(f"\n{'='*50}")
    print(f"LTX-2 MLX Video Generation")
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
    if not skip_vae:
        print(f"VAE tiling: {describe_vae_tiling_config(vae_tiling_config, vae_auto_tiling)}")
    if skip_vae:
        print(f"VAE decoding: SKIPPED")
    elif vae_decoder_backend == "native-conv3d":
        print("VAE decoder: native Conv3d")
    else:
        print("VAE decoder: simple slice-conv baseline")
    if not skip_vae and vae_spatial_padding == "zero":
        print("VAE spatial padding: zero (boundary-flicker mitigation)")
    elif not skip_vae:
        print("VAE spatial padding: reflect")
    if upscale_spatial:
        print(f"Spatial upscaling: 2x (output will be {width*2}x{height*2})")
    if upscale_temporal:
        print(f"Temporal upscaling: 2x (frames will be ~{num_frames*2})")
    if generate_audio:
        print(f"Audio generation: ENABLED (stereo 24kHz)")

    # Resolve and report internal-audio state.  Validation: --internal-audio off
    # with --generate-audio is incoherent — the audio branch produces what the
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
        print(f"Latent sidecar: ENABLED")
    if save_text_embeddings:
        print(f"Text conditioning sidecar: ENABLED")
    if save_run_log:
        print(f"Run log sidecar: ENABLED")
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
        print(f"Low memory mode: ENABLED (sequential CFG, aggressive eval)")
    if fast_mode:
        print(f"Fast mode: ENABLED (no intermediate evals)")
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
        print(f"Text encoder: DUMMY (testing mode)")

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
        print(f"  Using enhanced prompt for generation")
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
            print(f"\n  To download Gemma 3 12B:")
            print(f"    python scripts/download_gemma.py")
            print(f"\n  Or use --no-gemma flag to use dummy embeddings for testing")
            return

        if use_av_encoder:
            # Encode both prompt AND negative prompt in one Gemma load, then
            # free Gemma before loading the AV connector to reduce peak memory.
            #
            # Skip the negative encoding entirely for distilled two-stage
            # (which doesn't pass negative_encoding through the pipeline) when
            # no text-embedding sidecar is requested.  Saves ~5-7s/run by
            # avoiding one Gemma + one AV-connector forward pass.
            # Re-enable with LTX_ENCODE_UNUSED_NEGATIVE=1 for debugging.
            skip_negative = (
                distilled_two_stage_requested
                and not save_text_embeddings
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
                # Negative not used by distilled two-stage; leave as None.
                null_encoding, null_audio_encoding, null_mask = None, None, None
                print("  Encoded positive prompt only with Gemma 3 "
                      "(distilled two-stage doesn't use negative)")
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
                print(f"  Encoded both prompts with Gemma 3 (AudioVideo, single load)")
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
            print(f"  Encoded with Gemma 3")
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
        if text_encoding is None or null_encoding is None or text_mask is None or null_mask is None:
            print("  WARNING: Text conditioning sidecar requested, but text encodings are unavailable")
        else:
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
            # Audio pretranspose mirrors video by default for AV models — the
            # audio modules see the same per-step dispatch pattern, so the
            # same weight.T contiguity helps.  Disable with
            # LTX_DISABLE_AUDIO_PRETRANSPOSE=1.
            if os.environ.get("LTX_DISABLE_AUDIO_PRETRANSPOSE"):
                _audio_ff_layout_specs = ()
                _audio_attn_layout_specs = ()
            else:
                _audio_ff_layout_specs = video_ff_layout_specs
                _audio_attn_layout_specs = video_attn_layout_specs
            # AdaLN pretranspose is cache-integrated (no per-load RAM spike),
            # but measured neutral-to-slight-regression at small T because the
            # per-step adaln matmul count is low (8) and batch is tiny.  Off
            # by default; opt-in with LTX_ADALN_PRETRANSPOSE=1.
            _adaln_pretranspose = bool(os.environ.get("LTX_ADALN_PRETRANSPOSE"))
            model = load_av_transformer(
                transformer_weights_path, num_layers=48, compute_dtype=compute_dtype,
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
            )
        else:
            model = None
            print("  Skipping model load (placeholder mode)")
    else:
        print("\n[2/5] Loading transformer...")
        if not use_placeholder and transformer_weights_path:
            velocity_model = load_transformer(
                transformer_weights_path,
                num_layers=48,
                compute_dtype=compute_dtype,
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
            )

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

    # Apply LoRA if provided
    if lora_path and model is not None:
        print(f"\n  Applying LoRA from {lora_path} (strength={lora_strength})")
        lora_config = LoRAConfig(path=lora_path, strength=lora_strength)

        # Get target model (handle X0Model wrapper)
        if hasattr(model, 'velocity_model'):
            target_model = model.velocity_model
        else:
            target_model = model

        # Fuse LoRA weights into model
        flat_params = dict(tree_flatten(target_model.parameters()))
        fused_weights = fuse_lora_into_weights(flat_params, [lora_config])
        target_model.load_weights(list(fused_weights.items()))
        mx.eval(target_model.parameters())
        print(f"  LoRA applied successfully")

    # Whether to use CFG
    # Distilled models (LTX-2 distilled) are trained without CFG and produce artifacts if CFG > 1.0
    # HOWEVER: Two-stage pipeline specifically uses CFG in Stage 1 (at low res), so we allow it there.
    if model_variant == "distilled" and cfg_scale > 1.2 and pipeline_type != "two-stage":
        print(f"  WARNING: Distilled model requires CFG=1.0 (no guidance). You requested {cfg_scale}.")
        print(f"  Forcing CFG=1.0 to prevent visual artifacts.")
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
        print(f"  APG guidance enabled (replaces standard CFG)")

    # Create STG guider if enabled
    stg_guider = None
    if stg_scale > 0:
        stg_guider = STGGuider(scale=stg_scale)
        print(f"  STG guidance enabled (scale={stg_scale})")
    timings.mark("guidance setup")

    # Load VAE decoder.
    vae_decoder = None
    if not skip_vae:
        print(f"\n[3/5] Loading VAE decoder...")
        # Read VAE config from checkpoint to build correct architecture
        vae_config = get_vae_config(config_weights_path) if config_weights_path else {}
        decoder_blocks = vae_config.get("decoder_blocks", None)
        base_channels = vae_config.get("decoder_base_channels", 128)
        timestep_cond = vae_config.get("timestep_conditioning", True)
        if decoder_blocks:
            print(f"  VAE config: {len(decoder_blocks)} blocks, base_ch={base_channels}, timestep={timestep_cond}")

        if vae_decoder_backend == "native-conv3d":
            vae_decoder = NativeConv3dVideoDecoder(
                decoder_blocks=decoder_blocks,
                base_channels=base_channels,
                timestep_conditioning=timestep_cond,
                compute_dtype=compute_dtype,
                spatial_padding_mode=vae_spatial_padding,
            )
        elif vae_decoder_backend == "simple":
            vae_decoder = SimpleVideoDecoder(
                decoder_blocks=decoder_blocks,
                base_channels=base_channels,
                timestep_conditioning=timestep_cond,
                compute_dtype=compute_dtype,
                spatial_padding_mode=vae_spatial_padding,
            )
        else:
            raise ValueError(f"Unsupported VAE decoder backend: {vae_decoder_backend}")
        if video_vae_load_path and not use_placeholder:
            if vae_decoder_backend == "native-conv3d":
                load_native_vae_decoder_weights(vae_decoder, video_vae_load_path)
            else:
                load_vae_decoder_weights(vae_decoder, video_vae_load_path)
        elif use_placeholder:
            print("  Skipping weights load (placeholder)")
    else:
        print("\n[3/5] VAE decoder skipped by user")
    timings.mark("vae decoder load")

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
                        return mx.zeros((1))
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
            print(f"  WARNING: Two-stage pipeline requires resolution divisible by 64.")
            print(f"  Adjusting resolution from {height}x{width} to {new_height}x{new_width}")
            height = new_height
            width = new_width

        print("\n=== Using Two-Stage Pipeline ===")
        print(f"  Stage 1: {steps_stage1} steps at {height//2}x{width//2} with CFG {cfg_stage1 or cfg_scale}")
        if guidance_rescale > 0:
            print(f"  Guidance rescale: {guidance_rescale}")
        print(f"  Stage 2: 3 steps at {height}x{width} (distilled refinement)")
        if generate_audio:
            print(f"  Audio generation: ENABLED")

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
        video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
        if not use_placeholder:
            load_vae_encoder_weights(video_encoder, video_vae_load_path)
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
        if distilled_lora:
            print(f"  Distilled LoRA: {distilled_lora} (scale {distilled_lora_scale})")
            distilled_lora_config = LoRAConfig(path=distilled_lora, strength=distilled_lora_scale)
        elif pipeline_type == "two-stage":
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
        print(f"\n[5/5] Running two-stage generation...")
        video, audio_waveform = pipeline(
            positive_encoding=text_encoding,
            negative_encoding=null_encoding,
            config=config,
            images=images,
            positive_audio_encoding=text_audio_encoding,
            negative_audio_encoding=null_audio_encoding,
        )

        # Convert to frames list for save_video
        # decode_latent returns (T, H, W, C) in uint8, so just convert to numpy list
        video_np = np.array(video)  # (T, H, W, C)
        frames = [video_np[t] for t in range(video_np.shape[0])]
        print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")

        if audio_waveform is not None:
            print(f"  Generated audio: {audio_waveform.shape}")

        # Save video
        print(f"\nSaving video to {output_path}...")
        if audio_waveform is not None:
            save_video_with_audio(frames, audio_waveform, output_path, fps=output_fps, speed=output_speed, audio_sample_rate=audio_sample_rate)
        else:
            save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
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
        video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
        if video_vae_load_path and not use_placeholder:
            load_vae_encoder_weights(video_encoder, video_vae_load_path)
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

        # Get base transformer weights for restoration after stage 1
        if hasattr(model, 'velocity_model'):
            base_weights = dict(tree_flatten(model.velocity_model.parameters()))
        else:
            base_weights = dict(tree_flatten(model.parameters()))

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
            base_transformer_weights=base_weights,
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
        print(f"\n[5/5] Running IC-LoRA generation...")
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
        save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
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
        video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
        if video_vae_load_path and not use_placeholder:
            load_vae_encoder_weights(video_encoder, video_vae_load_path)
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
        save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
        return

    # === AUDIO-VIDEO PIPELINE ===
    # Use OneStagePipeline for joint audio-video generation
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

        if vae_decoder is None and not use_placeholder:
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
        if images or distilled_two_stage:
            print("[3.5/5] Loading VAE encoder...")
            video_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
            if video_vae_load_path and not use_placeholder:
                load_vae_encoder_weights(video_encoder, video_vae_load_path)
            else:
                print("  Skipping weights load (placeholder)")
        else:
            print("[3.5/5] VAE encoder skipped (no image conditioning)")

        spatial_upscaler = None
        if distilled_two_stage:
            if model_variant != "distilled":
                raise ValueError("Distilled two-stage requires --model-variant distilled")
            if not spatial_upscaler_weights:
                raise ValueError("Distilled two-stage requires --spatial-upscaler-weights")
            print("[3.6/5] Loading spatial upscaler...")
            spatial_upscaler = SpatialUpscaler()
            if not use_placeholder:
                load_spatial_upscaler_weights(spatial_upscaler, spatial_upscaler_weights)
            else:
                print("  Skipping weights load (placeholder)")

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

        # Create one-stage pipeline with audio support
        print("\n[4/5] Creating audio-video pipeline...")
        av_pipeline = OneStagePipeline(
            transformer=model,
            video_encoder=video_encoder,
            video_decoder=vae_decoder,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
        )

        del model  # pipeline now holds the only reference; del self.transformer in one_stage.py can actually free it
        model = None
        gc.collect()

        # Create config with audio enabled
        # LTX-2.3 reference: video_cfg=3.0, audio_cfg=7.0, rescale=0.7
        #
        # Internal-audio resolution.  By default V2/AV models always run the
        # internal audio branch (audio self-attn + A2V/V2A cross-modal) even
        # when --generate-audio is off, and discard the result — wasted compute.
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
        av_config = OneStageCFGConfig(
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

            stage_progress = None
            active_stage = None
            stage_labels = {
                "stage_1": "Stage 1 denoising",
                "stage_2": "Stage 2 denoising",
            }

            def stage_progress_callback(stage_name: str, step: int, total: int):
                nonlocal stage_progress, active_stage
                if active_stage != stage_name:
                    if stage_progress is not None:
                        stage_progress.finish()
                    active_stage = stage_name
                    stage_progress = DenoiseProgress(
                        label=stage_labels.get(stage_name, stage_name),
                        total=total,
                    )
                    stage_progress.start()
                stage_progress.update(step, total)
                if step >= total:
                    stage_progress.finish()

            def distilled_progress_message(message: str):
                if stage_progress is not None:
                    stage_progress.log(message)
                else:
                    print(message, flush=True)

            video, audio_waveform = av_pipeline.generate_distilled_two_stage(
                positive_encoding=text_encoding,
                config=av_config,
                spatial_upscaler=spatial_upscaler,
                images=images,
                stage_callback=stage_progress_callback,
                progress_message=distilled_progress_message,
                positive_audio_encoding=text_audio_encoding,
                latent_save_path=latent_sidecar_path(output_path) if save_latents else None,
            )
            if stage_progress is not None:
                stage_progress.finish()
        else:
            print(f"\n[5/5] Running audio-video generation ({num_steps} steps)...")
            denoise_progress = DenoiseProgress(total=num_steps)
            denoise_progress.start()

            def progress_callback(step: int, total: int):
                denoise_progress.update(step, total)
                if step >= total:
                    denoise_progress.finish()

            video, audio_waveform = av_pipeline(
                positive_encoding=text_encoding,
                negative_encoding=null_encoding,
                config=av_config,
                images=images,
                callback=progress_callback,
                positive_audio_encoding=text_audio_encoding,
                negative_audio_encoding=null_audio_encoding,
                latent_save_path=latent_sidecar_path(output_path) if save_latents else None,
            )
            denoise_progress.finish()
        pipeline_timings = getattr(av_pipeline, "last_timing_sections", None)
        if pipeline_timings:
            timings.extend(pipeline_timings)
        else:
            timings.mark("generation + decode")

        # Convert to frames list for save_video
        video_np = np.array(video)
        print(f"  Raw video shape: {video_np.shape}, dtype: {video_np.dtype}")
        # Squeeze any singleton dimensions
        video_np = np.squeeze(video_np)
        # Handle (C, T, H, W) format — C=3 is always smallest dim
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
        if audio_waveform is not None:
            save_video_with_audio(frames, audio_waveform, output_path, fps=output_fps, speed=output_speed, audio_sample_rate=audio_sample_rate)
        else:
            save_video(frames, output_path, fps=output_fps, speed=output_speed)
        print(f"Done! Video saved to {output_path}")
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
        sigmas = mx.array(np.linspace(1.0, 0.0, num_steps + 1).tolist())
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
    np.savez(latent_path, latent=np.array(latent))
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

        # Convert to frames list for save_video
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
    save_video(frames, output_path, fps=output_fps, speed=output_speed)
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


def save_video(frames: list, output_path: str, fps: float = NATIVE_FPS, speed: float = 1.0):
    """Save frames as video using ffmpeg with optional speed adjustment.

    Args:
        frames: List of frame arrays (H, W, C) in uint8.
        output_path: Output video file path.
        fps: Generation and output frame rate.
        speed: Playback speed multiplier (0.5=slow-mo, 1.0=normal, 2.0=fast).
    """
    import subprocess
    import tempfile
    from PIL import Image

    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as images with progress
        print("  Writing frames...")
        if HAS_TQDM:
            iterator = tqdm(enumerate(frames), desc="  Saving frames", total=len(frames), ncols=80, ascii=True, mininterval=1.0, miniters=10)
        else:
            iterator = enumerate(frames)

        for i, frame in iterator:
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        # Build ffmpeg filter chain
        filters = []

        # Speed adjustment
        # setpts: lower value = faster, higher value = slower
        if speed != 1.0:
            # speed=2.0 means 2x faster, so PTS should be halved
            pts_multiplier = 1.0 / speed
            filters.append(f"setpts={pts_multiplier}*PTS")
        # Build ffmpeg command
        print("\n  Encoding video...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),  # Input frame cadence matches generation fps
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
        ]

        # Add filter chain if needed
        if filters:
            filter_str = ",".join(filters)
            cmd.extend(["-vf", filter_str])
            if speed != 1.0:
                print(f"  Applying speed: {speed}x")

        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-loglevel", "error",
            output_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def save_video_with_audio(
    frames: list,
    audio_waveform: mx.array,
    output_path: str,
    fps: float = NATIVE_FPS,
    speed: float = 1.0,
    audio_sample_rate: int = 24000,
):
    """Save frames as video with audio using ffmpeg with optional speed adjustment.

    Args:
        frames: List of frame arrays (H, W, C) in uint8.
        audio_waveform: Audio waveform tensor (B, 2, samples).
        output_path: Output video file path.
        fps: Generation and output frame rate.
        speed: Playback speed multiplier (0.5=slow-mo, 1.0=normal, 2.0=fast).
        audio_sample_rate: Audio sample rate in Hz.
    """
    import os
    import shutil
    import subprocess
    import tempfile
    import wave
    from PIL import Image

    # Create temp directory for frames and audio
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as images with progress
        print("  Writing frames...")
        if HAS_TQDM:
            iterator = tqdm(enumerate(frames), desc="  Saving frames", total=len(frames), ncols=80, ascii=True, mininterval=1.0, miniters=10)
        else:
            iterator = enumerate(frames)

        for i, frame in iterator:
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        # Save audio as WAV file
        audio_path = os.path.join(tmpdir, "audio.wav")
        print("\n  Writing audio...")

        # audio_waveform shape: (B, 2, samples) - stereo
        audio_np = np.array(audio_waveform[0])  # (2, samples)

        # Convert from float [-1, 1] to int16
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

        # Interleave stereo channels: (2, samples) -> (samples, 2) -> flat
        audio_interleaved = audio_int16.T  # (samples, 2)
        audio_flat = audio_interleaved.flatten()

        # Write WAV file
        with wave.open(audio_path, 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(audio_sample_rate)
            wav_file.writeframes(audio_flat.tobytes())

        # Save a sidecar WAV beside the MP4 so export compression can be compared directly.
        sidecar_wav_path = os.path.splitext(output_path)[0] + ".wav"
        shutil.copy2(audio_path, sidecar_wav_path)
        print(f"    Sidecar WAV: {sidecar_wav_path}")

        print(f"    Audio: {len(audio_flat) // 2} samples, {len(audio_flat) // 2 / audio_sample_rate:.2f}s")

        # Build video filter chain
        video_filters = []

        # Speed adjustment for video
        if speed != 1.0:
            pts_multiplier = 1.0 / speed
            video_filters.append(f"setpts={pts_multiplier}*PTS")

        # Build audio filter chain for speed adjustment
        # atempo filter range is 0.5-2.0, so chain multiple for extreme speeds
        audio_filters = []
        if speed != 1.0:
            remaining_speed = speed
            while remaining_speed > 2.0:
                audio_filters.append("atempo=2.0")
                remaining_speed /= 2.0
            while remaining_speed < 0.5:
                audio_filters.append("atempo=0.5")
                remaining_speed /= 0.5
            if remaining_speed != 1.0:
                audio_filters.append(f"atempo={remaining_speed}")

        # Build ffmpeg command
        print("\n  Encoding video with audio...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),  # Input frame cadence matches generation fps
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-i", audio_path,
        ]

        # Add video filter chain if needed
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])
            if speed != 1.0:
                print(f"  Applying speed: {speed}x")

        # Add audio filter chain if needed
        if audio_filters:
            cmd.extend(["-af", ",".join(audio_filters)])

        cmd.extend([
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "320k",
            "-ar", str(audio_sample_rate),
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-shortest",  # Use shortest duration (video or audio)
            "-loglevel", "error",
            output_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")


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
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (0.5=slow-mo, 1.0=normal, 2.0=fast)")
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
            "$LTX_MLX_WEIGHTS_CACHE_DIR, then /Users/Shared/huggingface/mlx/"
            "LTX-2-MLX-cache when available. The older "
            "$LTX_MLX_TRANSFORMER_CACHE_DIR is still honored."
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
        "--vae-decoder",
        choices=["simple", "native-conv3d"],
        default="native-conv3d",
        help=(
            "Video VAE decoder backend. native-conv3d is the default lower-memory "
            "MLX Conv3d decoder; simple keeps the PyTorch-layout slice-conv baseline "
            "for A/B testing."
        ),
    )
    parser.add_argument(
        "--vae-spatial-padding",
        choices=["reflect", "zero"],
        default="zero",
        help=(
            "Spatial padding mode for VAE decoder convolutions. zero is the default "
            "boundary-flicker mitigation; reflect keeps the released Lightricks "
            "boundary behavior for A/B testing."
        ),
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
            "project_in:pretranspose,project_out:pretranspose. Use 'off' to disable."
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
            "Same-math video-output attention layout transform. Default is "
            "to_out:pretranspose. Use 'off' to disable."
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
        default=None,
        help="Path to LoRA weights (.safetensors)"
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="LoRA strength (-2.0 to 2.0, default 1.0)"
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
        "--save-all-sidecars",
        "--save-debug-sidecars",
        dest="save_all_sidecars",
        action="store_true",
        help=(
            "Enable all reproducibility/debug sidecars: latents, text "
            "conditioning, and run metadata"
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

    generate_video(
        distilled_lora=args.distilled_lora,
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
        vae_decoder_backend=args.vae_decoder,
        vae_spatial_padding=args.vae_spatial_padding,
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
        video_ff_layout_specs=args.video_ff_layout,
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
        lora_path=args.lora,
        lora_strength=args.lora_strength,
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
        # GE (Gradient Estimation) parameter
        ge_gamma=args.ge_gamma,
        # Output FPS and speed
        output_fps=args.fps,
        output_speed=args.speed,
        # IC-LoRA and Keyframe Interpolation
        keyframes=args.keyframe,
        ic_lora_weights=args.ic_lora_weights,
    )


if __name__ == "__main__":
    main()
