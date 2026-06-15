"""LoRA (Low-Rank Adaptation) loading and fusion for LTX-2 models."""

import gc
import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


# A LoRA target can be filtered by several overlapping tag families; excluding
# any tag drops every target that carries it. Each target carries one tag from
# each applicable family:
#   branch (coarse modality)  -- video / audio / cross
#   module type (coarse role) -- attn / gate / ff
#   module (exact block)      -- attn1, attn2, audio_attn1, audio_attn2,
#                                video_to_audio_attn, audio_to_video_attn,
#                                ff, audio_ff
#   projection (exact linear) -- to_q, to_k, to_v, to_out, to_gate_logits,
#                                project_in, project_out
#   control path              -- adaln, prompt_adaln, scale_shift,
#                                prompt_scale_shift, gate_adaln, av_ca,
#                                cross_control, distill_control
# The branch/type aliases are coarse shortcuts; the module/projection tags give
# full granularity (e.g. exclude just `audio_to_video_attn` to drop the
# lip-sync direction, or `attn2` to revert prompt-conditioning to stock).
# "ff" means all feed-forward (it is both the type and the video-ff module
# name); use "audio_ff" to target audio feed-forward alone. Classification is
# by the MLX weight key, whose module/projection names survive conversion.
_LORA_BRANCH_TAGS = frozenset({"video", "audio", "cross"})
_LORA_TYPE_TAGS = frozenset({"attn", "gate", "ff"})
_LORA_MODULE_TAGS = frozenset({
    "attn1", "attn2", "audio_attn1", "audio_attn2",
    "video_to_audio_attn", "audio_to_video_attn", "ff", "audio_ff",
})
_LORA_PROJ_TAGS = frozenset({
    "to_q", "to_k", "to_v", "to_out", "to_gate_logits",
    "project_in", "project_out",
})
_LORA_CONTROL_TAGS = frozenset({
    "adaln", "prompt_adaln", "scale_shift", "prompt_scale_shift",
    "gate_adaln", "av_ca", "cross_control", "distill_control",
})
_LORA_CROSS_CONTROL_TAGS = frozenset({
    "adaln", "attn2", "audio_attn2", "cross",
})
_LORA_DISTILL_CONTROL_TAGS = frozenset({
    "cross", "gate", "adaln", "prompt_adaln", "scale_shift",
    "prompt_scale_shift", "gate_adaln", "av_ca",
})
_LORA_CATEGORIES = (
    _LORA_BRANCH_TAGS
    | _LORA_TYPE_TAGS
    | _LORA_MODULE_TAGS
    | _LORA_PROJ_TAGS
    | _LORA_CONTROL_TAGS
)
_LORA_FF_PRETRANSPOSE_SLOTS = (
    ("ff", "_project_in_weight_t", "ff.project_in.proj.weight"),
    ("ff", "_project_out_weight_t", "ff.project_out.weight"),
    ("audio_ff", "_project_in_weight_t", "audio_ff.project_in.proj.weight"),
    ("audio_ff", "_project_out_weight_t", "audio_ff.project_out.weight"),
)
_LORA_RESTORE_CACHE_ATTR = "_lora_restore_cache_source"
_LORA_LOAD_MODES = frozenset({"auto", "fast", "stream"})
_LORA_FAST_LOAD_SAFETY_FACTOR = 1.25
_LORA_FUSE_RANGE_GUARD_ENV = "LTX_LORA_FUSE_RANGE_GUARD"


def _lora_key_categories(mlx_key: str) -> set:
    cats = set()
    # coarse branch
    if "video_to_audio" in mlx_key or "audio_to_video" in mlx_key:
        cats.add("cross")
    elif mlx_key.startswith("audio_") or ".audio_" in mlx_key or "audio_" in mlx_key:
        cats.add("audio")
    else:
        cats.add("video")
    # coarse module type
    if "to_gate_logits" in mlx_key:
        cats.add("gate")
    elif any(t in mlx_key for t in ("to_q", "to_k", "to_v", "to_out")):
        cats.add("attn")
    elif "ff" in mlx_key:
        cats.add("ff")
    # Top-level V2 control paths. These are not transformer block projections,
    # so they need direct tags for targeted LoRA suppression.
    if "av_ca" in mlx_key or "a2v" in mlx_key or "v2a" in mlx_key:
        cats.add("av_ca")
    if "adaln" in mlx_key:
        cats.add("adaln")
    if "prompt_adaln" in mlx_key:
        cats.add("prompt_adaln")
        cats.add("prompt_scale_shift")
    if "scale_shift" in mlx_key:
        cats.add("scale_shift")
    if "prompt_scale_shift" in mlx_key:
        cats.add("prompt_scale_shift")
    if "gate_adaln" in mlx_key:
        cats.add("gate_adaln")
    # exact module + projection (finest granularity)
    parts = mlx_key.split(".")
    if "transformer_blocks" in parts:
        i = parts.index("transformer_blocks")
        seg = parts[i + 2:]  # segments after the block index
        if seg and seg[0] in _LORA_MODULE_TAGS:
            cats.add(seg[0])
        for p in seg:
            if p in _LORA_PROJ_TAGS:
                cats.add(p)
                break
    if cats & _LORA_CROSS_CONTROL_TAGS:
        cats.add("cross_control")
    if cats & _LORA_DISTILL_CONTROL_TAGS:
        cats.add("distill_control")
    return cats


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter.

    ``strength`` is the default/global adapter strength. Two-stage pipelines can
    optionally use ``stage_1_strength`` and/or ``stage_2_strength`` to override
    that default for a specific denoise stage.

    ``exclude`` lists categories (see ``_LORA_CATEGORIES``) whose targets are
    dropped from this adapter's fusion -- e.g. ``("audio", "cross")`` applies
    only the video-branch style and leaves the audio path stock. Use
    ``("cross_control",)`` to drop prompt/audio-video cross-conditioning
    paths: attn2, audio_attn2, cross bridges, and AdaLN. Use
    ``("distill_control",)`` to drop the official-distillation-style control
    paths: cross bridges, gate logits, AdaLN, scale-shift, prompt scale-shift,
    and AV cross-conditioning AdaLN.
    """

    path: str
    strength: float = 1.0
    stage_1_strength: Optional[float] = None
    stage_2_strength: Optional[float] = None
    exclude: Tuple[str, ...] = ()

    def __post_init__(self):
        for name, value in (
            ("strength", self.strength),
            ("stage_1_strength", self.stage_1_strength),
            ("stage_2_strength", self.stage_2_strength),
        ):
            if value is not None and not -2.0 <= value <= 2.0:
                raise ValueError(
                    f"LoRA {name} should be between -2.0 and 2.0, got {value}"
                )
        bad = sorted(set(self.exclude) - _LORA_CATEGORIES)
        if bad:
            raise ValueError(
                f"Unknown LoRA exclude categories {bad}; "
                f"valid: {sorted(_LORA_CATEGORIES)}"
            )

    def has_stage_strengths(self) -> bool:
        return self.stage_1_strength is not None or self.stage_2_strength is not None

    def strength_for_stage(self, stage: int) -> float:
        if stage == 1:
            return self.strength if self.stage_1_strength is None else self.stage_1_strength
        if stage == 2:
            return self.strength if self.stage_2_strength is None else self.stage_2_strength
        raise ValueError(f"LoRA stage must be 1 or 2, got {stage}")

    def with_strength(self, strength: float) -> "LoRAConfig":
        return LoRAConfig(path=self.path, strength=strength, exclude=self.exclude)


def lora_configs_have_stage_strengths(lora_configs: Optional[List[LoRAConfig]]) -> bool:
    return any(cfg.has_stage_strengths() for cfg in (lora_configs or ()))


def lora_configs_for_stage(
    lora_configs: Optional[List[LoRAConfig]],
    stage: int,
) -> List[LoRAConfig]:
    return [
        cfg.with_strength(cfg.strength_for_stage(stage))
        for cfg in (lora_configs or ())
        if cfg.strength_for_stage(stage) != 0.0
    ]


def lora_configs_for_stage_delta(
    lora_configs: Optional[List[LoRAConfig]],
    *,
    from_stage: int,
    to_stage: int,
) -> List[LoRAConfig]:
    out: List[LoRAConfig] = []
    for cfg in lora_configs or ():
        delta = cfg.strength_for_stage(to_stage) - cfg.strength_for_stage(from_stage)
        if delta != 0.0:
            out.append(cfg.with_strength(delta))
    return out


def format_lora_stage_scale_lines(
    lora_configs: Optional[List[LoRAConfig]],
    stage: int,
    *,
    from_stage: Optional[int] = None,
    include_unchanged: bool = False,
) -> List[str]:
    """Return human-readable stage LoRA total/change lines."""
    lines: List[str] = []
    for cfg in lora_configs or ():
        total = cfg.strength_for_stage(stage)
        name = Path(cfg.path).name
        if from_stage is None:
            if total == 0.0 and not include_unchanged:
                continue
            lines.append(f"    {name}: total={total:.4f}")
            continue
        change = total - cfg.strength_for_stage(from_stage)
        if change == 0.0 and not include_unchanged:
            continue
        lines.append(f"    {name}: total={total:.4f}, change={change:+.4f}")
    return lines


def load_lora_weights(path: str) -> Dict[str, mx.array]:
    """
    Load LoRA weights from a safetensors file.

    Args:
        path: Path to the LoRA weights file.

    Returns:
        Dictionary mapping weight names to arrays.
    """
    return dict(mx.load(path))


class _DictLoRATensorSource:
    def __init__(self, weights: Dict[str, mx.array]):
        self.weights = weights

    def __iter__(self):
        return iter(self.weights)

    def __contains__(self, key: str) -> bool:
        return key in self.weights

    def keys(self):
        return self.weights.keys()

    def load(self, key: str) -> mx.array:
        return self.weights[key]

    def close(self) -> None:
        self.weights.clear()


class _SafetensorsLoRATensorSource:
    _DTYPES = {
        "F16": np.dtype("<f2"),
        "F32": np.dtype("<f4"),
        "F64": np.dtype("<f8"),
        "I8": np.dtype("i1"),
        "I16": np.dtype("<i2"),
        "I32": np.dtype("<i4"),
        "I64": np.dtype("<i8"),
        "U8": np.dtype("u1"),
        "U16": np.dtype("<u2"),
        "U32": np.dtype("<u4"),
        "U64": np.dtype("<u8"),
    }

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(self._mmap) < 8:
                raise ValueError("safetensors file is too small")
            header_len = struct.unpack("<Q", self._mmap[:8])[0]
            header_start = 8
            header_end = header_start + header_len
            if header_end > len(self._mmap):
                raise ValueError("safetensors header extends past EOF")
            header = json.loads(self._mmap[header_start:header_end])
            self._data_start = header_end
            self._tensors = {
                key: value
                for key, value in header.items()
                if key != "__metadata__"
            }
        except Exception:
            self.close()
            raise

    def __iter__(self):
        return iter(self._tensors)

    def __contains__(self, key: str) -> bool:
        return key in self._tensors

    def keys(self):
        return self._tensors.keys()

    def load(self, key: str) -> mx.array:
        spec = self._tensors[key]
        dtype_name = spec["dtype"]
        shape = tuple(int(x) for x in spec["shape"])
        start, end = spec["data_offsets"]
        start += self._data_start
        end += self._data_start
        view = memoryview(self._mmap)[start:end]
        try:
            if dtype_name == "BF16":
                np_view = np.frombuffer(view, dtype=np.dtype("<u2")).reshape(shape)
                bits = mx.array(np_view, dtype=mx.uint16)
                arr = bits.view(mx.bfloat16)
                mx.eval(arr)
                del bits, np_view
                return arr
            np_dtype = self._DTYPES.get(dtype_name)
            if np_dtype is None:
                raise TypeError(f"Unsupported safetensors dtype {dtype_name}")
            np_view = np.frombuffer(view, dtype=np_dtype).reshape(shape)
            arr = mx.array(np_view)
            mx.eval(arr)
            del np_view
            return arr
        finally:
            view.release()

    def close(self) -> None:
        mmap_obj = getattr(self, "_mmap", None)
        if mmap_obj is not None:
            self._mmap = None
            mmap_obj.close()
        file_obj = getattr(self, "_file", None)
        if file_obj is not None:
            self._file = None
            file_obj.close()


def _estimate_lora_bulk_load_bytes(lora_configs: List[LoRAConfig]) -> int:
    total = 0
    for cfg in lora_configs:
        try:
            total += Path(cfg.path).stat().st_size
        except OSError:
            # Unknown size: make auto choose the conservative streaming path.
            return 2**63 - 1
    return total


def _lora_auto_load_mode(lora_configs: List[LoRAConfig]):
    from ..model.video_vae.tiling import (
        default_vae_decode_budget_gb,
        detect_system_memory_gb,
    )

    budget_gb = default_vae_decode_budget_gb(detect_system_memory_gb())
    estimate_bytes = _estimate_lora_bulk_load_bytes(lora_configs)
    estimate_gb = (estimate_bytes * _LORA_FAST_LOAD_SAFETY_FACTOR) / (1000**3)
    return ("fast" if estimate_gb <= budget_gb else "stream"), estimate_gb, budget_gb


def _resolve_lora_load_mode(lora_configs: List[LoRAConfig], requested: str):
    if requested not in _LORA_LOAD_MODES:
        raise ValueError(
            f"lora_load_mode must be one of {sorted(_LORA_LOAD_MODES)}, got {requested!r}"
        )
    if requested != "auto":
        return requested, None, None
    return _lora_auto_load_mode(lora_configs)


def _open_lora_tensor_source(path: str, load_mode: str):
    if load_mode == "fast":
        return _DictLoRATensorSource(load_lora_weights(path))
    try:
        return _SafetensorsLoRATensorSource(path)
    except Exception:
        return _DictLoRATensorSource(load_lora_weights(path))


def _pretransposed_lora_slots(model):
    blocks = getattr(model, "transformer_blocks", []) or []
    for idx, block in enumerate(blocks):
        for module_name, attr, suffix in _LORA_FF_PRETRANSPOSE_SLOTS:
            module = getattr(block, module_name, None)
            if module is None:
                continue
            value = getattr(module, attr, None)
            if value is not None:
                yield idx, module_name, attr, suffix, value


def snapshot_lora_base_weights(model) -> dict:
    """Capture the current model weights before a temporary LoRA fuse."""
    target = getattr(model, "velocity_model", model)
    cache_source = getattr(target, _LORA_RESTORE_CACHE_ATTR, None)
    if cache_source and cache_source.get("valid"):
        return {
            "kind": "cache",
            "source": dict(cache_source),
        }
    return {
        "kind": "memory",
        "parameters": list(tree_flatten(target.parameters())),
        "pretransposed": list(_pretransposed_lora_slots(target)),
    }


def restore_lora_base_weights(model, weights) -> None:
    """Restore a snapshot produced by ``snapshot_lora_base_weights``."""
    target = getattr(model, "velocity_model", model)
    if isinstance(weights, dict) and weights.get("kind") == "cache":
        from .transformer_cache import load_transformer_cache

        source = dict(weights["source"])
        persistent_loras = tuple(source.get("persistent_loras", ()))
        load_transformer_cache(
            target,
            source["cache_path"],
            transformer_cache_quantize=source["transformer_cache_quantize"],
            video_ff_quantize_specs=source["video_ff_quantize_specs"],
            video_ff_quantize_group_size=source["video_ff_quantize_group_size"],
            video_ff_quantize_bits=source["video_ff_quantize_bits"],
        )
        source["persistent_loras"] = ()
        source["valid"] = True
        setattr(target, _LORA_RESTORE_CACHE_ATTR, source)
        for entry in persistent_loras:
            fuse_loras_into_model(
                target,
                list(entry["configs"]),
                include_audio=entry["include_audio"],
                min_coverage=entry["min_coverage"],
                allow_partial=entry["allow_partial"],
                verbose=False,
                track_for_restore=True,
            )
        return
    if isinstance(weights, dict) and "parameters" in weights:
        items = list(weights["parameters"])
        pretransposed = list(weights.get("pretransposed", ()))
    else:
        items = list(weights.items()) if isinstance(weights, dict) else list(weights)
        pretransposed = []
    target.load_weights(items)
    restored_private = []
    blocks = getattr(target, "transformer_blocks", []) or []
    for idx, module_name, attr, _suffix, value in pretransposed:
        if idx >= len(blocks):
            continue
        module = getattr(blocks[idx], module_name, None)
        if module is None:
            continue
        setattr(module, attr, value)
        restored_private.append(value)
    mx.eval(target.parameters())
    if restored_private:
        mx.eval(*restored_private)


def _invalidate_lora_restore_cache(model) -> None:
    source = getattr(model, _LORA_RESTORE_CACHE_ATTR, None)
    if source:
        source = dict(source)
        source["valid"] = False
        setattr(model, _LORA_RESTORE_CACHE_ATTR, source)


def _record_persistent_loras(
    model,
    lora_configs: List[LoRAConfig],
    *,
    include_audio: bool,
    min_coverage: float,
    allow_partial: bool,
) -> None:
    source = getattr(model, _LORA_RESTORE_CACHE_ATTR, None)
    if not source:
        return
    source = dict(source)
    entry = {
        "configs": tuple(lora_configs),
        "include_audio": include_audio,
        "min_coverage": min_coverage,
        "allow_partial": allow_partial,
    }
    source["persistent_loras"] = tuple(source.get("persistent_loras", ())) + (entry,)
    source["valid"] = True
    setattr(model, _LORA_RESTORE_CACHE_ATTR, source)


# Max finite magnitude per narrow cache dtype. This check is sync-heavy, so it
# only runs when LTX_LORA_FUSE_RANGE_GUARD=1 is set.
_DTYPE_MAX_FINITE = {
    mx.float16: 65504.0,
}


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


def _lora_fuse_range_guard_enabled() -> bool:
    return _env_truthy(_LORA_FUSE_RANGE_GUARD_ENV)


def _guard_fused_range(fused_f32: mx.array, target_dtype, mlx_key: str) -> None:
    """Raise if the float32 fused weight would overflow a narrow target dtype."""
    limit = _DTYPE_MAX_FINITE.get(target_dtype)
    if limit is None:
        return
    peak = float(mx.max(mx.abs(fused_f32)).item())
    if not math.isfinite(peak):
        raise ValueError(
            f"LoRA fusion produced a non-finite weight at '{mlx_key}' -- the "
            f"base weight or the delta already overflowed float32. Lower "
            f"--lora-strength or check the adapter."
        )
    if peak > limit:
        raise ValueError(
            f"LoRA fusion overflows {target_dtype} at '{mlx_key}': fused "
            f"|max|={peak:.4g} exceeds the dtype ceiling {limit:.4g}. Cache the "
            f"transformer in a wider dtype (--transformer-dtype bf16) or lower "
            f"--lora-strength."
        )


def _lora_base_to_ab(
    lora_weights: Dict[str, mx.array],
) -> Dict[str, Tuple[str, str]]:
    """Map each LoRA base key to its (A, B) weight keys."""
    suffixes = [
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_A", ".lora_B"),
        (".lora_down", ".lora_up"),
    ]
    out: Dict[str, Tuple[str, str]] = {}
    for key in lora_weights:
        for suf_a, suf_b in suffixes:
            if key.endswith(suf_a):
                base = key[: -len(suf_a)]
                kb = base + suf_b
                if kb in lora_weights:
                    out[base] = (key, kb)
                break
    return out


def fuse_loras_into_model(
    model,
    lora_configs: List[LoRAConfig],
    *,
    include_audio: bool = True,
    min_coverage: float = 0.5,
    allow_partial: bool = False,
    verbose: bool = True,
    track_for_restore: bool = True,
    clear_every: Optional[int] = None,
    lora_load_mode: str = "auto",
) -> None:
    """Fuse one or more LoRAs into an already-loaded model, in place.

    A translation table maps each LoRA target (raw checkpoint naming, e.g.
    ``ff.net.0.proj``) to the model's MLX weight key (``ff.project_in.proj.
    weight``) via the same converter the cache uses, so FF and attention both
    match. The delta ``(B @ A)`` is ``[out, in]``; it is added directly to a
    standard weight or transposed for a pretransposed ``_*_weight_t`` slot,
    decided by shape.

    Per-target scale is exactly ``strength``. This matches the official LTX
    fuser, which treats LoRA products as already normalized and computes
    ``sum((B * strength) @ A)``. Multiple LoRAs sum their deltas.

    Memory: only LoRA-touched weights are rewritten, one at a time.
    ``lora_load_mode="auto"`` uses the same total-RAM budget heuristic as VAE
    tiling: bulk ``mx.load`` when the adapter set fits the budget, otherwise
    safetensors are indexed and streamed pair-by-pair. By default the MLX cache
    is drained less often for fast/bulk mode and more often for stream mode;
    pass ``clear_every`` to override.

    Robustness: per-weight shape mismatches and absent targets are skipped
    with a warning rather than aborting. Only the *aggregate* coverage gates
    the load -- if fewer than ``min_coverage`` of the LoRA's resolved targets
    actually fuse (or none resolve at all), it raises, since that signals a
    format/model mismatch. A LoRA that intentionally touches few weights still
    has ~100% coverage *of its own targets* and passes; pass
    ``allow_partial=True`` to force a genuinely low-coverage fuse.
    """
    from .weight_converter import convert_pytorch_key_to_mlx

    target = getattr(model, "velocity_model", model)
    selected_load_mode, load_estimate_gb, load_budget_gb = _resolve_lora_load_mode(
        lora_configs,
        lora_load_mode,
    )
    range_guard_enabled = _lora_fuse_range_guard_enabled()
    if clear_every is None:
        clear_every = 4 if selected_load_mode == "stream" else 16
    else:
        clear_every = max(1, int(clear_every))
    if verbose:
        if load_estimate_gb is None:
            print(
                f"  LoRA load mode: {selected_load_mode} "
                f"(clear every {clear_every} weights)"
            )
        else:
            print(
                f"  LoRA load mode: {selected_load_mode} "
                f"(auto estimate {load_estimate_gb:.2f} GB, "
                f"budget {load_budget_gb:.2f} GB, "
                f"clear every {clear_every} weights)"
            )
        if range_guard_enabled:
            print(f"  LoRA fused-range guard: enabled ({_LORA_FUSE_RANGE_GUARD_ENV}=1)")
    tensor_sources = []
    table: Dict[str, list] = {}

    try:
        # 1. Translation table: MLX logical ".weight" key -> tensor source refs.
        unresolved = 0   # LoRA pairs whose key did not map to any model weight name
        excluded = 0     # targets dropped by a cfg.exclude category filter
        for cfg in lora_configs:
            source = _open_lora_tensor_source(cfg.path, selected_load_mode)
            tensor_sources.append(source)
            ab = _lora_base_to_ab(source)
            exclude = set(cfg.exclude)
            cfg_excluded = 0
            for base, (ka, kb) in ab.items():
                pytorch_key = base.replace("diffusion_model.", "") + ".weight"
                mlx_key = convert_pytorch_key_to_mlx(
                    pytorch_key, include_audio=include_audio
                )
                if mlx_key is None:
                    unresolved += 1
                    continue
                if exclude and (_lora_key_categories(mlx_key) & exclude):
                    cfg_excluded += 1
                    continue
                table.setdefault(mlx_key, []).append((source, ka, kb, cfg.strength))
            excluded += cfg_excluded
            if verbose:
                excl_note = (
                    "" if not exclude
                    else f", excluding {sorted(exclude)} ({cfg_excluded} dropped)"
                )
                print(
                    f"  LoRA {Path(cfg.path).name}: strength={cfg.strength}, "
                    f"{len(ab)} targets, scale={cfg.strength:.4f}{excl_note}"
                )
            del ab

        resolved = len(table)
        if resolved == 0:
            msg = (
                "LoRA fusion: no LoRA keys resolved to any model weight -- "
                f"unrecognized format or wrong model ({unresolved} unmapped keys)."
            )
            if allow_partial:
                if verbose:
                    print(f"  [warn] {msg} (allow_partial: nothing fused)")
                return
            raise RuntimeError(
                msg + " Pass allow_partial=True (--lora-allow-partial) to ignore."
            )
        _invalidate_lora_restore_cache(target)

        def delta_fp32(mlx_key: str) -> mx.array:
            acc = None
            for source, ka, kb, scale in table[mlx_key]:
                a = source.load(ka)
                b = source.load(kb)
                term = mx.matmul(b.astype(mx.float32), a.astype(mx.float32)) * scale
                if acc is None:
                    acc = term
                else:
                    acc = acc + term
                    del term
                del a, b
            return acc

        def fused_value(weight: mx.array, d: mx.array, mlx_key: str) -> Optional[mx.array]:
            if tuple(weight.shape) == tuple(d.shape):
                fused = weight.astype(mx.float32) + d
            elif tuple(weight.shape) == tuple(reversed(d.shape)):  # pretransposed slot
                fused = weight.astype(mx.float32) + d.T
            else:
                return None  # shape mismatch -> skip, counted below
            if range_guard_enabled:
                _guard_fused_range(fused, weight.dtype, mlx_key)
            return fused.astype(weight.dtype)

        # Navigate to each weight on demand rather than holding a flattened
        # parameter list across the loop (that would pin every old array and pile
        # up a second full copy of the model). Each fused weight is evaluated and
        # written back immediately so the old one and the FP32 temporaries free
        # before the next.
        def _navigate(root, key: str):
            parts = key.split(".")
            obj = root
            for p in parts[:-1]:
                obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
            return obj, parts[-1]

        applied = set()
        shape_skipped: list = []

        # 2a. Standard params (attention, AdaLN, non-pretransposed FF, gates).
        fuse_keys = [k for k, _ in tree_flatten(target.parameters()) if k in table]
        for n, key in enumerate(fuse_keys):
            obj, leaf = _navigate(target, key)
            weight = getattr(obj, leaf)
            d = delta_fp32(key)
            new = fused_value(weight, d, key)
            if new is None:
                shape_skipped.append(key)
                del weight, d
                continue
            mx.eval(new)
            setattr(obj, leaf, new)
            applied.add(key)
            del weight, d, new
            if (n + 1) % clear_every == 0:
                gc.collect()
                mx.clear_cache()
        del fuse_keys
        mx.clear_cache()

        # 2b. Pretransposed FF slots live in private _project_*_weight_t attrs
        # (the original .weight was deleted at cache install), so they are absent
        # from parameters() and handled here by direct attr replacement.
        blocks = getattr(target, "transformer_blocks", []) or []
        since_clear = 0
        for i, blk in enumerate(blocks):
            for mod_name, attr, suffix in _LORA_FF_PRETRANSPOSE_SLOTS:
                ff = getattr(blk, mod_name, None)
                if ff is None:
                    continue
                wt = getattr(ff, attr, None)
                if wt is None:
                    continue
                mlx_key = f"transformer_blocks.{i}.{suffix}"
                if mlx_key not in table:
                    continue
                d = delta_fp32(mlx_key)
                new = fused_value(wt, d, mlx_key)
                if new is None:
                    shape_skipped.append(mlx_key)
                    del wt, d
                    continue
                mx.eval(new)
                setattr(ff, attr, new)
                applied.add(mlx_key)
                del wt, d, new
                since_clear += 1
                if since_clear >= clear_every:
                    gc.collect()
                    mx.clear_cache()
                    since_clear = 0
        mx.clear_cache()

        # 3. Coverage gate. Per-tensor misses (above) were skipped, not fatal;
        # only the aggregate fraction of resolved targets actually placed decides
        # whether this is a real fusion or a format/model mismatch.
        placed = len(applied)
        coverage = placed / resolved
        if verbose:
            bits = [f"placed {placed}/{resolved}"]
            if excluded:
                bits.append(f"{excluded} excluded")
            if shape_skipped:
                bits.append(f"{len(shape_skipped)} shape-skipped")
            if unresolved:
                bits.append(f"{unresolved} unmapped keys")
            print(
                f"  Fused {placed} weights from {len(lora_configs)} LoRA(s) "
                f"[{'; '.join(bits)}; coverage {coverage:.0%}]."
            )
        if coverage < min_coverage and not allow_partial:
            unplaced = sorted(set(table) - applied)[:3]
            raise RuntimeError(
                f"LoRA fusion coverage {coverage:.0%} < {min_coverage:.0%}: only "
                f"{placed}/{resolved} resolved targets fused (e.g. unplaced "
                f"{unplaced}). Likely a layout/model mismatch. If this LoRA "
                "intentionally targets few weights, pass allow_partial=True "
                "(--lora-allow-partial)."
            )
    finally:
        table.clear()
        for source in tensor_sources:
            source.close()
        gc.collect()
        mx.clear_cache()

    if track_for_restore:
        _record_persistent_loras(
            target,
            lora_configs,
            include_audio=include_audio,
            min_coverage=min_coverage,
            allow_partial=allow_partial,
        )


# Common LoRA target modules in transformer models
LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "ff.project_in",
    "ff.project_out",
    "attn.query",
    "attn.key",
    "attn.value",
    "attn.output",
]


def get_lora_target_keys(model_weights: Dict[str, mx.array]) -> List[str]:
    """
    Find weight keys that are common LoRA targets.

    Args:
        model_weights: Model weights dictionary.

    Returns:
        List of keys that are typically modified by LoRA.
    """
    target_keys = []
    for key in model_weights.keys():
        for target in LORA_TARGET_MODULES:
            if target in key and key.endswith(".weight"):
                target_keys.append(key)
                break
    return target_keys
