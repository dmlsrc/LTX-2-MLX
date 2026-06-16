"""LoRA (Low-Rank Adaptation) loading and fusion for LTX-2 models."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

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
_LORA_FUSE_RANGE_GUARD_ENV = "LTX_LORA_FUSE_RANGE_GUARD"


def _lora_key_categories(mlx_key: str) -> set:
    cats = set()
    if "video_to_audio" in mlx_key or "audio_to_video" in mlx_key:
        cats.add("cross")
    elif mlx_key.startswith("audio_") or ".audio_" in mlx_key or "audio_" in mlx_key:
        cats.add("audio")
    else:
        cats.add("video")

    if "to_gate_logits" in mlx_key:
        cats.add("gate")
    elif any(t in mlx_key for t in ("to_q", "to_k", "to_v", "to_out")):
        cats.add("attn")
    elif "ff" in mlx_key:
        cats.add("ff")

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

    parts = mlx_key.split(".")
    if "transformer_blocks" in parts:
        i = parts.index("transformer_blocks")
        seg = parts[i + 2:]
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
    """Configuration for a single LoRA adapter."""

    path: str
    strength: float = 1.0
    stage_1_strength: float | None = None
    stage_2_strength: float | None = None
    exclude: tuple[str, ...] = ()

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

    def with_strength(self, strength: float) -> LoRAConfig:
        return LoRAConfig(path=self.path, strength=strength, exclude=self.exclude)


def lora_configs_have_stage_strengths(lora_configs: list[LoRAConfig] | None) -> bool:
    return any(cfg.has_stage_strengths() for cfg in (lora_configs or ()))


def lora_configs_for_stage(
    lora_configs: list[LoRAConfig] | None,
    stage: int,
) -> list[LoRAConfig]:
    return [
        cfg.with_strength(cfg.strength_for_stage(stage))
        for cfg in (lora_configs or ())
        if cfg.strength_for_stage(stage) != 0.0
    ]


def lora_configs_for_stage_delta(
    lora_configs: list[LoRAConfig] | None,
    *,
    from_stage: int,
    to_stage: int,
) -> list[LoRAConfig]:
    out: list[LoRAConfig] = []
    for cfg in lora_configs or ():
        delta = cfg.strength_for_stage(to_stage) - cfg.strength_for_stage(from_stage)
        if delta != 0.0:
            out.append(cfg.with_strength(delta))
    return out


def format_lora_stage_scale_lines(
    lora_configs: list[LoRAConfig] | None,
    stage: int,
    *,
    from_stage: int | None = None,
    include_unchanged: bool = False,
) -> list[str]:
    """Return human-readable stage LoRA total/change lines."""
    lines: list[str] = []
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
    lora_weights: dict[str, mx.array],
) -> dict[str, tuple[str, str]]:
    """Map each LoRA base key to its (A, B) weight keys."""
    suffixes = [
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_A", ".lora_B"),
        (".lora_down", ".lora_up"),
    ]
    out: dict[str, tuple[str, str]] = {}
    for key in lora_weights:
        for suf_a, suf_b in suffixes:
            if key.endswith(suf_a):
                base = key[: -len(suf_a)]
                kb = base + suf_b
                if kb in lora_weights:
                    out[base] = (key, kb)
                break
    return out


def _navigate_existing(root, key: str):
    obj = root
    parts = key.split(".")
    try:
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            return None
        return obj, parts[-1]
    except (AttributeError, IndexError, TypeError):
        return None


def _pretransposed_target(root, key: str):
    prefix = "transformer_blocks."
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    block_idx_text, sep, suffix = rest.partition(".")
    if not sep or not block_idx_text.isdigit():
        return None
    blocks = getattr(root, "transformer_blocks", None)
    if blocks is None:
        return None
    block_idx = int(block_idx_text)
    if block_idx >= len(blocks):
        return None
    block = blocks[block_idx]
    for module_name, attr, slot_suffix in _LORA_FF_PRETRANSPOSE_SLOTS:
        if suffix != slot_suffix:
            continue
        module = getattr(block, module_name, None)
        if module is not None and getattr(module, attr, None) is not None:
            return module, attr
    return None


def fuse_loras_into_model(
    model,
    lora_configs: list[LoRAConfig],
    *,
    include_audio: bool = True,
    min_coverage: float = 0.5,
    allow_partial: bool = False,
    verbose: bool = True,
) -> None:
    """Fuse one or more LoRAs into an already-loaded model, in place.

    The loader uses ``mx.load`` for the full LoRA file and builds lazy MLX
    expressions for ``base + (B @ A) * strength``. It does not stream tensors,
    manually clear the MLX cache, or keep a RAM copy of base transformer
    weights. Temporary stage-specific restores are handled by the transformer
    cache restore-state helpers.

    LoRA alpha/rank metadata is intentionally ignored here. The official LTX
    fuser treats adapter products as already normalized and applies only the
    requested strength. Generic PEFT/Diffusers alpha semantics are real, but
    LTX adapter intent cannot be inferred safely from metadata alone. Verify a
    new adapter against its model card or reference loader before changing this
    production rule.
    """
    from .transformer_cache import invalidate_transformer_cache_restore_state
    from .weight_converter import convert_pytorch_key_to_mlx

    target = getattr(model, "velocity_model", model)
    range_guard_enabled = _lora_fuse_range_guard_enabled()
    if verbose and range_guard_enabled:
        print(f"  LoRA fused-range guard: enabled ({_LORA_FUSE_RANGE_GUARD_ENV}=1)")

    table: dict[str, list] = {}
    unresolved = 0
    excluded = 0

    for cfg in lora_configs:
        weights = mx.load(cfg.path)
        ab = _lora_base_to_ab(weights)
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
            table.setdefault(mlx_key, []).append((weights, ka, kb, cfg.strength))
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

    invalidate_transformer_cache_restore_state(target)

    def delta_fp32(mlx_key: str) -> mx.array:
        acc = None
        for weights, ka, kb, scale in table[mlx_key]:
            a = weights[ka]
            b = weights[kb]
            term = mx.matmul(b.astype(mx.float32), a.astype(mx.float32)) * scale
            acc = term if acc is None else acc + term
        return acc

    def fused_value(weight: mx.array, d: mx.array, mlx_key: str) -> mx.array | None:
        if tuple(weight.shape) == tuple(d.shape):
            fused = weight.astype(mx.float32) + d
        elif tuple(weight.shape) == tuple(reversed(d.shape)):
            fused = weight.astype(mx.float32) + d.T
        else:
            return None
        if range_guard_enabled:
            _guard_fused_range(fused, weight.dtype, mlx_key)
        return fused.astype(weight.dtype)

    applied = set()
    shape_skipped: list[str] = []
    for key in table:
        target_ref = _navigate_existing(target, key) or _pretransposed_target(target, key)
        if target_ref is None:
            continue
        obj, leaf = target_ref
        weight = getattr(obj, leaf)
        new = fused_value(weight, delta_fp32(key), key)
        if new is None:
            shape_skipped.append(key)
            continue
        setattr(obj, leaf, new)
        applied.add(key)

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
