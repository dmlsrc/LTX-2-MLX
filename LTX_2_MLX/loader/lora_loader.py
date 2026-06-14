"""LoRA (Low-Rank Adaptation) loading and fusion for LTX-2 models."""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx
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
_LORA_CATEGORIES = (
    _LORA_BRANCH_TAGS | _LORA_TYPE_TAGS | _LORA_MODULE_TAGS | _LORA_PROJ_TAGS
)
_LORA_FF_PRETRANSPOSE_SLOTS = (
    ("ff", "_project_in_weight_t", "ff.project_in.proj.weight"),
    ("ff", "_project_out_weight_t", "ff.project_out.weight"),
    ("audio_ff", "_project_in_weight_t", "audio_ff.project_in.proj.weight"),
    ("audio_ff", "_project_out_weight_t", "audio_ff.project_out.weight"),
)
_LORA_RESTORE_CACHE_ATTR = "_lora_restore_cache_source"


def _lora_key_categories(mlx_key: str) -> set:
    cats = set()
    # coarse branch
    if "video_to_audio" in mlx_key or "audio_to_video" in mlx_key:
        cats.add("cross")
    elif "audio_attn" in mlx_key or "audio_ff" in mlx_key:
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
    return cats


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter.

    ``exclude`` lists categories (see ``_LORA_CATEGORIES``) whose targets are
    dropped from this adapter's fusion -- e.g. ``("audio", "cross")`` applies
    only the video-branch style and leaves the audio path stock.
    """

    path: str
    strength: float = 1.0
    exclude: Tuple[str, ...] = ()

    def __post_init__(self):
        if not -2.0 <= self.strength <= 2.0:
            raise ValueError(f"LoRA strength should be between -2.0 and 2.0, got {self.strength}")
        bad = sorted(set(self.exclude) - _LORA_CATEGORIES)
        if bad:
            raise ValueError(
                f"Unknown LoRA exclude categories {bad}; "
                f"valid: {sorted(_LORA_CATEGORIES)}"
            )


def load_lora_weights(path: str) -> Dict[str, mx.array]:
    """
    Load LoRA weights from a safetensors file.

    Args:
        path: Path to the LoRA weights file.

    Returns:
        Dictionary mapping weight names to arrays.
    """
    return dict(mx.load(path))


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


# Reject alphas that are non-finite or absurdly large; a junk value would
# otherwise scale the delta into garbage. Mirrors the mflux loader's filter.
_LORA_ALPHA_MAX = 1.0e6

# Max finite magnitude per float cache dtype. LoRA fusion runs in float32 and is
# cast back to the weight's dtype; a fused value beyond that dtype's range casts
# silently to +-inf, which becomes NaN in the forward pass and renders as a flat
# (green) frame. An overflowed weight never produces useful output, so the fuse
# fails fast here -- for ANY dtype -- rather than after a full generation.
_DTYPE_MAX_FINITE = {
    mx.float16: 65504.0,
    mx.bfloat16: 3.3895313892515355e38,
    mx.float32: 3.4028234663852886e38,
}


def _guard_fused_range(fused_f32: mx.array, target_dtype, mlx_key: str) -> None:
    """Raise if the float32 fused weight is non-finite or would overflow
    ``target_dtype`` when cast back. Overflow -> inf -> NaN -> degenerate
    output, so this is unconditionally an error."""
    peak = float(mx.max(mx.abs(fused_f32)).item())
    if not math.isfinite(peak):
        raise ValueError(
            f"LoRA fusion produced a non-finite weight at '{mlx_key}' -- the "
            f"base weight or the delta already overflowed float32. Lower "
            f"--lora-strength or check the adapter."
        )
    limit = _DTYPE_MAX_FINITE.get(target_dtype)
    if limit is not None and peak > limit:
        raise ValueError(
            f"LoRA fusion overflows {target_dtype} at '{mlx_key}': fused "
            f"|max|={peak:.4g} exceeds the dtype ceiling {limit:.4g}. Cache the "
            f"transformer in a wider dtype (--transformer-dtype bf16) or lower "
            f"--lora-strength."
        )


def _lora_metadata_alpha(path: str) -> Optional[float]:
    """File-level LoRA alpha from safetensors metadata, or None.

    Honors diffusers-style ``lora_alpha`` and kohya-style ``ss_network_alpha``.
    Per-module ``.alpha`` tensors (read at fuse time) take precedence; this is
    only the fallback for LoRAs that record alpha solely in metadata. The
    companion rank is taken from the actual ``lora_A`` shape, not metadata, so
    only the alpha value is needed here.
    """
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            meta = json.loads(f.read(n)).get("__metadata__", {}) or {}
    except Exception:
        return None
    for key in ("lora_alpha", "ss_network_alpha", "alpha"):
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                pass
    return None


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
) -> None:
    """Fuse one or more LoRAs into an already-loaded model, in place.

    A translation table maps each LoRA target (raw checkpoint naming, e.g.
    ``ff.net.0.proj``) to the model's MLX weight key (``ff.project_in.proj.
    weight``) via the same converter the cache uses, so FF and attention both
    match. The delta ``(B @ A)`` is ``[out, in]``; it is added directly to a
    standard weight or transposed for a pretransposed ``_*_weight_t`` slot,
    decided by shape.

    Per-target scale = ``strength * alpha/rank`` where rank is the LoRA's
    actual rank (``lora_A`` row count) and alpha is, in precedence order: the
    per-module ``.alpha`` tensor, the file metadata alpha (diffusers
    ``lora_alpha`` or kohya ``ss_network_alpha``), else rank (factor 1.0).
    Non-finite / absurd alphas are rejected. Multiple LoRAs sum their deltas.

    Memory: only LoRA-touched weights are rewritten, one at a time, freeing
    the previous copy as it goes -- never a second full copy of the model.

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

    # 1. Translation table: MLX logical ".weight" key -> [(A, B, scale), ...].
    table: Dict[str, list] = {}
    unresolved = 0   # LoRA pairs whose key did not map to any model weight name
    bad_alpha = 0    # rejected non-finite / out-of-range alpha values
    excluded = 0     # targets dropped by a cfg.exclude category filter
    for cfg in lora_configs:
        lw = load_lora_weights(cfg.path)
        meta_alpha = _lora_metadata_alpha(cfg.path)
        ab = _lora_base_to_ab(lw)
        exclude = set(cfg.exclude)
        scales_seen = set()
        cfg_excluded = 0
        for base, (ka, kb) in ab.items():
            a = lw[ka]
            rank = a.shape[0] if a.ndim >= 1 else 0
            alpha = None
            alpha_t = lw.get(base + ".alpha")
            if alpha_t is not None:
                try:
                    alpha = float(alpha_t)
                except Exception:
                    alpha = None
            if alpha is None:
                alpha = meta_alpha
            if alpha is not None and not (
                math.isfinite(alpha) and 0.0 < alpha <= _LORA_ALPHA_MAX
            ):
                bad_alpha += 1
                alpha = None
            ratio = (alpha / rank) if (alpha is not None and rank) else 1.0
            scale = cfg.strength * ratio
            scales_seen.add(round(scale, 4))
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
            table.setdefault(mlx_key, []).append((lw[ka], lw[kb], scale))
        excluded += cfg_excluded
        if verbose:
            srep = sorted(scales_seen)
            srep_str = (
                f"{srep[0]:.4f}" if len(srep) == 1
                else f"{len(srep)} distinct ({min(srep):.4f}-{max(srep):.4f})"
            )
            meta_note = "" if meta_alpha is None else f", meta_alpha={meta_alpha:g}"
            excl_note = (
                "" if not exclude
                else f", excluding {sorted(exclude)} ({cfg_excluded} dropped)"
            )
            print(
                f"  LoRA {Path(cfg.path).name}: strength={cfg.strength}, "
                f"{len(ab)} targets, scale={srep_str}{meta_note}{excl_note}"
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
        raise RuntimeError(msg + " Pass allow_partial=True (--lora-allow-partial) to ignore.")
    _invalidate_lora_restore_cache(target)

    def delta_fp32(mlx_key: str) -> mx.array:
        acc = None
        for a, b, scale in table[mlx_key]:
            term = mx.matmul(b.astype(mx.float32), a.astype(mx.float32)) * scale
            acc = term if acc is None else acc + term
        return acc

    def fused_value(weight: mx.array, d: mx.array, mlx_key: str) -> Optional[mx.array]:
        if tuple(weight.shape) == tuple(d.shape):
            fused = weight.astype(mx.float32) + d
        elif tuple(weight.shape) == tuple(reversed(d.shape)):  # pretransposed slot
            fused = weight.astype(mx.float32) + d.T
        else:
            return None  # shape mismatch -> skip, counted below
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
        new = fused_value(getattr(obj, leaf), delta_fp32(key), key)
        if new is None:
            shape_skipped.append(key)
            continue
        mx.eval(new)
        setattr(obj, leaf, new)
        applied.add(key)
        if (n + 1) % 32 == 0:
            mx.clear_cache()
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
            new = fused_value(wt, delta_fp32(mlx_key), mlx_key)
            if new is None:
                shape_skipped.append(mlx_key)
                continue
            mx.eval(new)
            setattr(ff, attr, new)
            applied.add(mlx_key)
            since_clear += 1
            if since_clear >= 32:
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
        if bad_alpha:
            bits.append(f"{bad_alpha} bad-alpha ignored")
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
