"""LoRA (Low-Rank Adaptation) loading and fusion for LTX-2 models."""

import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


# Categories a LoRA target can be filtered by, on two orthogonal axes:
#   branch -- video / audio / cross (the video<->audio bridge attentions)
#   module type -- attn (to_q/k/v/out) / gate (to_gate_logits) / ff
# Each target carries one branch tag and one type tag; excluding any tag drops
# every target that carries it (e.g. exclude {audio, cross} keeps a video-only
# style without touching the audio branch; exclude {ff} drops feed-forward
# everywhere). Classification is by substring on the MLX weight key, whose
# module names survive conversion.
_LORA_CATEGORIES = frozenset({"video", "audio", "cross", "attn", "gate", "ff"})


def _lora_key_categories(mlx_key: str) -> set:
    cats = set()
    if "video_to_audio" in mlx_key or "audio_to_video" in mlx_key:
        cats.add("cross")
    elif "audio_attn" in mlx_key or "audio_ff" in mlx_key:
        cats.add("audio")
    else:
        cats.add("video")
    if "to_gate_logits" in mlx_key:
        cats.add("gate")
    elif any(t in mlx_key for t in ("to_q", "to_k", "to_v", "to_out")):
        cats.add("attn")
    elif "ff" in mlx_key:
        cats.add("ff")
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


def find_lora_keys_for_weight(
    lora_weights: Dict[str, mx.array],
    base_key: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find LoRA A and B keys corresponding to a base weight key.

    LoRA weights typically follow naming patterns like:
    - base_key.lora_A.weight / base_key.lora_B.weight
    - base_key.lora_down.weight / base_key.lora_up.weight

    Args:
        lora_weights: Dictionary of LoRA weights.
        base_key: The base model weight key (e.g., "transformer.layers.0.attn.to_q.weight").

    Returns:
        Tuple of (lora_A_key, lora_B_key) or (None, None) if not found.
    """
    # Remove .weight suffix if present
    prefix = base_key.replace(".weight", "")

    # Candidate prefixes to try
    candidate_prefixes = [prefix]
    if not prefix.startswith("diffusion_model."):
        candidate_prefixes.append(f"diffusion_model.{prefix}")
    if prefix.startswith("model."):
        candidate_prefixes.append(prefix.replace("model.", "diffusion_model."))
    
    # Redefine patterns as suffixes to apply to prefix
    suffixes = [
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_A", ".lora_B"),
        (".lora_down", ".lora_up"),
    ]

    for cand_prefix in candidate_prefixes:
        for suff_a, suff_b in suffixes:
            key_a = f"{cand_prefix}{suff_a}"
            key_b = f"{cand_prefix}{suff_b}"
            
            if key_a in lora_weights and key_b in lora_weights:
                return key_a, key_b

    return None, None


def compute_lora_delta(
    lora_weights: Dict[str, mx.array],
    key_a: str,
    key_b: str,
    strength: float = 1.0,
) -> mx.array:
    """
    Compute the LoRA delta: strength * (lora_B @ lora_A).

    Args:
        lora_weights: Dictionary of LoRA weights.
        key_a: Key for LoRA A weights (down projection).
        key_b: Key for LoRA B weights (up projection).
        strength: Scaling factor for the delta.

    Returns:
        The computed delta to add to base weights.
    """
    lora_a = lora_weights[key_a]
    lora_b = lora_weights[key_b]

    # LoRA computation: delta = B @ A
    # A: (rank, in_features)
    # B: (out_features, rank)
    # Result: (out_features, in_features)
    delta = mx.matmul(lora_b, lora_a) * strength

    return delta


def fuse_lora_into_weights(
    model_weights: Dict[str, mx.array],
    lora_configs: List[LoRAConfig],
    target_dtype: Optional[mx.Dtype] = None,
    verbose: bool = True,
) -> Dict[str, mx.array]:
    """
    Fuse LoRA weights into model weights.

    Computes W_final = W_base + sum(strength_i * (B_i @ A_i)) for each LoRA.

    Args:
        model_weights: Base model weights dictionary.
        lora_configs: List of LoRA configurations to apply.
        target_dtype: Target dtype for output weights. If None, uses base weight dtype.
        verbose: Whether to print progress information.

    Returns:
        New dictionary with LoRA-fused weights.
    """
    # Load all LoRA weights
    all_loras = []
    for config in lora_configs:
        if verbose:
            print(f"Loading LoRA: {config.path} (strength={config.strength})")
        lora_weights = load_lora_weights(config.path)
        all_loras.append((lora_weights, config.strength))

    # Create output dictionary
    fused_weights = {}
    fused_count = 0
    skipped_count = 0

    for key, base_weight in model_weights.items():
        # Determine target dtype
        out_dtype = target_dtype if target_dtype is not None else base_weight.dtype

        # Start with base weight
        fused = base_weight.astype(mx.float32) if base_weight.dtype != mx.float32 else base_weight

        # Try to find and apply LoRA for this key
        applied = False
        for lora_weights, strength in all_loras:
            key_a, key_b = find_lora_keys_for_weight(lora_weights, key)
            if key_a is not None and key_b is not None:
                delta = compute_lora_delta(lora_weights, key_a, key_b, strength)

                # Ensure shapes match
                if delta.shape == fused.shape:
                    fused = fused + delta
                    applied = True
                elif verbose:
                    print(f"  Shape mismatch for {key}: base={fused.shape}, delta={delta.shape}")

        if applied:
            fused_count += 1
        else:
            skipped_count += 1

        # Convert to target dtype
        fused_weights[key] = fused.astype(out_dtype)

    if verbose:
        print(f"Fused LoRA into {fused_count} weights, skipped {skipped_count}")

    return fused_weights


def apply_lora_to_model(
    model,
    lora_configs: List[LoRAConfig],
    verbose: bool = True,
) -> None:
    """
    Apply LoRA weights directly to a model's parameters in-place.

    Args:
        model: MLX model with parameters to modify.
        lora_configs: List of LoRA configurations to apply.
        verbose: Whether to print progress information.
    """
    # Get current model parameters
    params = dict(model.parameters())

    # Flatten nested parameters
    flat_params = {}
    def flatten_params(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_params(v, full_key)
            else:
                flat_params[full_key] = v
    flatten_params(params)

    # Load all LoRAs
    all_loras = []
    for config in lora_configs:
        if verbose:
            print(f"Loading LoRA: {config.path} (strength={config.strength})")
        lora_weights = load_lora_weights(config.path)
        all_loras.append((lora_weights, config.strength))

    # Apply LoRAs to matching parameters
    fused_count = 0
    for key, param in flat_params.items():
        for lora_weights, strength in all_loras:
            key_a, key_b = find_lora_keys_for_weight(lora_weights, key)
            if key_a is not None and key_b is not None:
                delta = compute_lora_delta(lora_weights, key_a, key_b, strength)

                if delta.shape == param.shape:
                    # Update parameter
                    new_value = param + delta.astype(param.dtype)

                    # Navigate to and update the nested parameter
                    parts = key.split(".")
                    obj = model
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], new_value)
                    fused_count += 1

    if verbose:
        print(f"Applied LoRA to {fused_count} parameters")

    # Evaluate to ensure changes are applied
    mx.eval(model.parameters())


# Reject alphas that are non-finite or absurdly large; a junk value would
# otherwise scale the delta into garbage. Mirrors the mflux loader's filter.
_LORA_ALPHA_MAX = 1.0e6


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

    def delta_fp32(mlx_key: str) -> mx.array:
        acc = None
        for a, b, scale in table[mlx_key]:
            term = mx.matmul(b.astype(mx.float32), a.astype(mx.float32)) * scale
            acc = term if acc is None else acc + term
        return acc

    def fused_value(weight: mx.array, d: mx.array) -> Optional[mx.array]:
        if tuple(weight.shape) == tuple(d.shape):
            return (weight.astype(mx.float32) + d).astype(weight.dtype)
        if tuple(weight.shape) == tuple(reversed(d.shape)):  # pretransposed slot
            return (weight.astype(mx.float32) + d.T).astype(weight.dtype)
        return None  # shape mismatch -> skip, counted below

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
        new = fused_value(getattr(obj, leaf), delta_fp32(key))
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
    ff_slots = (
        ("ff", "_project_in_weight_t", "ff.project_in.proj.weight"),
        ("ff", "_project_out_weight_t", "ff.project_out.weight"),
        ("audio_ff", "_project_in_weight_t", "audio_ff.project_in.proj.weight"),
        ("audio_ff", "_project_out_weight_t", "audio_ff.project_out.weight"),
    )
    since_clear = 0
    for i, blk in enumerate(blocks):
        for mod_name, attr, suffix in ff_slots:
            ff = getattr(blk, mod_name, None)
            if ff is None:
                continue
            wt = getattr(ff, attr, None)
            if wt is None:
                continue
            mlx_key = f"transformer_blocks.{i}.{suffix}"
            if mlx_key not in table:
                continue
            new = fused_value(wt, delta_fp32(mlx_key))
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
