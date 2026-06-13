"""LoRA (Low-Rank Adaptation) loading and fusion for LTX-2 models."""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter."""

    path: str
    strength: float = 1.0

    def __post_init__(self):
        if not -2.0 <= self.strength <= 2.0:
            raise ValueError(f"LoRA strength should be between -2.0 and 2.0, got {self.strength}")


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


def _lora_alpha_scale(path: str, strength: float) -> float:
    """Effective scale = strength * (alpha/rank), reading the LoRA's metadata.

    Diffusers-style LoRA applies ``(alpha/rank) * (B @ A)``. Many LTX LoRAs
    ship ``alpha == rank`` (factor 1.0), but honoring the metadata keeps the
    fuser correct for adapters where they differ. Falls back to ``strength``
    when the metadata is absent.
    """
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            meta = json.loads(f.read(n)).get("__metadata__", {}) or {}
        alpha = float(meta["lora_alpha"])
        rank = float(meta["lora_rank"])
        if rank:
            return strength * (alpha / rank)
    except Exception:
        pass
    return strength


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
    verbose: bool = True,
) -> None:
    """Fuse one or more LoRAs into an already-loaded model, in place.

    Designed for the cached/pretransposed model: a translation table maps
    each LoRA target (raw checkpoint naming, e.g. ``ff.net.0.proj``) to the
    model's MLX weight key (e.g. ``ff.project_in.proj.weight``) via the same
    converter the cache uses, so FF and attention both match. The delta
    ``(B @ A)`` is ``[out, in]``; it is added directly to a standard weight
    or transposed for a pretransposed ``_*_weight_t`` slot, decided by shape.

    Memory: only LoRA-touched weights are rewritten, in chunks, freeing the
    previous copy as it goes -- it never materializes a second full set of
    weights (unlike fuse_lora_into_weights). Multiple LoRAs sum at their
    individual strengths. Raises if any LoRA target cannot be placed, so an
    unsupported layout fails loud instead of silently dropping adaptation.
    """
    from .weight_converter import convert_pytorch_key_to_mlx

    target = getattr(model, "velocity_model", model)

    # 1. Translation table: MLX logical ".weight" key -> [(A, B, scale), ...]
    table: Dict[str, list] = {}
    for cfg in lora_configs:
        lw = load_lora_weights(cfg.path)
        scale = _lora_alpha_scale(cfg.path, cfg.strength)
        if verbose:
            print(
                f"  LoRA {Path(cfg.path).name}: strength={cfg.strength} "
                f"effective_scale={scale:.4f}"
            )
        for base, (ka, kb) in _lora_base_to_ab(lw).items():
            pytorch_key = base.replace("diffusion_model.", "") + ".weight"
            mlx_key = convert_pytorch_key_to_mlx(
                pytorch_key, include_audio=include_audio
            )
            if mlx_key is not None:
                table.setdefault(mlx_key, []).append((lw[ka], lw[kb], scale))

    if not table:
        if verbose:
            print("  No LoRA targets resolved; nothing fused.")
        return

    def delta_fp32(mlx_key: str) -> mx.array:
        acc = None
        for a, b, scale in table[mlx_key]:
            term = mx.matmul(b.astype(mx.float32), a.astype(mx.float32)) * scale
            acc = term if acc is None else acc + term
        return acc

    def fused_value(weight: mx.array, d: mx.array, key: str) -> mx.array:
        if tuple(weight.shape) == tuple(d.shape):
            return (weight.astype(mx.float32) + d).astype(weight.dtype)
        if tuple(weight.shape) == tuple(reversed(d.shape)):  # pretransposed slot
            return (weight.astype(mx.float32) + d.T).astype(weight.dtype)
        raise ValueError(
            f"LoRA shape mismatch for {key}: weight {tuple(weight.shape)} "
            f"vs delta {tuple(d.shape)}"
        )

    applied = set()

    # 2a. Standard params (attention, AdaLN, non-pretransposed FF, gates).
    chunk: list = []
    chunk_bytes = 0
    for key, weight in tree_flatten(target.parameters()):
        if key not in table:
            continue
        new = fused_value(weight, delta_fp32(key), key)
        chunk.append((key, new))
        chunk_bytes += new.nbytes
        applied.add(key)
        if chunk_bytes >= 2 * 1024**3:
            target.load_weights(chunk, strict=False)
            mx.eval(target.parameters())
            chunk, chunk_bytes = [], 0
    if chunk:
        target.load_weights(chunk, strict=False)
        mx.eval(target.parameters())

    # 2b. Pretransposed FF slots live in private _project_*_weight_t attrs
    # (the original .weight was deleted at cache install), so they are absent
    # from parameters() and handled here by direct, chunked attr replacement.
    blocks = getattr(target, "transformer_blocks", []) or []
    ff_slots = (
        ("ff", "_project_in_weight_t", "ff.project_in.proj.weight"),
        ("ff", "_project_out_weight_t", "ff.project_out.weight"),
        ("audio_ff", "_project_in_weight_t", "audio_ff.project_in.proj.weight"),
        ("audio_ff", "_project_out_weight_t", "audio_ff.project_out.weight"),
    )
    pending = []
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
            setattr(ff, attr, fused_value(wt, delta_fp32(mlx_key), mlx_key))
            pending.append(getattr(ff, attr))
            applied.add(mlx_key)
            if len(pending) >= 32:
                mx.eval(pending)
                pending = []
    if pending:
        mx.eval(pending)

    # 3. Fail loud if any resolved LoRA target had no home in this model/layout
    # (e.g. an unsupported pretransposed attention layout), rather than
    # silently shipping a partial adaptation.
    unplaced = sorted(set(table) - applied)
    if unplaced:
        raise RuntimeError(
            f"LoRA fusion incomplete: {len(unplaced)} target(s) had no matching "
            f"model weight (e.g. {unplaced[:3]}). This usually means a "
            "pretransposed layout the fuser does not yet handle; refusing to "
            "ship a partial fusion."
        )
    if verbose:
        print(f"  Fused {len(applied)} weights from {len(lora_configs)} LoRA(s).")


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
