"""Attention mechanisms for LTX-2 Transformer."""

import os
from collections.abc import Callable
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ...kernels.steel_attention import maybe_steel_attention
from .rope import LTXRopeType, apply_rotary_emb
from ...utils.signpost import signpost as _signpost, signpost_barrier as _sp_barrier


# Set LTX_DISABLE_COMPILED_ATTN=1 to bypass the @mx.compile wrappers around the
# reshape+SDPA+reshape sequence.  Used to A/B compile overhead vs fusion.
_USE_COMPILED_ATTN = not os.environ.get("LTX_DISABLE_COMPILED_ATTN")


def _env_enabled_by_default(name: str, disable_name: str) -> bool:
    false_values = {"", "0", "false", "no", "off"}
    if os.environ.get(disable_name, "").strip().lower() not in false_values:
        return False
    value = os.environ.get(name)
    return value is None or value.strip().lower() not in false_values


# The local STEEL retile is the default for supported no-mask D128/D64 shapes.
# Set LTX_DISABLE_STEEL_ATTN=1 or LTX_STEEL_ATTN=0 to force stock MLX SDPA.
_USE_STEEL_ATTN = _env_enabled_by_default("LTX_STEEL_ATTN", "LTX_DISABLE_STEEL_ATTN")


_KV_DOWNSAMPLE_CONFIG: Optional[dict[str, object]] = None
_KV_DOWNSAMPLE_COUNTS: dict[str, int] = {"applied": 0, "fallback": 0}
_KV_DOWNSAMPLE_REASONS: dict[str, int] = {}


def configure_kv_downsample(
    *,
    pool_h: int,
    pool_w: int,
    frames: int,
    height: int,
    width: int,
    heads: int = 32,
    dim_head: int = 128,
    mode: str = "mean",
    max_applied: Optional[int] = None,
) -> None:
    """Enable a failed-experiment K/V reduction path for video self-attn.

    This stays diagnostic-only in the stage-2 harness.  Local A/B showed real
    latency wins but unacceptable blur/ghosting, even when disabled for the
    final stage-2 refinement step.
    """
    if pool_h < 1 or pool_w < 1:
        raise ValueError("K/V downsample pool dimensions must be positive")
    if height % pool_h or width % pool_w:
        raise ValueError(
            f"K/V downsample pool {pool_h}x{pool_w} does not divide "
            f"stage-2 grid {height}x{width}"
        )
    if mode not in {"mean", "stride"}:
        raise ValueError(f"Unsupported K/V downsample mode: {mode}")
    if max_applied is not None and max_applied < 0:
        raise ValueError("K/V downsample max_applied must be non-negative")

    global _KV_DOWNSAMPLE_CONFIG
    _KV_DOWNSAMPLE_CONFIG = {
        "pool_h": int(pool_h),
        "pool_w": int(pool_w),
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "tokens": int(frames * height * width),
        "heads": int(heads),
        "dim_head": int(dim_head),
        "mode": mode,
    }
    if max_applied is not None:
        _KV_DOWNSAMPLE_CONFIG["max_applied"] = int(max_applied)
    reset_kv_downsample_stats()


def clear_kv_downsample() -> None:
    global _KV_DOWNSAMPLE_CONFIG
    _KV_DOWNSAMPLE_CONFIG = None
    reset_kv_downsample_stats()


def reset_kv_downsample_stats() -> None:
    _KV_DOWNSAMPLE_COUNTS["applied"] = 0
    _KV_DOWNSAMPLE_COUNTS["fallback"] = 0
    _KV_DOWNSAMPLE_REASONS.clear()


def kv_downsample_summary() -> Optional[dict]:
    if _KV_DOWNSAMPLE_CONFIG is None:
        return None
    return {
        "config": dict(_KV_DOWNSAMPLE_CONFIG),
        "counts": dict(_KV_DOWNSAMPLE_COUNTS),
        "fallback_reasons": dict(sorted(_KV_DOWNSAMPLE_REASONS.items())),
    }


def _kv_downsample_record(reason: Optional[str]) -> None:
    if reason is None:
        _KV_DOWNSAMPLE_COUNTS["applied"] += 1
        return
    _KV_DOWNSAMPLE_COUNTS["fallback"] += 1
    _KV_DOWNSAMPLE_REASONS[reason] = _KV_DOWNSAMPLE_REASONS.get(reason, 0) + 1


def _kv_downsample_candidate(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
) -> tuple[bool, str]:
    cfg = _KV_DOWNSAMPLE_CONFIG
    if cfg is None:
        return False, "disabled"
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False, "ndim"
    if heads != cfg["heads"] or dim_head != cfg["dim_head"]:
        return False, "heads_dim"
    if q.shape[0] != 1:
        return False, "batch"
    if q.shape[2] != heads * dim_head:
        return False, "inner_dim"
    if k.shape != q.shape or v.shape != q.shape:
        return False, "not_self_attn"
    if q.shape[1] != cfg["tokens"]:
        return False, "tokens"
    if q.dtype not in (mx.bfloat16, mx.float16):
        return False, "dtype"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False, "dtype_mismatch"
    max_applied = cfg.get("max_applied")
    if max_applied is not None and _KV_DOWNSAMPLE_COUNTS["applied"] >= max_applied:
        return False, "budget"
    return True, ""


def _pool_spatial_heads(x: mx.array) -> mx.array:
    cfg = _KV_DOWNSAMPLE_CONFIG
    if cfg is None:
        return x
    dtype = x.dtype
    b, heads, tokens, dim = x.shape
    frames = cfg["frames"]
    height = cfg["height"]
    width = cfg["width"]
    pool_h = cfg["pool_h"]
    pool_w = cfg["pool_w"]

    x = x.reshape(
        b,
        heads,
        frames,
        height // pool_h,
        pool_h,
        width // pool_w,
        pool_w,
        dim,
    )
    if cfg.get("mode", "mean") == "stride":
        x = x[:, :, :, :, 0, :, 0, :]
    else:
        x = mx.mean(x.astype(mx.float32), axis=(4, 6)).astype(dtype)
    pooled_tokens = frames * (height // pool_h) * (width // pool_w)
    return mx.contiguous(x.reshape(b, heads, pooled_tokens, dim))


def _sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    if _USE_STEEL_ATTN:
        out = maybe_steel_attention(q, k, v, scale=scale, mask=mask)
        if out is not None:
            return out
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def _attention_core_inline_no_mask(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
) -> mx.array:
    """Inline (uncompiled) attention core without mask."""
    b, t_q, _ = q.shape
    _, t_k, _ = k.shape

    # Reshape for multi-head attention: (B, T, H*D) -> (B, H, T, D)
    q = q.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)
    k = k.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)
    v = v.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)

    # Compute attention using Flash Attention
    scale = 1.0 / (dim_head ** 0.5)
    out = _sdpa(q, k, v, scale=scale)

    # Reshape back: (B, H, T, D) -> (B, T, H*D)
    return out.transpose(0, 2, 1, 3).reshape(b, t_q, heads * dim_head)


def _attention_core_inline_no_mask_kv_downsample(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
) -> mx.array:
    """Inline no-mask attention with experimental spatial K/V pooling."""
    b, t_q, _ = q.shape
    _, t_k, _ = k.shape

    q = q.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)
    k = k.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)
    v = v.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)
    k = _pool_spatial_heads(k)
    v = _pool_spatial_heads(v)

    scale = 1.0 / (dim_head ** 0.5)
    out = _sdpa(q, k, v, scale=scale)

    return out.transpose(0, 2, 1, 3).reshape(b, t_q, heads * dim_head)


def _attention_core_inline_with_mask(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
    mask: mx.array,
) -> mx.array:
    """Inline (uncompiled) attention core with mask."""
    b, t_q, _ = q.shape
    _, t_k, _ = k.shape

    # Reshape for multi-head attention: (B, T, H*D) -> (B, H, T, D)
    q = q.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)
    k = k.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)
    v = v.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)

    # Handle mask dimensions
    if mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]

    # Boolean masks pass through unchanged — MLX SDPA handles them natively
    # (True = attend, False = mask out, equivalent to additive -inf at False).
    # Additive masks (e.g. context_mask with -inf at padding) must match q dtype.
    # An unconditional .astype(q.dtype) here would cast bool → 0/1 BF16, which
    # MLX SDPA then interprets as a soft additive bias (~2.72× preference
    # factor) rather than a hard mask — a real bug for any caller passing
    # bool.  Dormant on main today (no bool callers), kept correct preemptively.
    if mask.dtype != mx.bool_:
        mask = mask.astype(q.dtype)

    # Compute attention using Flash Attention
    scale = 1.0 / (dim_head ** 0.5)
    out = _sdpa(q, k, v, scale=scale, mask=mask)

    # Reshape back: (B, H, T, D) -> (B, T, H*D)
    return out.transpose(0, 2, 1, 3).reshape(b, t_q, heads * dim_head)


# Compiled attention core (without mask) - fuses reshape + SDPA + reshape.
# Falls through to the inline version when LTX_DISABLE_COMPILED_ATTN is set.
if _USE_COMPILED_ATTN:
    _compiled_attention_core_no_mask = mx.compile(_attention_core_inline_no_mask)
    _compiled_attention_core_with_mask = mx.compile(_attention_core_inline_with_mask)
else:
    _compiled_attention_core_no_mask = _attention_core_inline_no_mask
    _compiled_attention_core_with_mask = _attention_core_inline_with_mask


def _attention_core(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Dispatch to compiled attention core based on mask presence."""
    if mask is None:
        if _KV_DOWNSAMPLE_CONFIG is not None:
            use_kv_pool, reason = _kv_downsample_candidate(q, k, v, heads, dim_head)
            if use_kv_pool:
                _kv_downsample_record(None)
                return _attention_core_inline_no_mask_kv_downsample(
                    q, k, v, heads, dim_head
                )
            _kv_downsample_record(reason)
        return _compiled_attention_core_no_mask(q, k, v, heads, dim_head)
    else:
        return _compiled_attention_core_with_mask(q, k, v, heads, dim_head, mask)


def rms_norm(x: mx.array, weight: Optional[mx.array] = None, eps: float = 1e-6) -> mx.array:
    """
    Apply RMS normalization using optimized MLX implementation.

    Args:
        x: Input tensor.
        weight: Optional learnable scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        RMS normalized tensor.
    """
    return mx.fast.rms_norm(x, weight, eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array] = None,
    scale: Optional[float] = None,
) -> mx.array:
    """
    Scaled dot-product attention using optimized MLX implementation.

    Uses mx.fast.scaled_dot_product_attention which is memory-efficient
    (Flash Attention style) and runs optimized Metal kernels.

    Args:
        q: Query tensor of shape (B, H, T_q, D).
        k: Key tensor of shape (B, H, T_k, D).
        v: Value tensor of shape (B, H, T_k, D).
        mask: Optional attention mask.
        scale: Optional scale factor (default: 1/sqrt(D)).

    Returns:
        Attention output of shape (B, H, T_q, D).
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    return _sdpa(q, k, v, scale=scale, mask=mask)


class Attention(nn.Module):
    """
    Multi-head attention with RMSNorm on Q/K and optional RoPE.

    This attention module follows the LTX-2 architecture:
    - RMSNorm applied to Q and K before attention
    - RoPE applied to Q and K (if position embeddings provided)
    - Standard scaled dot-product attention
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,  # PyTorch default
        apply_gated_attention: bool = False,
    ):
        """
        Initialize attention module.

        Args:
            query_dim: Dimension of query input.
            context_dim: Dimension of key/value input (defaults to query_dim).
            heads: Number of attention heads.
            dim_head: Dimension per head.
            norm_eps: Epsilon for RMSNorm.
            rope_type: Type of RoPE to use.
            apply_gated_attention: Per-head gating (V2).
        """
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        context_dim = query_dim if context_dim is None else context_dim

        # RMSNorm for Q and K
        self.q_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = RMSNorm(inner_dim, eps=norm_eps)

        # Linear projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating (V2)
        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        # Output projection
        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)
        self._to_out_weight_t = None
        self._to_q_weight_t = None
        self._to_k_weight_t = None
        self._to_v_weight_t = None
        self._to_gate_logits_weight_t = None  # only used when to_gate_logits is set

    def _projection_call(
        self,
        linear: nn.Linear,
        cached_t: Optional[mx.array],
        x: mx.array,
    ) -> mx.array:
        """Apply a Linear, using a pre-transposed weight cache when present."""
        if cached_t is None:
            return linear(x)
        bias = linear.get("bias")
        if bias is not None:
            return mx.addmm(bias, x, cached_t)
        return x @ cached_t

    def _to_out(self, x: mx.array) -> mx.array:
        """Run to_out, optionally using a pre-transposed contiguous weight."""
        return self._projection_call(self.to_out, self._to_out_weight_t, x)

    def _pretranspose_linear(
        self,
        linear_attr: str,
        cache_attr: str,
    ) -> list[mx.array]:
        """Cache ``mx.contiguous(weight.T)`` on this Attention for ``linear_attr``."""
        linear = getattr(self, linear_attr)
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"{linear_attr} pretranspose only supports nn.Linear")
        cached = getattr(self, cache_attr)
        if cached is not None:
            arrays = [cached]
            bias = linear.get("bias")
            if bias is not None:
                arrays.append(bias)
            return arrays
        if "weight" not in linear:
            raise ValueError(f"{linear_attr} weight is unavailable for pretranspose")

        new_t = mx.contiguous(linear.weight.T)
        setattr(self, cache_attr, new_t)
        arrays = [new_t]
        bias = linear.get("bias")
        if bias is not None:
            arrays.append(bias)
        return arrays

    def pretranspose_to_out(self) -> list[mx.array]:
        """Cache a contiguous ``weight.T`` for to_out same-math experiments."""
        return self._pretranspose_linear("to_out", "_to_out_weight_t")

    def pretranspose_to_q(self) -> list[mx.array]:
        return self._pretranspose_linear("to_q", "_to_q_weight_t")

    def pretranspose_to_k(self) -> list[mx.array]:
        return self._pretranspose_linear("to_k", "_to_k_weight_t")

    def pretranspose_to_v(self) -> list[mx.array]:
        return self._pretranspose_linear("to_v", "_to_v_weight_t")

    def pretranspose_to_gate_logits(self) -> list[mx.array]:
        # No-op when this Attention is non-V2 (to_gate_logits is None).
        if self.to_gate_logits is None:
            return []
        return self._pretranspose_linear("to_gate_logits", "_to_gate_logits_weight_t")

    # Map layout-spec target → (apply method name, cache attr name).
    _PRETRANSPOSE_TARGETS: dict[str, tuple[str, str]] = {
        "to_out": ("pretranspose_to_out", "_to_out_weight_t"),
        "to_q":   ("pretranspose_to_q",   "_to_q_weight_t"),
        "to_k":   ("pretranspose_to_k",   "_to_k_weight_t"),
        "to_v":   ("pretranspose_to_v",   "_to_v_weight_t"),
        "to_gate_logits": ("pretranspose_to_gate_logits", "_to_gate_logits_weight_t"),
    }

    def drop_layout_sources(
        self,
        layout_specs: tuple[tuple[str, str], ...],
    ) -> None:
        """Drop original arrays replaced by materialized layout transforms."""
        for target, layout in layout_specs:
            if layout != "pretranspose" or target not in self._PRETRANSPOSE_TARGETS:
                raise ValueError(f"Unsupported attention layout spec: {target}:{layout}")
            _, cache_attr = self._PRETRANSPOSE_TARGETS[target]
            cached = getattr(self, cache_attr)
            linear = getattr(self, target, None)
            # to_gate_logits can be None on non-V2 attentions — skip silently.
            if linear is None:
                continue
            if cached is not None and "weight" in linear:
                del linear.weight

    def apply_layouts(
        self,
        layout_specs: tuple[tuple[str, str], ...],
    ) -> list[mx.array]:
        """Apply selected same-math layout transforms to attention projections."""
        arrays: list[mx.array] = []
        for target, layout in layout_specs:
            if layout != "pretranspose" or target not in self._PRETRANSPOSE_TARGETS:
                raise ValueError(f"Unsupported attention layout spec: {target}:{layout}")
            apply_method, _ = self._PRETRANSPOSE_TARGETS[target]
            arrays.extend(getattr(self, apply_method)())
        return arrays

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        pe: Optional[tuple] = None,
        k_pe: Optional[tuple] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Query input of shape (B, T, D).
            context: Key/Value input (defaults to x for self-attention).
            mask: Optional attention mask.
            pe: Position embeddings for Q and K (cos, sin tuple).
            k_pe: Separate position embeddings for K (if different from Q).

        Returns:
            Attention output of shape (B, T, D).

        Note on op order: compute gate logits FIRST, then V, then Q/K.  This
        matches mlx-video's pattern, which lets MLX's lazy graph pipeline the
        gate-logits matmul alongside the V/Q/K projection matmuls instead of
        serializing gate after SDPA.  At small T this measurably reduces
        per-step time; at large T the effect is in the noise but still neutral.
        """
        # Sub-phase signposts (no-op when LTX_PROFILE_SIGNPOSTS is unset).
        # Three regions: qkv (projections + norms + rope) → sdpa → out.
        # Aggregates across all attention call sites; the parent phase
        # signpost (video_self_attn / video_text_ca / etc.) wraps the whole
        # call so trace tools can correlate.
        with _signpost("attn_qkv"):
            # 1) Gate logits first — independent of V/Q/K, lets MLX schedule it
            #    in parallel with the projections below.
            gate = None
            if self.to_gate_logits is not None:
                # 2 * sigmoid so zero-init weight gives identity (2 * 0.5 = 1.0)
                gate_logits = self._projection_call(
                    self.to_gate_logits, self._to_gate_logits_weight_t, x,
                )
                gate = 2.0 * mx.sigmoid(gate_logits)  # (B, T, H)

            # 2) Project V before Q, K (mlx-video order).
            context = x if context is None else context
            v = self._projection_call(self.to_v, self._to_v_weight_t, context)
            q = self._projection_call(self.to_q, self._to_q_weight_t, x)
            k = self._projection_call(self.to_k, self._to_k_weight_t, context)

            # 3) Apply RMSNorm to Q and K
            q = self.q_norm(q)
            k = self.k_norm(k)

            # 4) Apply RoPE if position embeddings provided
            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
            _sp_barrier(q, k, v)

        with _signpost("attn_sdpa"):
            # 5) Attention core (compiled or inline depending on env toggle).
            out = _attention_core(q, k, v, self.heads, self.dim_head, mask)
            _sp_barrier(out)

        with _signpost("attn_out"):
            # 6) Apply per-head gating if enabled (V2).
            if gate is not None:
                b, t, _ = out.shape
                out = out.reshape(b, t, self.heads, self.dim_head)
                out = out * gate[:, :, :, None]  # (B, T, H, D) * (B, T, H, 1)
                out = out.reshape(b, t, self.heads * self.dim_head)

            # 7) Output projection
            out = self._to_out(out)
            _sp_barrier(out)
        return out

    def profile(
        self,
        x: mx.array,
        name: str,
        mark_profile: Callable[..., None],
        context: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        pe: Optional[tuple] = None,
        k_pe: Optional[tuple] = None,
    ) -> mx.array:
        """Forward pass with forced-eval timing checkpoints for diagnostics."""
        q = self._projection_call(self.to_q, self._to_q_weight_t, x)
        mark_profile(f"{name} q", q)

        context = x if context is None else context
        k = self._projection_call(self.to_k, self._to_k_weight_t, context)
        mark_profile(f"{name} k", k)

        v = self._projection_call(self.to_v, self._to_v_weight_t, context)
        mark_profile(f"{name} v", v)

        q = self.q_norm(q)
        k = self.k_norm(k)
        mark_profile(f"{name} qk norm", q, k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
            mark_profile(f"{name} rope", q, k)

        out = _attention_core(q, k, v, self.heads, self.dim_head, mask)
        mark_profile(f"{name} sdpa", out)

        if self.to_gate_logits is not None:
            gate_logits = self._projection_call(
                self.to_gate_logits, self._to_gate_logits_weight_t, x,
            )
            mark_profile(f"{name} gate logits", gate_logits)
            b, t, _ = out.shape
            out = out.reshape(b, t, self.heads, self.dim_head)
            gates = 2.0 * mx.sigmoid(gate_logits)
            out = out * gates[:, :, :, None]
            out = out.reshape(b, t, self.heads * self.dim_head)
            mark_profile(f"{name} gate apply", out)

        out = self._to_out(out)
        mark_profile(f"{name} out", out)
        return out


class SelfAttention(nn.Module):
    """Self-attention layer (convenience wrapper)."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
    ):
        super().__init__()
        self.attn = Attention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        pe: Optional[tuple] = None,
    ) -> mx.array:
        return self.attn(x, context=None, mask=mask, pe=pe)


class CrossAttention(nn.Module):
    """Cross-attention layer (convenience wrapper)."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Cross-attention typically doesn't use RoPE
        self.attn = Attention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        return self.attn(x, context=context, mask=mask, pe=None)
