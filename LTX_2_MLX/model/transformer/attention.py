"""Attention mechanisms for LTX-2 Transformer."""

from collections.abc import Callable
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .rope import LTXRopeType, apply_rotary_emb


# Compiled attention core (without mask) - fuses reshape + SDPA + reshape
@mx.compile
def _compiled_attention_core_no_mask(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
) -> mx.array:
    """Compiled attention core without mask."""
    b, t_q, _ = q.shape
    _, t_k, _ = k.shape

    # Reshape for multi-head attention: (B, T, H*D) -> (B, H, T, D)
    q = q.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)
    k = k.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)
    v = v.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)

    # Compute attention using Flash Attention
    scale = 1.0 / (dim_head ** 0.5)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    # Reshape back: (B, H, T, D) -> (B, T, H*D)
    return out.transpose(0, 2, 1, 3).reshape(b, t_q, heads * dim_head)


# Compiled attention core (with mask) - fuses reshape + SDPA + reshape
@mx.compile
def _compiled_attention_core_with_mask(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
    mask: mx.array,
) -> mx.array:
    """Compiled attention core with mask."""
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

    # Ensure mask dtype matches query dtype for scaled_dot_product_attention
    mask = mask.astype(q.dtype)

    # Compute attention using Flash Attention
    scale = 1.0 / (dim_head ** 0.5)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    # Reshape back: (B, H, T, D) -> (B, T, H*D)
    return out.transpose(0, 2, 1, 3).reshape(b, t_q, heads * dim_head)


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

    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


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

    def _to_out(self, x: mx.array) -> mx.array:
        """Run to_out, optionally using a pre-transposed contiguous weight."""
        if self._to_out_weight_t is None:
            return self.to_out(x)

        bias = self.to_out.get("bias")
        if bias is not None:
            return mx.addmm(bias, x, self._to_out_weight_t)
        return x @ self._to_out_weight_t

    def pretranspose_to_out(self) -> list[mx.array]:
        """Cache a contiguous ``weight.T`` for to_out same-math experiments."""
        if not isinstance(self.to_out, nn.Linear):
            raise ValueError("to_out pretranspose only supports nn.Linear")
        if self._to_out_weight_t is not None:
            arrays = [self._to_out_weight_t]
            bias = self.to_out.get("bias")
            if bias is not None:
                arrays.append(bias)
            return arrays
        if "weight" not in self.to_out:
            raise ValueError("to_out weight is unavailable for pretranspose")

        self._to_out_weight_t = mx.contiguous(self.to_out.weight.T)
        arrays = [self._to_out_weight_t]
        bias = self.to_out.get("bias")
        if bias is not None:
            arrays.append(bias)
        return arrays

    def drop_layout_sources(
        self,
        layout_specs: tuple[tuple[str, str], ...],
    ) -> None:
        """Drop original arrays replaced by materialized layout transforms."""
        for target, layout in layout_specs:
            if target == "to_out" and layout == "pretranspose":
                if self._to_out_weight_t is not None and "weight" in self.to_out:
                    del self.to_out.weight
            else:
                raise ValueError(f"Unsupported attention layout spec: {target}:{layout}")

    def apply_layouts(
        self,
        layout_specs: tuple[tuple[str, str], ...],
    ) -> list[mx.array]:
        """Apply selected same-math layout transforms to attention projections."""
        arrays: list[mx.array] = []
        for target, layout in layout_specs:
            if target == "to_out" and layout == "pretranspose":
                arrays.extend(self.pretranspose_to_out())
            else:
                raise ValueError(f"Unsupported attention layout spec: {target}:{layout}")
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
        """
        # Project to Q, K, V
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        # Apply RMSNorm to Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if position embeddings provided
        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Use compiled attention core for better performance
        out = _attention_core(q, k, v, self.heads, self.dim_head, mask)

        # Apply per-head gating if enabled (V2)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            out = out.reshape(b, t, self.heads, self.dim_head)
            # 2 * sigmoid so zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * mx.sigmoid(gate_logits)
            out = out * gates[:, :, :, None]  # (B, T, H, D) * (B, T, H, 1)
            out = out.reshape(b, t, self.heads * self.dim_head)

        # Output projection
        return self._to_out(out)

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
        q = self.to_q(x)
        mark_profile(f"{name} q", q)

        context = x if context is None else context
        k = self.to_k(context)
        mark_profile(f"{name} k", k)

        v = self.to_v(context)
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
            gate_logits = self.to_gate_logits(x)
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
