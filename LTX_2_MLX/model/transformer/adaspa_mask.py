"""
AdaSpa (Adaptive Sparse Attention) for LTX-2-MLX.

Reference: "Training-free and Adaptive Sparse Attention for Efficient Long Video
Generation" (Xia et al., ICCV 2025) — https://arxiv.org/abs/2502.21079

Core algorithm (Algorithm 2: LSE-Cached Online Search):
1. At search step (the first call after `t_w` warmup steps, then again at refresh
   steps `Ts`): run dense attention with LSE extracted per query row.  Use LSE to
   compute block-summed attention weights `W_sum_attn[p, q] = Σ exp(S[i,j] −
   LSE[i])` for each (Q-block p, K-block q) pair.  Top-k per Q-block gives a
   binary block mask, kept as state until the next search step.

2. Between search steps: use block-sparse SDPA with the cached mask
   (reuses `radial_sparse_attention`).

Differences from the paper in this v1 port:
- **Shared mask across heads** instead of per-head.  Aggregates W_sum_attn across
  the head axis before top-k.  Simpler dispatch (one mask per attention call
  instead of 32) at a quality cost.  Per-head variant is a v2 follow-up if v1
  validates.
- **Block_size auto-aligned to tpf** (same picker as radial_mask).  Paper uses
  fixed block_size=64 which would create the same misalignment artifacts we
  saw with radial on LTX.

Quality risk: AdaSpa was published as tested on 50-step models with t_w=10.
LTX 2.3 distilled has 11 total steps; we'll have to use t_w=1 or 2.  Whether
attention patterns at step 2 of distilled are predictive of patterns at step
8 is empirically unknown.  This port is a probe, not a guaranteed fix.

Memory: the LSE pass needs the full (B, H, T, T) attention matrix in memory.
At LTX bakery stage 2 (T=35136, H=32, BF16), that's ~76 GB — won't fit on
M1 Max.  v1 limits AdaSpa to attention shapes where the matrix fits (stage 1
T=8784 = 290 MB; 384×640 stage 2 T=14640 = 13 GB — borderline).  Stage 2 at
full bakery resolution is gated on v2's tiled LSE computation.
"""
from __future__ import annotations

from typing import Tuple

import mlx.core as mx


def choose_adaspa_block_size(tpf: int, max_size: int = 64) -> int:
    """Pick the largest divisor of tpf that is <= max_size.

    Same frame-alignment logic as radial_mask's picker.  Default max_size=64
    matches the paper; smaller for tpf<64.
    """
    bs = min(max_size, tpf)
    while bs > 1 and tpf % bs != 0:
        bs -= 1
    return bs


def adaspa_search(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    block_size: int,
    sparsity: float,
) -> Tuple[mx.array, mx.array]:
    """Dense attention with LSE-derived block mask.

    Args:
        q, k, v: (B, H, T, D) tensors.  T must be divisible by `block_size`
            (caller pads if needed; we don't pad here).
        scale: typical 1/sqrt(D).
        block_size: token granularity of the mask.
        sparsity: fraction of K-blocks to MASK OUT per Q-block.  E.g., 0.5 =
            keep top 50% by attention weight.

    Returns:
        (attn_output, block_mask) where
        - attn_output: (B, H, T, D) — the dense attention output for this step
        - block_mask: (n_qb, n_kb) bool, shared across batch and heads.
            True = "this Q-block attends to this K-block".

    Memory: materializes the full (B, H, T, T) attention matrix.  Caller must
    ensure the shape fits — see module docstring.
    """
    B, H, T, D = q.shape
    assert k.shape == (B, H, T, D) and v.shape == (B, H, T, D)
    assert T % block_size == 0, (
        f"adaspa_search: T={T} must be divisible by block_size={block_size}"
    )

    # Pass 1: full attention matrix + LSE per row.  This is the same compute as
    # mx.fast.scaled_dot_product_attention but we need the intermediate values.
    S = (q @ k.swapaxes(-2, -1)) * scale  # (B, H, T, T)
    LSE = mx.logsumexp(S, axis=-1, keepdims=True)  # (B, H, T, 1)
    attn_weights = mx.exp(S - LSE)  # (B, H, T, T)
    attn_output = attn_weights @ v  # (B, H, T, D)

    # Pass 2: block-sum attention weights to build the mask.
    n_qb = T // block_size
    n_kb = T // block_size
    # Reshape (B, H, T, T) -> (B, H, n_qb, bs, n_kb, bs)
    blocked = attn_weights.reshape(B, H, n_qb, block_size, n_kb, block_size)
    # Sum over the in-block axes (axes 3 and 5)
    W_sum = blocked.sum(axis=(3, 5))  # (B, H, n_qb, n_kb)
    # Aggregate across batch + head — shared mask in v1.
    W_sum_shared = W_sum.sum(axis=(0, 1))  # (n_qb, n_kb)

    # Per-Q-block top-k selection.
    k_keep = max(1, int(round((1.0 - sparsity) * n_kb)))
    # mx.argpartition would be ideal; argsort works and we only need top indices.
    # argsort ascending → take last k_keep entries per row.
    sorted_idx = mx.argsort(W_sum_shared, axis=-1)  # (n_qb, n_kb)
    topk_idx = sorted_idx[:, -k_keep:]  # (n_qb, k_keep)

    # Build the bool mask via scatter.  MLX doesn't have a 1-call scatter for
    # bool; expand topk_idx into a (n_qb, n_kb) mask manually.
    block_mask = _topk_idx_to_mask(topk_idx, n_qb, n_kb)
    # Force diagonal to True so self-attention within each block is preserved
    # even at extreme sparsity (matches radial's guarantee).
    diag = mx.eye(n_qb, dtype=mx.bool_)
    block_mask = mx.logical_or(block_mask, diag)
    mx.eval(block_mask)
    return attn_output, block_mask


def _topk_idx_to_mask(topk_idx: mx.array, n_qb: int, n_kb: int) -> mx.array:
    """Convert (n_qb, k_keep) topk column indices to (n_qb, n_kb) bool mask."""
    # Use broadcasting: for each (row r, col c), True iff c ∈ topk_idx[r].
    # Build via outer-equality: (n_qb, n_kb, 1) == (n_qb, 1, k_keep).
    cols = mx.arange(0, n_kb, dtype=topk_idx.dtype).reshape(1, n_kb, 1)  # (1, n_kb, 1)
    idx_b = topk_idx.reshape(n_qb, 1, -1)  # (n_qb, 1, k_keep)
    mask = (cols == idx_b).any(axis=-1)  # (n_qb, n_kb)
    return mask


def mask_density(block_mask: mx.array) -> float:
    """Fraction of True entries in the block mask."""
    return float(block_mask.astype(mx.float32).mean().item())
