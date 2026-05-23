"""
Radial Attention — path (b): gather-based block-sparse SDPA.

Path (a) (LTX_2_MLX/model/transformer/radial_mask.py + transformer.py wiring)
passes a boolean mask to mx.fast.scaled_dot_product_attention so the existing
MLX SDPA kernel still computes full QK^T then masks — quality probe only, NO
speed win.

Path (b) (this file) replaces the full-QK^T compute with a per-Q-block dense
SDPA, where each Q-block attends only to the K-blocks selected by the radial
mask.  This is where the actual sparsity savings live.

NOT wired in yet — gated on path (a) clearing a quality bar at bakery scale.
This module is a self-contained design that can be tested in isolation
against the dense baseline before integration.

Algorithm
---------
Given:
- Q, K, V of shape (B, H, T, D)
- block_mask (B_blocks, B_blocks) bool, where B_blocks = T // block_size
- block_size (default 128, matches the radial mask construction)

For each Q-block i in [0, B_blocks):
  1. active = indices j where block_mask[i, j] == True
  2. gather K_active = K[:, :, active * block_size : (active + 1) * block_size, :]
     gather V_active = V[:, :, active * block_size : (active + 1) * block_size, :]
     (concretely: K reshaped to (B, H, B_blocks, block_size, D), then mx.take
      along axis 2 with `active` indices, then flatten the active * block_size
      dimensions.)
  3. Q_i = Q[:, :, i*block_size:(i+1)*block_size, :]
  4. out_i = mx.fast.scaled_dot_product_attention(Q_i, K_active, V_active, scale)

Concatenate out_i along axis 2 → final output (B, H, B_blocks * block_size, D).

If T is not divisible by block_size, the trailing tail tokens are handled by
a single dense SDPA against the full K/V — saves no compute on the tail but
keeps math exact.

Expected compute savings
------------------------
At bakery stage 1 (T=8784, mask density 33.79%):
- Dense:  ~33 BFLOPs per video_self_attn
- Sparse: ~11 BFLOPs (1 - 0.6621 = 33.79%)
- SDPA wall: ~11 s/step → expected ~4 s/step (savings ~7 s)
- Step wall: ~45.5 s → expected ~38–39 s (after dispatch tax)

Bakery stage 2 (T=35136, density 30.48% at decay=0.5):
- Stage 2 wall: 313 s/step → expected ~245 s/step
- Stage 2 total: 15m 40s → expected ~12m 15s

These are upper bounds — Python-side dispatch overhead and gather memory
traffic will eat some of the headroom.

Dispatch overhead
-----------------
At bakery stage 1: 69 Q-blocks × 48 transformer blocks × 8 steps × 1 video
self-attn per block = 26,496 SDPA dispatches per stage 1 (vs. 384 today).
At ~50 µs per dispatch that's 1.3 s of pure Python→MLX dispatch tax on
stage 1.  Tolerable — under 3% of step time.

Memory
------
Per-call K_active gather is transient: ~(B × H × active_kblocks × block_size
× D × 2) bytes.  At bakery stage 1 with mean 23 active blocks: ~24 MB per
call.  MLX's allocator cache should reuse this aggressively.

References
----------
- Algorithm from mit-han-lab/radial-attention (arXiv:2506.19852) but the
  block-sparse SDPA structure is independently implementable.
- block_mask comes from LTX_2_MLX.model.transformer.radial_mask.radial_block_mask
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx

from LTX_2_MLX.model.transformer.radial_mask import radial_block_mask


def _active_kblocks_per_row(block_mask: mx.array) -> List[List[int]]:
    """For each Q-block row, return a Python list of active K-block column indices.

    Computed on host because per-row active counts vary (ragged) — we use the
    lists as static loop bounds in the Python-side per-Q-block dispatch.
    Called once per (shape, decay) and cached at the call site.
    """
    n = int(block_mask.shape[0])
    # Force materialization, then move to a Python-side bool list.
    mx.eval(block_mask)
    rows = block_mask.astype(mx.int32).tolist()
    return [
        [j for j, v in enumerate(row) if v]
        for row in rows
    ]


def _gather_kv_blocks(
    k_aligned_blocked: mx.array,
    v_aligned_blocked: mx.array,
    k_tail: Optional[mx.array],
    v_tail: Optional[mx.array],
    active_blocks: List[int],
    block_size: int,
) -> tuple[mx.array, mx.array]:
    """Gather K and V at the given block indices, plus the unaligned tail.

    Args:
        k_aligned_blocked: (B, H, n_blocks, block_size, D) — block-aligned K
            prefix, pre-reshaped by the caller.
        v_aligned_blocked: same for V.
        k_tail: (B, H, T_tail, D) when T is not block_size-aligned; None when
            T % block_size == 0.  These tokens are always attended (matches
            radial_mask._expand_block_to_token's tail behavior).
        v_tail: same for V.
        active_blocks: Python list of active K-block indices.
        block_size: token granularity (must match k_aligned_blocked.shape[3]).

    Returns:
        (k_gather, v_gather), each (B, H, n_active * block_size + T_tail, D).
    """
    B, H, _, _, D = k_aligned_blocked.shape
    idx = mx.array(active_blocks, dtype=mx.int32)
    k_take = mx.take(k_aligned_blocked, idx, axis=2)
    v_take = mx.take(v_aligned_blocked, idx, axis=2)
    n_active = len(active_blocks)
    k_gather = k_take.reshape(B, H, n_active * block_size, D)
    v_gather = v_take.reshape(B, H, n_active * block_size, D)
    if k_tail is not None:
        k_gather = mx.concatenate([k_gather, k_tail], axis=2)
        v_gather = mx.concatenate([v_gather, v_tail], axis=2)
    return k_gather, v_gather


def radial_sparse_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    block_mask: mx.array,
    block_size: int = 128,
    active_per_qblock: Optional[List[List[int]]] = None,
) -> mx.array:
    """Block-sparse SDPA driven by a (B_blocks, B_blocks) bool mask.

    Args:
        q, k, v: (B, H, T, D) tensors.  T need not be divisible by block_size;
            the unaligned tail is handled by one dense SDPA against the full K/V.
        scale: SDPA scale (typically 1.0 / sqrt(D)).
        block_mask: (B_blocks, B_blocks) bool, where B_blocks = T // block_size.
            block_mask[i, j] == True means Q-block i attends to K-block j.
        block_size: token granularity of the mask (must match what built the mask).
        active_per_qblock: optional precomputed Python lists of active K-block
            indices per Q-block row.  When None, derived from block_mask via a
            one-time host materialization.  Pass when the mask is reused across
            many calls (caller caches outside).

    Returns:
        (B, H, T, D) attention output.

    Math is exactly equivalent to dense SDPA with the same boolean mask
    expanded to token level — modulo BF16 rounding-order drift from doing
    SDPA on smaller tiles.  Equivalence test lives in
    `scripts/test_radial_sparse.py`.
    """
    B, H, T, D = q.shape
    assert k.shape == (B, H, T, D), "K shape mismatch"
    assert v.shape == (B, H, T, D), "V shape mismatch"
    n_blocks = T // block_size
    tail_start = n_blocks * block_size
    has_tail = tail_start < T

    if active_per_qblock is None:
        active_per_qblock = _active_kblocks_per_row(block_mask)
    assert len(active_per_qblock) == n_blocks, (
        f"active_per_qblock len {len(active_per_qblock)} != n_blocks {n_blocks}"
    )

    # Pre-reshape the block-aligned K/V prefix once.  The reshape is view-like
    # under MLX; only the per-qi take materializes.
    k_aligned = k[:, :, :tail_start, :].reshape(B, H, n_blocks, block_size, D)
    v_aligned = v[:, :, :tail_start, :].reshape(B, H, n_blocks, block_size, D)
    k_tail = k[:, :, tail_start:, :] if has_tail else None
    v_tail = v[:, :, tail_start:, :] if has_tail else None

    out_blocks: List[mx.array] = []
    for qi in range(n_blocks):
        active = active_per_qblock[qi]
        if not active:
            # Should not happen — diagonal is force-True in radial mask — but
            # guard so the loop is total.  Return zeros for this Q-block.
            out_blocks.append(mx.zeros((B, H, block_size, D), dtype=q.dtype))
            continue
        q_block = q[:, :, qi * block_size : (qi + 1) * block_size, :]
        k_gather, v_gather = _gather_kv_blocks(
            k_aligned, v_aligned, k_tail, v_tail, active, block_size
        )
        out_block = mx.fast.scaled_dot_product_attention(
            q_block, k_gather, v_gather, scale=scale
        )
        out_blocks.append(out_block)

    out_main = mx.concatenate(out_blocks, axis=2)

    if has_tail:
        # Unaligned tail Q: attend to all K — matches radial_mask's tail
        # (last seq_len % block_size rows pad-True in _expand_block_to_token).
        q_tail = q[:, :, tail_start:, :]
        out_tail_q = mx.fast.scaled_dot_product_attention(q_tail, k, v, scale=scale)
        out_main = mx.concatenate([out_main, out_tail_q], axis=2)

    return out_main


def precompute_active_kblocks(
    seq_len: int,
    num_frame: int,
    decay_factor: float = 0.5,
    block_size: int = 128,
    model_type: str = "ltx",
) -> List[List[int]]:
    """Compute the active-K-block lists per Q-block row, cached at the radial
    block-mask level.  Stable per (seq_len, num_frame, decay_factor).

    Call once at pipeline setup, pass the result to `radial_sparse_attention`
    repeatedly across all blocks/steps.
    """
    bm = radial_block_mask(
        seq_len=seq_len,
        num_frame=num_frame,
        decay_factor=decay_factor,
        block_size=block_size,
        model_type=model_type,  # type: ignore[arg-type]
    )
    return _active_kblocks_per_row(bm)
