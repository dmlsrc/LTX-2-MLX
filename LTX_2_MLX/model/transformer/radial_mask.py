"""
Radial Attention mask construction for LTX-2-MLX (path-a quality probe).

Pure-MLX port of mit-han-lab/radial-attention's gen_log_mask_shrinked.  Produces a
boolean (T, T) attention mask where True = attend, False = mask out.  Returned mask
is passed to mx.fast.scaled_dot_product_attention's `mask=` argument so the existing
MLX SDPA kernel still runs full QK^T compute — this path is a quality oracle, not a
speed win.  The expansion from block-level (68×68 at bakery stage 1) to token-level
(8784×8784) lives in this file too.

Algorithm reference:
    https://github.com/mit-han-lab/radial-attention/blob/main/radial_attn/attn_mask.py
    paper: arXiv:2506.19852

Notes / deltas from the reference:
- model_type "ltx" is added as a synonym for the wan handling
  (dist <= 1 → full intra-pair attention, no diagonal split).
- The mask is computed block-by-block at block_size=128 granularity per the paper,
  then expanded to token-level just before SDPA.
- The reference has two zero-effect lines (`mask[0:0] = True`, `mask[-1:-1] = True`)
  in shrinkMaskStrict — empty slices, no-op.  Omitted here.
- Caching is keyed on (seq_len, num_frames, decay_factor, model_type, block_size).
  Cache size 8 is enough for stage-1 + stage-2 across a couple of A/B configs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import mlx.core as mx


def _window_width(
    i: int,
    j: int,
    token_per_frame: int,
    num_frame: int,
    decay_factor: float,
    block_size: int,
    model_type: str,
) -> int:
    """Intra-frame-pair attention window width (radial decay)."""
    dist = abs(i - j)
    if model_type in ("wan", "ltx"):
        if dist <= 1:
            return token_per_frame
    elif model_type == "hunyuan":
        if dist <= 1:
            return token_per_frame
    else:
        raise ValueError(f"Unknown radial model_type: {model_type!r}")
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * decay_factor
    if decay_length >= block_size:
        return int(decay_length)
    return block_size


def _diagonal_split_keep(i: int, j: int, token_per_frame: int, block_size: int) -> bool:
    """Frame-pair-level temporal subsampling.

    At distances where the decay window is narrower than the block threshold, only
    every `split_factor`-th frame-pair is kept active.  Returns True if frame-pair
    (i, j) survives the subsampling.
    """
    dist = abs(i - j)
    if dist == 0:
        return True
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group
    if decay_length >= block_size:
        return True
    split_factor = int(block_size / decay_length)
    return (dist % split_factor) == 0


def _shrink_block(
    padded_pair_mask: mx.array,
    block_size: int,
) -> mx.array:
    """Reduce a token-level mask block to block-level density check.

    Reference logic:
      - Reshape (rows, cols) into (block_rows, block_size, block_cols, block_size)
      - For each (block_row, block_col): col_density[col_in_block] = mean over the
        block_size rows.
      - high_density_cols = col_density > 1/3
      - non_zero_cols = col_density > 0
      - keep block iff (high_density_cols / non_zero_cols) > 0.6
    """
    rows, cols = padded_pair_mask.shape
    br = rows // block_size
    bc = cols // block_size
    if br == 0 or bc == 0:
        return mx.zeros((br, bc), dtype=mx.bool_)
    # (br, block_size, bc, block_size)
    reshaped = padded_pair_mask[: br * block_size, : bc * block_size].reshape(
        br, block_size, bc, block_size
    )
    # col density along the inner block-row axis (axis=1)
    col_density = reshaped.astype(mx.float32).sum(axis=1) / float(block_size)
    # col_density shape: (br, bc, block_size)
    high = (col_density > (1.0 / 3.0)).astype(mx.float32).sum(axis=-1)
    nonzero = (col_density > 0.0).astype(mx.float32).sum(axis=-1)
    frac = high / (nonzero + 1e-9)
    return frac > 0.6


def _build_block_mask(
    seq_len: int,
    num_frame: int,
    decay_factor: float,
    block_size: int,
    model_type: str,
) -> mx.array:
    """Construct the (block_seq, block_seq) bool block-mask. Pure host loops."""
    token_per_frame = seq_len // num_frame
    block_seq = seq_len // block_size
    final_block_mask = mx.zeros((block_seq, block_seq), dtype=mx.bool_)

    # Precomputed within-frame row/col index grids for the |col - row| <= W test.
    col_idx = mx.arange(0, token_per_frame).reshape(1, -1)
    row_idx = mx.arange(0, token_per_frame).reshape(-1, 1)
    abs_diff = mx.abs(col_idx - row_idx)

    for i in range(num_frame):
        for j in range(num_frame):
            # Frame-pair local intra-pair mask (token_per_frame × token_per_frame).
            if j == 0 and model_type in ("wan", "ltx"):
                # Attention sink: full attention to first frame.
                local = mx.ones((token_per_frame, token_per_frame), dtype=mx.bool_)
            elif not _diagonal_split_keep(i, j, token_per_frame, block_size):
                # Subsampled out — pair contributes no blocks.
                continue
            else:
                w = _window_width(
                    i, j, token_per_frame, num_frame, decay_factor, block_size, model_type
                )
                local = abs_diff <= w  # bool (token_per_frame, token_per_frame)

            # Place local mask into a block-aligned padded buffer, then shrink.
            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            n_blocks_row = (token_per_frame - 1) // block_size + 1
            n_blocks_col = (token_per_frame - 1) // block_size + 1
            pad_rows = remainder_row + n_blocks_row * block_size
            pad_cols = remainder_col + n_blocks_col * block_size

            padded = mx.zeros((pad_rows, pad_cols), dtype=mx.bool_)
            # MLX doesn't support boolean fancy-set; build via concatenation.
            # Equivalently: pad above/below + left/right with zeros.
            top = mx.zeros((remainder_row, pad_cols), dtype=mx.bool_)
            mid_left = mx.zeros((token_per_frame, remainder_col), dtype=mx.bool_)
            mid_right = mx.zeros(
                (token_per_frame, pad_cols - remainder_col - token_per_frame), dtype=mx.bool_
            )
            mid = mx.concatenate([mid_left, local, mid_right], axis=1)
            bottom = mx.zeros((pad_rows - remainder_row - token_per_frame, pad_cols), dtype=mx.bool_)
            padded = mx.concatenate([top, mid, bottom], axis=0)

            block_shrunk = _shrink_block(padded, block_size)
            bh = block_shrunk.shape[0]
            bw = block_shrunk.shape[1]

            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = min(block_row_start + bh, block_seq)
            block_col_end = min(block_col_start + bw, block_seq)
            if block_row_end <= block_row_start or block_col_end <= block_col_start:
                continue
            bh_eff = block_row_end - block_row_start
            bw_eff = block_col_end - block_col_start

            # OR block_shrunk[:bh_eff, :bw_eff] into final_block_mask[rs:re, cs:ce].
            # MLX has no in-place set on bool slices; rebuild via concat.
            chunk = block_shrunk[:bh_eff, :bw_eff]
            existing = final_block_mask[block_row_start:block_row_end, block_col_start:block_col_end]
            merged = mx.logical_or(existing, chunk)
            final_block_mask = _set_block(
                final_block_mask, block_row_start, block_col_start, merged
            )

    # Force diagonal to True — guarantees self-attention even at extreme decay.
    diag_idx = mx.arange(0, block_seq)
    final_block_mask = _set_diagonal_true(final_block_mask, diag_idx)

    return final_block_mask


def _set_block(
    mask: mx.array, row_start: int, col_start: int, sub: mx.array
) -> mx.array:
    """Set mask[row_start:row_start+sub.shape[0], col_start:col_start+sub.shape[1]] = sub.

    MLX arrays support direct slice assignment via the .at-style API in newer versions,
    but for portability we use index_put-style concat.  Cost is amortized away by the
    mask cache (computed once per shape).
    """
    rows = mask.shape[0]
    cols = mask.shape[1]
    h, w = sub.shape
    # Build full-width row block: left | sub | right
    left = mask[row_start : row_start + h, :col_start]
    right = mask[row_start : row_start + h, col_start + w :]
    mid_row = mx.concatenate([left, sub, right], axis=1)
    # Stitch above + mid_row + below
    top = mask[:row_start, :]
    bottom = mask[row_start + h :, :]
    return mx.concatenate([top, mid_row, bottom], axis=0)


def _set_diagonal_true(mask: mx.array, diag_idx: mx.array) -> mx.array:
    """Force the diagonal entries to True."""
    n = mask.shape[0]
    # OR with identity-shaped bool matrix.
    eye = mx.eye(n).astype(mx.bool_)
    return mx.logical_or(mask, eye)


def _expand_block_to_token(
    block_mask: mx.array, block_size: int, seq_len: int
) -> mx.array:
    """Expand (B, B) block mask to (seq_len, seq_len) token-level mask.

    Each block entry replicates to a (block_size, block_size) tile.  Crops to seq_len
    in case seq_len is not a multiple of block_size.
    """
    # repeat each block_size times along rows then cols
    expanded = mx.repeat(block_mask, block_size, axis=0)
    expanded = mx.repeat(expanded, block_size, axis=1)
    # Pad the tail with True if seq_len isn't aligned (no info loss; the dropped
    # tail in block-construction means those rows/cols default to True for
    # safety — equivalent to "attend to padded region").
    expanded_h, expanded_w = expanded.shape
    if expanded_h < seq_len:
        pad_h = mx.ones((seq_len - expanded_h, expanded_w), dtype=mx.bool_)
        expanded = mx.concatenate([expanded, pad_h], axis=0)
    if expanded_w < seq_len:
        pad_w = mx.ones((expanded.shape[0], seq_len - expanded_w), dtype=mx.bool_)
        expanded = mx.concatenate([expanded, pad_w], axis=1)
    return expanded[:seq_len, :seq_len]


@lru_cache(maxsize=8)
def radial_block_mask(
    seq_len: int,
    num_frame: int,
    decay_factor: float = 0.5,
    block_size: int = 128,
    model_type: Literal["ltx", "wan", "hunyuan"] = "ltx",
) -> mx.array:
    """Build the *block-level* (B, B) bool radial attention mask.

    B = seq_len // block_size.  Used by path-b (gather-based sparse SDPA),
    which needs the block granularity for active-K-block selection.

    Cached on (seq_len, num_frame, decay_factor, model_type, block_size).
    """
    if seq_len % num_frame != 0:
        raise ValueError(
            f"radial_block_mask: seq_len ({seq_len}) must be divisible by num_frame "
            f"({num_frame}). LTX token order is (frame, h_lat, w_lat) flattened."
        )
    token_per_frame = seq_len // num_frame
    if token_per_frame == 0:
        raise ValueError("radial_block_mask: token_per_frame is zero")

    block_mask = _build_block_mask(
        seq_len=seq_len,
        num_frame=num_frame,
        decay_factor=decay_factor,
        block_size=block_size,
        model_type=model_type,
    )
    mx.eval(block_mask)
    return block_mask


@lru_cache(maxsize=8)
def radial_mask(
    seq_len: int,
    num_frame: int,
    decay_factor: float = 0.5,
    block_size: int = 128,
    model_type: Literal["ltx", "wan", "hunyuan"] = "ltx",
) -> mx.array:
    """Build the token-level (seq_len, seq_len) bool radial attention mask.

    Used by path-a (masked dense SDPA quality probe).  Path-b should call
    `radial_block_mask` directly to avoid materializing the dense token mask.

    Cached: identical (seq_len, num_frame, decay_factor, model_type, block_size)
    returns the same MLX array reference.  Safe to call from a hot loop.

    The mask is materialized as bool (1 byte per element).  At LTX bakery stage 1
    (T=8784), this is 77 MB; at stage 2 (T=35136), 1.23 GB.  Stage-2 memory cost
    is non-trivial — path (b) gather-based sparse attention is the planned
    follow-up.
    """
    block_mask = radial_block_mask(
        seq_len=seq_len,
        num_frame=num_frame,
        decay_factor=decay_factor,
        block_size=block_size,
        model_type=model_type,
    )
    token_mask = _expand_block_to_token(block_mask, block_size, seq_len)
    mx.eval(token_mask)
    return token_mask


def mask_sparsity(token_mask: mx.array) -> float:
    """Fraction of False entries in the token-level mask (i.e. masked-out fraction)."""
    total = float(token_mask.size)
    kept = float(token_mask.astype(mx.float32).sum().item())
    return 1.0 - (kept / total)
