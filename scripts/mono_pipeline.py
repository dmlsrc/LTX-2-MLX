#!/usr/bin/env python3
"""Monolithic inlined AV transformer forward — same-math reference impl.

This module is a library, not a standalone script.  It provides
``InlinedAVModel`` — a drop-in replacement for ``X0Model(LTXAVModel)`` that
the pipeline can call instead of the modular block stack.  The 48-block
forward, AdaLN preprocess, output projection, and X0 conversion are all
inlined into one function (``transformer_step``) with flat pretransposed
weights pulled from the loaded modular model.  Everything else (Gemma
encoder, VAE decoder, spatial upscaler, audio decoder, MP4 save) is reused
unchanged.

**Result (kept for historical record, see docs/PERFORMANCE.md):** wall
clock is neutral vs the modular path; final latent cosine similarity is
0.99922+ for both video and audio against a modular reference.  Inlining
does not reduce per-step cost — MLX's lazy graph already optimizes through
the ``nn.Module`` dispatch chain.  The remaining gap to mlx-video (when
it exists at any shape) is structural at the MLX kernel / Metal command-
buffer level, not abstraction-level.

The code is kept as a same-math reference: a documented, flat description
of the V2.3 distilled AV block math that doesn't require reading the
modular Module tree.  Useful for future investigators who want to know
what the model actually computes per step.

To use: ``stage2_harness.py`` checks ``LTX_MONO_INLINED=1`` and, if set,
replaces ``av_pipeline.transformer`` with ``InlinedAVModel(model)`` after
the pipeline is built.  Everything else in the harness stays the same so
the A/B isolates exactly the transformer forward.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# ─── Reused, non-transformer pieces ──────────────────────────────────────────
from LTX_2_MLX.model.transformer.rope import (  # noqa: E402
    LTXRopeType,
    apply_rotary_emb,
    precompute_freqs_cis,
)


# ─── Constants (V2.3 distilled, hardcoded for happy path) ────────────────────

VIDEO_INNER_DIM = 4096
AUDIO_INNER_DIM = 2048
VIDEO_HEADS = 32
AUDIO_HEADS = 32
VIDEO_HEAD_DIM = 128
AUDIO_HEAD_DIM = 64
NUM_BLOCKS = 48
NORM_EPS = 1e-6
ROPE_TYPE = LTXRopeType.SPLIT
POSITIONAL_EMBEDDING_THETA = 10000.0
ROPE_MAX_POS = [20, 2048, 2048]
CROSS_PE_MAX_POS = 20  # AUDIO_CROSS_PE_MAX_POS on LTXAVModel; both modalities use it.
AV_CA_TIMESTEP_SCALE = 1000
TIMESTEP_SCALE = 1000
ADALN_NUM_EMBEDDINGS = 9  # V2.3: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_q, scale_q, gate_q
PROMPT_ADALN_NUM = 2
AV_CA_NUM_SCALE_SHIFT = 4
AV_CA_NUM_GATE = 1


# ─── Helpers (thin wrappers, exactly the math the modular path does) ─────────


def rms_norm(x: mx.array, eps: float = NORM_EPS) -> mx.array:
    """Bare RMS norm — matches our ``rms_norm`` helper without a weight."""
    return mx.fast.rms_norm(x, None, eps)


def linear(x: mx.array, weight: mx.array, bias: Optional[mx.array] = None) -> mx.array:
    """Linear projection.  ``weight`` is row-major (out, in)."""
    # Match nn.Linear math: x @ weight.T + bias.  MLX will lazily handle the .T.
    if bias is not None:
        return mx.addmm(bias, x, weight.T)
    return x @ weight.T


def linear_pretransposed(
    x: mx.array,
    weight_t: mx.array,
    bias: Optional[mx.array] = None,
) -> mx.array:
    """Linear with a pre-transposed weight (shape (in, out), contiguous).

    Same math as ``linear`` but skips the implicit ``weight.T`` op.  This is
    what the modular path's ``Attention._projection_call`` / ``FeedForward._project_*``
    do when the layout cache is populated, which is the case here.
    """
    if bias is not None:
        return mx.addmm(bias, x, weight_t)
    return x @ weight_t


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    dim_head: int,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Reshape → SDPA → reshape, same as our ``_attention_core``."""
    b, t_q, _ = q.shape
    _, t_k, _ = k.shape
    q = q.reshape(b, t_q, heads, dim_head).transpose(0, 2, 1, 3)
    k = k.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)
    v = v.reshape(b, t_k, heads, dim_head).transpose(0, 2, 1, 3)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        mask = mask.astype(q.dtype)
    scale = 1.0 / math.sqrt(dim_head)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return out.transpose(0, 2, 1, 3).reshape(b, t_q, heads * dim_head)


def get_ada_values(
    table: mx.array,
    timestep: mx.array,
    start: int,
    end: int,
) -> Tuple[mx.array, ...]:
    """Slice ``[start:end]`` rows of an AdaLN table, add the timestep slice.

    table:    (num_params, D)
    timestep: (B, T, num_params, D)
    returns:  tuple of (end - start) arrays, each (B, T, D)
    """
    table_slice = table[start:end]  # (S, D)
    ada = table_slice[None, None, :, :] + timestep[:, :, start:end, :]  # (B, T, S, D)
    return tuple(ada[:, :, i, :] for i in range(end - start))


def get_av_ca_ada(
    table: mx.array,
    scale_shift_timestep: mx.array,
    gate_timestep: mx.array,
    num_ss: int = AV_CA_NUM_SCALE_SHIFT,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """AV cross-modal AdaLN: 4 scale/shift values + 1 gate."""
    ss_slice = table[:num_ss]
    ss = ss_slice[None, None, :, :] + scale_shift_timestep  # (B, 1, 4, D)
    gate_slice = table[num_ss:]
    g = gate_slice[None, None, :, :] + gate_timestep  # (B, 1, 1, D)
    return (
        ss[:, :, 0, :],
        ss[:, :, 1, :],
        ss[:, :, 2, :],
        ss[:, :, 3, :],
        g[:, :, 0, :],
    )


# ─── The inlined AV transformer block forward ────────────────────────────────


def av_block_forward(
    # State
    vx: mx.array,
    ax: mx.array,
    # Static per-step inputs
    v_context: mx.array,
    a_context: mx.array,
    v_timestep_emb: mx.array,         # (B, T, 9, D_v)
    a_timestep_emb: mx.array,         # (B, T, 9, D_a)
    v_prompt_timestep: mx.array,      # (B, T, 2, D_v)
    a_prompt_timestep: mx.array,      # (B, T, 2, D_a)
    v_pe: Tuple[mx.array, mx.array],
    a_pe: Tuple[mx.array, mx.array],
    v_cross_pe: Tuple[mx.array, mx.array],
    a_cross_pe: Tuple[mx.array, mx.array],
    v_cross_ss_ts: mx.array,          # (B, 1, 4, D_v)
    v_cross_gate_ts: mx.array,        # (B, 1, 1, D_v)
    a_cross_ss_ts: mx.array,          # (B, 1, 4, D_a)
    a_cross_gate_ts: mx.array,        # (B, 1, 1, D_a)
    v_context_mask: Optional[mx.array],
    a_context_mask: Optional[mx.array],
    # Block weights (flat dict, pretransposed for every Linear)
    w: dict,
) -> Tuple[mx.array, mx.array]:
    """Single AV block forward, inlined.

    Mirrors ``BasicAVTransformerBlock.__call__`` math exactly with V2.3
    (cross_attention_adaln + apply_gated_attention both on).  Uses
    pretransposed weights everywhere (``..._weight_t``).
    """
    B = vx.shape[0]

    # ─── Video self-attention ───
    v_shift_msa, v_scale_msa, v_gate_msa = get_ada_values(
        w["scale_shift_table"], v_timestep_emb, 0, 3,
    )
    # AdaLN
    norm_vx = rms_norm(vx) * (1 + v_scale_msa) + v_shift_msa
    # Gate logits (V2: 2*sigmoid)
    v_gate = 2.0 * mx.sigmoid(
        linear(norm_vx, w["attn1.to_gate_logits.weight"], w["attn1.to_gate_logits.bias"])
    )
    # V, Q, K (order matches mlx-video; lets MLX pipeline the gate-logits matmul)
    v_v = linear_pretransposed(norm_vx, w["attn1.to_v.weight_t"], w["attn1.to_v.bias"])
    v_q = linear_pretransposed(norm_vx, w["attn1.to_q.weight_t"], w["attn1.to_q.bias"])
    v_k = linear_pretransposed(norm_vx, w["attn1.to_k.weight_t"], w["attn1.to_k.bias"])
    v_q = mx.fast.rms_norm(v_q, w["attn1.q_norm.weight"], NORM_EPS)
    v_k = mx.fast.rms_norm(v_k, w["attn1.k_norm.weight"], NORM_EPS)
    v_q = apply_rotary_emb(v_q, v_pe, ROPE_TYPE)
    v_k = apply_rotary_emb(v_k, v_pe, ROPE_TYPE)
    attn_out = attention(v_q, v_k, v_v, VIDEO_HEADS, VIDEO_HEAD_DIM)
    # Per-head gate
    Bs, T, _ = attn_out.shape
    attn_out = attn_out.reshape(Bs, T, VIDEO_HEADS, VIDEO_HEAD_DIM)
    attn_out = attn_out * v_gate[:, :, :, None]
    attn_out = attn_out.reshape(Bs, T, VIDEO_INNER_DIM)
    attn_out = linear_pretransposed(attn_out, w["attn1.to_out.weight_t"], w["attn1.to_out.bias"])
    vx = vx + attn_out * v_gate_msa

    # ─── Video text cross-attention (V2 prompt-adaln path) ───
    v_shift_q, v_scale_q, v_gate_q = get_ada_values(
        w["scale_shift_table"], v_timestep_emb, 6, 9,
    )
    v_prompt_shift_kv, v_prompt_scale_kv = get_ada_values(
        w["prompt_scale_shift_table"], v_prompt_timestep, 0, 2,
    )
    attn_in = rms_norm(vx) * (1 + v_scale_q) + v_shift_q
    enc_ctx = v_context * (1 + v_prompt_scale_kv) + v_prompt_shift_kv
    v_gate2 = 2.0 * mx.sigmoid(
        linear(attn_in, w["attn2.to_gate_logits.weight"], w["attn2.to_gate_logits.bias"])
    )
    v_v = linear_pretransposed(enc_ctx, w["attn2.to_v.weight_t"], w["attn2.to_v.bias"])
    v_q = linear_pretransposed(attn_in, w["attn2.to_q.weight_t"], w["attn2.to_q.bias"])
    v_k = linear_pretransposed(enc_ctx, w["attn2.to_k.weight_t"], w["attn2.to_k.bias"])
    v_q = mx.fast.rms_norm(v_q, w["attn2.q_norm.weight"], NORM_EPS)
    v_k = mx.fast.rms_norm(v_k, w["attn2.k_norm.weight"], NORM_EPS)
    attn_out = attention(v_q, v_k, v_v, VIDEO_HEADS, VIDEO_HEAD_DIM, mask=v_context_mask)
    Bs, T, _ = attn_out.shape
    attn_out = attn_out.reshape(Bs, T, VIDEO_HEADS, VIDEO_HEAD_DIM)
    attn_out = attn_out * v_gate2[:, :, :, None]
    attn_out = attn_out.reshape(Bs, T, VIDEO_INNER_DIM)
    attn_out = linear_pretransposed(attn_out, w["attn2.to_out.weight_t"], w["attn2.to_out.bias"])
    vx = vx + attn_out * v_gate_q

    # ─── Audio self-attention ───
    a_shift_msa, a_scale_msa, a_gate_msa = get_ada_values(
        w["audio_scale_shift_table"], a_timestep_emb, 0, 3,
    )
    norm_ax = rms_norm(ax) * (1 + a_scale_msa) + a_shift_msa
    a_gate = 2.0 * mx.sigmoid(
        linear(norm_ax, w["audio_attn1.to_gate_logits.weight"], w["audio_attn1.to_gate_logits.bias"])
    )
    a_v = linear_pretransposed(norm_ax, w["audio_attn1.to_v.weight_t"], w["audio_attn1.to_v.bias"])
    a_q = linear_pretransposed(norm_ax, w["audio_attn1.to_q.weight_t"], w["audio_attn1.to_q.bias"])
    a_k = linear_pretransposed(norm_ax, w["audio_attn1.to_k.weight_t"], w["audio_attn1.to_k.bias"])
    a_q = mx.fast.rms_norm(a_q, w["audio_attn1.q_norm.weight"], NORM_EPS)
    a_k = mx.fast.rms_norm(a_k, w["audio_attn1.k_norm.weight"], NORM_EPS)
    a_q = apply_rotary_emb(a_q, a_pe, ROPE_TYPE)
    a_k = apply_rotary_emb(a_k, a_pe, ROPE_TYPE)
    attn_out = attention(a_q, a_k, a_v, AUDIO_HEADS, AUDIO_HEAD_DIM)
    Bs, T, _ = attn_out.shape
    attn_out = attn_out.reshape(Bs, T, AUDIO_HEADS, AUDIO_HEAD_DIM)
    attn_out = attn_out * a_gate[:, :, :, None]
    attn_out = attn_out.reshape(Bs, T, AUDIO_INNER_DIM)
    attn_out = linear_pretransposed(attn_out, w["audio_attn1.to_out.weight_t"], w["audio_attn1.to_out.bias"])
    ax = ax + attn_out * a_gate_msa

    # ─── Audio text cross-attention (V2 prompt-adaln path) ───
    a_shift_q, a_scale_q, a_gate_q = get_ada_values(
        w["audio_scale_shift_table"], a_timestep_emb, 6, 9,
    )
    a_prompt_shift_kv, a_prompt_scale_kv = get_ada_values(
        w["audio_prompt_scale_shift_table"], a_prompt_timestep, 0, 2,
    )
    attn_in = rms_norm(ax) * (1 + a_scale_q) + a_shift_q
    enc_ctx = a_context * (1 + a_prompt_scale_kv) + a_prompt_shift_kv
    a_gate2 = 2.0 * mx.sigmoid(
        linear(attn_in, w["audio_attn2.to_gate_logits.weight"], w["audio_attn2.to_gate_logits.bias"])
    )
    a_v = linear_pretransposed(enc_ctx, w["audio_attn2.to_v.weight_t"], w["audio_attn2.to_v.bias"])
    a_q = linear_pretransposed(attn_in, w["audio_attn2.to_q.weight_t"], w["audio_attn2.to_q.bias"])
    a_k = linear_pretransposed(enc_ctx, w["audio_attn2.to_k.weight_t"], w["audio_attn2.to_k.bias"])
    a_q = mx.fast.rms_norm(a_q, w["audio_attn2.q_norm.weight"], NORM_EPS)
    a_k = mx.fast.rms_norm(a_k, w["audio_attn2.k_norm.weight"], NORM_EPS)
    attn_out = attention(a_q, a_k, a_v, AUDIO_HEADS, AUDIO_HEAD_DIM, mask=a_context_mask)
    Bs, T, _ = attn_out.shape
    attn_out = attn_out.reshape(Bs, T, AUDIO_HEADS, AUDIO_HEAD_DIM)
    attn_out = attn_out * a_gate2[:, :, :, None]
    attn_out = attn_out.reshape(Bs, T, AUDIO_INNER_DIM)
    attn_out = linear_pretransposed(attn_out, w["audio_attn2.to_out.weight_t"], w["audio_attn2.to_out.bias"])
    ax = ax + attn_out * a_gate_q

    # ─── AV cross-modal (A2V + V2A) ───
    vx_norm3 = rms_norm(vx)
    ax_norm3 = rms_norm(ax)
    (
        a_scale_a2v, a_shift_a2v, a_scale_v2a, a_shift_v2a, gate_out_v2a,
    ) = get_av_ca_ada(
        w["scale_shift_table_a2v_ca_audio"], a_cross_ss_ts, a_cross_gate_ts,
    )
    (
        v_scale_a2v, v_shift_a2v, v_scale_v2a, v_shift_v2a, gate_out_a2v,
    ) = get_av_ca_ada(
        w["scale_shift_table_a2v_ca_video"], v_cross_ss_ts, v_cross_gate_ts,
    )

    # A2V: Q from video, K/V from audio
    vx_scaled = vx_norm3 * (1 + v_scale_a2v) + v_shift_a2v
    ax_scaled = ax_norm3 * (1 + a_scale_a2v) + a_shift_a2v
    a2v_gate = 2.0 * mx.sigmoid(
        linear(vx_scaled, w["audio_to_video_attn.to_gate_logits.weight"], w["audio_to_video_attn.to_gate_logits.bias"])
    )
    a2v_v = linear_pretransposed(ax_scaled, w["audio_to_video_attn.to_v.weight_t"], w["audio_to_video_attn.to_v.bias"])
    a2v_q = linear_pretransposed(vx_scaled, w["audio_to_video_attn.to_q.weight_t"], w["audio_to_video_attn.to_q.bias"])
    a2v_k = linear_pretransposed(ax_scaled, w["audio_to_video_attn.to_k.weight_t"], w["audio_to_video_attn.to_k.bias"])
    a2v_q = mx.fast.rms_norm(a2v_q, w["audio_to_video_attn.q_norm.weight"], NORM_EPS)
    a2v_k = mx.fast.rms_norm(a2v_k, w["audio_to_video_attn.k_norm.weight"], NORM_EPS)
    a2v_q = apply_rotary_emb(a2v_q, v_cross_pe, ROPE_TYPE)
    a2v_k = apply_rotary_emb(a2v_k, a_cross_pe, ROPE_TYPE)
    # Note: audio_to_video_attn uses audio_heads x audio_head_dim per the modular path
    attn_out = attention(a2v_q, a2v_k, a2v_v, AUDIO_HEADS, AUDIO_HEAD_DIM)
    Bs, T, _ = attn_out.shape
    attn_out = attn_out.reshape(Bs, T, AUDIO_HEADS, AUDIO_HEAD_DIM)
    attn_out = attn_out * a2v_gate[:, :, :, None]
    attn_out = attn_out.reshape(Bs, T, -1)
    attn_out = linear_pretransposed(attn_out, w["audio_to_video_attn.to_out.weight_t"], w["audio_to_video_attn.to_out.bias"])
    vx = vx + attn_out * gate_out_a2v

    # V2A: Q from audio, K/V from video
    ax_scaled = ax_norm3 * (1 + a_scale_v2a) + a_shift_v2a
    vx_scaled = vx_norm3 * (1 + v_scale_v2a) + v_shift_v2a
    v2a_gate = 2.0 * mx.sigmoid(
        linear(ax_scaled, w["video_to_audio_attn.to_gate_logits.weight"], w["video_to_audio_attn.to_gate_logits.bias"])
    )
    v2a_v = linear_pretransposed(vx_scaled, w["video_to_audio_attn.to_v.weight_t"], w["video_to_audio_attn.to_v.bias"])
    v2a_q = linear_pretransposed(ax_scaled, w["video_to_audio_attn.to_q.weight_t"], w["video_to_audio_attn.to_q.bias"])
    v2a_k = linear_pretransposed(vx_scaled, w["video_to_audio_attn.to_k.weight_t"], w["video_to_audio_attn.to_k.bias"])
    v2a_q = mx.fast.rms_norm(v2a_q, w["video_to_audio_attn.q_norm.weight"], NORM_EPS)
    v2a_k = mx.fast.rms_norm(v2a_k, w["video_to_audio_attn.k_norm.weight"], NORM_EPS)
    v2a_q = apply_rotary_emb(v2a_q, a_cross_pe, ROPE_TYPE)
    v2a_k = apply_rotary_emb(v2a_k, v_cross_pe, ROPE_TYPE)
    attn_out = attention(v2a_q, v2a_k, v2a_v, AUDIO_HEADS, AUDIO_HEAD_DIM)
    Bs, T, _ = attn_out.shape
    attn_out = attn_out.reshape(Bs, T, AUDIO_HEADS, AUDIO_HEAD_DIM)
    attn_out = attn_out * v2a_gate[:, :, :, None]
    attn_out = attn_out.reshape(Bs, T, -1)
    attn_out = linear_pretransposed(attn_out, w["video_to_audio_attn.to_out.weight_t"], w["video_to_audio_attn.to_out.bias"])
    ax = ax + attn_out * gate_out_v2a

    # ─── Video FFN ───
    v_shift_mlp, v_scale_mlp, v_gate_mlp = get_ada_values(
        w["scale_shift_table"], v_timestep_emb, 3, 6,
    )
    vx_scaled = rms_norm(vx) * (1 + v_scale_mlp) + v_shift_mlp
    h = linear_pretransposed(vx_scaled, w["ff.project_in.proj.weight_t"], w["ff.project_in.proj.bias"])
    h = nn.gelu_approx(h)
    ff_out = linear_pretransposed(h, w["ff.project_out.weight_t"], w["ff.project_out.bias"])
    vx = vx + ff_out * v_gate_mlp

    # ─── Audio FFN ───
    a_shift_mlp, a_scale_mlp, a_gate_mlp = get_ada_values(
        w["audio_scale_shift_table"], a_timestep_emb, 3, 6,
    )
    ax_scaled = rms_norm(ax) * (1 + a_scale_mlp) + a_shift_mlp
    h = linear_pretransposed(ax_scaled, w["audio_ff.project_in.proj.weight_t"], w["audio_ff.project_in.proj.bias"])
    h = nn.gelu_approx(h)
    ff_out = linear_pretransposed(h, w["audio_ff.project_out.weight_t"], w["audio_ff.project_out.bias"])
    ax = ax + ff_out * a_gate_mlp

    return vx, ax


# ─── Weight extraction: pull every per-block weight into a flat list of dicts ───


def extract_block_weights(model) -> List[dict]:
    """Walk the modular model and produce 48 dicts, one per block.

    Each dict holds the exact weight arrays the inlined block forward needs,
    keyed by short names matching ``av_block_forward``'s ``w[...]`` lookups.
    For pretransposed projections we use the cached ``_<name>_weight_t`` tensor
    that's already on the Attention/FeedForward instance after cache load.
    """
    blocks = []
    for block in model.transformer_blocks:
        w = {}
        # Per-block tables
        w["scale_shift_table"] = block.scale_shift_table
        w["audio_scale_shift_table"] = block.audio_scale_shift_table
        w["prompt_scale_shift_table"] = block.prompt_scale_shift_table
        w["audio_prompt_scale_shift_table"] = block.audio_prompt_scale_shift_table
        w["scale_shift_table_a2v_ca_audio"] = block.scale_shift_table_a2v_ca_audio
        w["scale_shift_table_a2v_ca_video"] = block.scale_shift_table_a2v_ca_video

        # 6 attention modules x (to_q/k/v/out pretransposed + bias + gate_logits)
        for attn_name in (
            "attn1", "attn2", "audio_to_video_attn",
            "audio_attn1", "audio_attn2", "video_to_audio_attn",
        ):
            attn = getattr(block, attn_name)
            # Q/K norm have learned weights (RMSNorm.weight) — extract them so
            # ``av_block_forward`` can pass them to mx.fast.rms_norm.  Skipping
            # this silently breaks attention (mismatched stats per head); was
            # the bug that turned the first inlined output into brown noise.
            w[f"{attn_name}.q_norm.weight"] = attn.q_norm.weight
            w[f"{attn_name}.k_norm.weight"] = attn.k_norm.weight
            for proj in ("to_q", "to_k", "to_v", "to_out"):
                weight_t = getattr(attn, f"_{proj}_weight_t")
                if weight_t is None:
                    raise RuntimeError(
                        f"Block {block.idx} {attn_name}.{proj} has no "
                        "pretransposed weight; expected cache to install one."
                    )
                w[f"{attn_name}.{proj}.weight_t"] = weight_t
                linear_mod = getattr(attn, proj)
                bias = linear_mod.get("bias")
                if bias is None:
                    raise RuntimeError(f"{attn_name}.{proj} missing bias")
                w[f"{attn_name}.{proj}.bias"] = bias
            # to_gate_logits is a regular Linear (not pretransposed by default)
            gl = attn.to_gate_logits
            w[f"{attn_name}.to_gate_logits.weight"] = gl.weight
            w[f"{attn_name}.to_gate_logits.bias"] = gl.bias

        # Feed-forwards (project_in is wrapped in GELUApprox(proj=Linear))
        for ff_name in ("ff", "audio_ff"):
            ff = getattr(block, ff_name)
            # project_in
            pi_t = ff._project_in_weight_t
            if pi_t is None:
                raise RuntimeError(f"{ff_name}.project_in.proj has no pretransposed weight")
            w[f"{ff_name}.project_in.proj.weight_t"] = pi_t
            w[f"{ff_name}.project_in.proj.bias"] = ff.project_in.proj.bias
            # project_out
            po_t = ff._project_out_weight_t
            if po_t is None:
                raise RuntimeError(f"{ff_name}.project_out has no pretransposed weight")
            w[f"{ff_name}.project_out.weight_t"] = po_t
            w[f"{ff_name}.project_out.bias"] = ff.project_out.bias

        blocks.append(w)
    return blocks


# ─── Inlined transformer preprocessing and forward (one denoise step) ────────


def transformer_step(
    *,
    video_latent: mx.array,         # (B, T_v, in_channels)
    audio_latent: mx.array,         # (B, T_a, in_channels)
    video_timestep_scalar: mx.array,  # (B,), fp32
    audio_timestep_scalar: mx.array,  # (B,), fp32
    video_context: mx.array,
    audio_context: mx.array,
    video_context_mask: Optional[mx.array],
    audio_context_mask: Optional[mx.array],
    video_pe: Tuple[mx.array, mx.array],
    audio_pe: Tuple[mx.array, mx.array],
    video_cross_pe: Tuple[mx.array, mx.array],
    audio_cross_pe: Tuple[mx.array, mx.array],
    video_cross_ss_ts: mx.array,
    video_cross_gate_ts: mx.array,
    audio_cross_ss_ts: mx.array,
    audio_cross_gate_ts: mx.array,
    weights: dict,
    block_weights: List[dict],
    compute_dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array]:
    """Run one full transformer forward and return raw velocities.

    Inputs are latents at the current sigma; outputs are raw velocities (the
    caller does the X0/Euler update inline).  This is the body of one denoise
    step; callers materialize the result.
    """
    # Patchify
    vx = linear(
        video_latent.astype(compute_dtype),
        weights["patchify_proj.weight"],
        weights["patchify_proj.bias"],
    )
    ax = linear(
        audio_latent.astype(compute_dtype),
        weights["audio_patchify_proj.weight"],
        weights["audio_patchify_proj.bias"],
    )
    B = vx.shape[0]

    # AdaLN-single timestep embedding for both modalities (V2: 9 emb each)
    v_timestep_emb, v_embedded_ts = _adaln_single_forward(
        video_timestep_scalar * TIMESTEP_SCALE,
        weights["adaln_single.emb"],
        weights["adaln_single.linear.weight"],
        weights["adaln_single.linear.bias"],
        ADALN_NUM_EMBEDDINGS,
        VIDEO_INNER_DIM,
        compute_dtype,
    )
    a_timestep_emb, a_embedded_ts = _adaln_single_forward(
        audio_timestep_scalar * TIMESTEP_SCALE,
        weights["audio_adaln_single.emb"],
        weights["audio_adaln_single.linear.weight"],
        weights["audio_adaln_single.linear.bias"],
        ADALN_NUM_EMBEDDINGS,
        AUDIO_INNER_DIM,
        compute_dtype,
    )

    # V2 prompt AdaLN.  Modular ``_prepare_timestep`` scales by
    # ``timestep_scale_multiplier`` (= TIMESTEP_SCALE) for every adaln call,
    # including prompt — so apply * TIMESTEP_SCALE here too, not just MSA.
    v_prompt_ts, _ = _adaln_single_forward(
        video_timestep_scalar * TIMESTEP_SCALE,
        weights["prompt_adaln_single.emb"],
        weights["prompt_adaln_single.linear.weight"],
        weights["prompt_adaln_single.linear.bias"],
        PROMPT_ADALN_NUM,
        VIDEO_INNER_DIM,
        compute_dtype,
    )
    a_prompt_ts, _ = _adaln_single_forward(
        audio_timestep_scalar * TIMESTEP_SCALE,
        weights["audio_prompt_adaln_single.emb"],
        weights["audio_prompt_adaln_single.linear.weight"],
        weights["audio_prompt_adaln_single.linear.bias"],
        PROMPT_ADALN_NUM,
        AUDIO_INNER_DIM,
        compute_dtype,
    )

    # Cross-attention timestep embeddings (driven by the OTHER modality's sigma)
    v_cross_ss_emb = video_cross_ss_ts.astype(compute_dtype)
    v_cross_gate_emb = video_cross_gate_ts.astype(compute_dtype)
    a_cross_ss_emb = audio_cross_ss_ts.astype(compute_dtype)
    a_cross_gate_emb = audio_cross_gate_ts.astype(compute_dtype)

    # Attention masks (dtype-cast for SDPA)
    v_ctx_mask = _prepare_mask(video_context_mask, compute_dtype)
    a_ctx_mask = _prepare_mask(audio_context_mask, compute_dtype)

    # ─── The block stack ───
    for w in block_weights:
        vx, ax = av_block_forward(
            vx, ax,
            video_context, audio_context,
            v_timestep_emb, a_timestep_emb,
            v_prompt_ts, a_prompt_ts,
            video_pe, audio_pe,
            video_cross_pe, audio_cross_pe,
            v_cross_ss_emb, v_cross_gate_emb,
            a_cross_ss_emb, a_cross_gate_emb,
            v_ctx_mask, a_ctx_mask,
            w,
        )

    # ─── Output projection (per-modality scale_shift_table + LayerNorm + proj_out) ───
    v_out = _output_project(
        vx, v_embedded_ts,
        weights["scale_shift_table"],
        weights["norm_out.weight"] if "norm_out.weight" in weights else None,
        weights["norm_out.bias"] if "norm_out.bias" in weights else None,
        weights["proj_out.weight"],
        weights["proj_out.bias"],
        VIDEO_INNER_DIM,
    )
    a_out = _output_project(
        ax, a_embedded_ts,
        weights["audio_scale_shift_table"],
        weights["audio_norm_out.weight"] if "audio_norm_out.weight" in weights else None,
        weights["audio_norm_out.bias"] if "audio_norm_out.bias" in weights else None,
        weights["audio_proj_out.weight"],
        weights["audio_proj_out.bias"],
        AUDIO_INNER_DIM,
    )
    return v_out, a_out


def _prepare_mask(mask: Optional[mx.array], dtype: mx.Dtype) -> Optional[mx.array]:
    if mask is None:
        return None
    return mask.astype(dtype)


def _adaln_single_forward(
    timestep_scaled: mx.array,
    emb_module,            # the SiLU + Embedding wrapper from PixArtAlphaCombinedTimestepSizeEmbeddings
    linear_weight: mx.array,
    linear_bias: mx.array,
    num_emb: int,
    inner_dim: int,
    dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array]:
    """Inline of ``AdaLayerNormSingle.__call__``.

    ``emb_module`` is the timestep embedding submodule (sinusoidal + MLP);
    inlining that is out of scope, but it's <<1ms anyway, so reuse.
    """
    embedded = emb_module(timestep_scaled.flatten())
    pre_lin = nn.silu(embedded)
    out = linear(pre_lin, linear_weight, linear_bias)
    # Reshape to (B, T, num_emb, dim) for slicing by AdaLN section
    out = out.reshape(out.shape[0], -1, num_emb, inner_dim)
    embedded_ts = embedded.reshape(embedded.shape[0], -1, inner_dim)
    return out.astype(dtype), embedded_ts.astype(dtype)


def _output_project(
    x: mx.array,
    embedded_ts: mx.array,
    scale_shift_table: mx.array,
    norm_weight: Optional[mx.array],
    norm_bias: Optional[mx.array],
    proj_weight: mx.array,
    proj_bias: mx.array,
    inner_dim: int,
) -> mx.array:
    """Match LTXModel._process_video_output: norm + (1+scale)*x + shift + proj."""
    scale_shift = (
        scale_shift_table[None, None, :, :] + embedded_ts[:, :, None, :]
    ).astype(x.dtype)
    shift = scale_shift[:, :, 0, :]
    scale = scale_shift[:, :, 1, :]
    # nn.LayerNorm with affine=False ⇒ no learnable params; use mx.fast.layer_norm
    x = mx.fast.layer_norm(x, None, None, NORM_EPS)
    x = x * (1 + scale) + shift
    return linear(x, proj_weight, proj_bias)


# ─── Top-level weight extraction (model-level params, not per-block) ──────────


def extract_top_level_weights(model) -> dict:
    w = {}
    w["patchify_proj.weight"] = model.patchify_proj.weight
    w["patchify_proj.bias"] = model.patchify_proj.bias
    w["audio_patchify_proj.weight"] = model.audio_patchify_proj.weight
    w["audio_patchify_proj.bias"] = model.audio_patchify_proj.bias
    w["proj_out.weight"] = model.proj_out.weight
    w["proj_out.bias"] = model.proj_out.bias
    w["audio_proj_out.weight"] = model.audio_proj_out.weight
    w["audio_proj_out.bias"] = model.audio_proj_out.bias
    w["scale_shift_table"] = model.scale_shift_table
    w["audio_scale_shift_table"] = model.audio_scale_shift_table

    # AdaLN-single emb + linear (we reuse the emb module, extract linear weight/bias)
    w["adaln_single.emb"] = model.adaln_single.emb
    w["adaln_single.linear.weight"] = model.adaln_single.linear.weight
    w["adaln_single.linear.bias"] = model.adaln_single.linear.bias
    w["audio_adaln_single.emb"] = model.audio_adaln_single.emb
    w["audio_adaln_single.linear.weight"] = model.audio_adaln_single.linear.weight
    w["audio_adaln_single.linear.bias"] = model.audio_adaln_single.linear.bias

    # Prompt AdaLN
    w["prompt_adaln_single.emb"] = model.prompt_adaln_single.emb
    w["prompt_adaln_single.linear.weight"] = model.prompt_adaln_single.linear.weight
    w["prompt_adaln_single.linear.bias"] = model.prompt_adaln_single.linear.bias
    w["audio_prompt_adaln_single.emb"] = model.audio_prompt_adaln_single.emb
    w["audio_prompt_adaln_single.linear.weight"] = model.audio_prompt_adaln_single.linear.weight
    w["audio_prompt_adaln_single.linear.bias"] = model.audio_prompt_adaln_single.linear.bias

    # AV CA AdaLN modules (4 of them) — reuse the call inline
    w["av_ca_video_scale_shift_adaln_single"] = model.av_ca_video_scale_shift_adaln_single
    w["av_ca_audio_scale_shift_adaln_single"] = model.av_ca_audio_scale_shift_adaln_single
    w["av_ca_a2v_gate_adaln_single"] = model.av_ca_a2v_gate_adaln_single
    w["av_ca_v2a_gate_adaln_single"] = model.av_ca_v2a_gate_adaln_single

    # norm_out is affine=False so no params
    return w


# ─── Cross-attention timestep prep (mirrors MultiModalTransformerArgsPreprocessor) ───


def prep_cross_attention_timestep(
    cross_sigma: mx.array,
    ss_adaln_module,
    gate_adaln_module,
    inner_dim: int,
    dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array]:
    """Inline of ``MultiModalTransformerArgsPreprocessor._prepare_cross_attention_timestep``."""
    scaled = cross_sigma * TIMESTEP_SCALE
    av_ca_factor = AV_CA_TIMESTEP_SCALE / TIMESTEP_SCALE
    ss_emb, _ = ss_adaln_module(scaled.flatten())
    ss_emb = ss_emb.reshape(scaled.shape[0], -1, AV_CA_NUM_SCALE_SHIFT, inner_dim).astype(dtype)
    gate_emb, _ = gate_adaln_module((scaled * av_ca_factor).flatten())
    gate_emb = gate_emb.reshape(scaled.shape[0], -1, AV_CA_NUM_GATE, inner_dim).astype(dtype)
    return ss_emb, gate_emb


# ─── Per-stage RoPE precompute ───


def precompute_stage_rope(
    positions: mx.array,
    inner_dim: int,
    max_pos: List[int],
    num_attention_heads: int,
    use_middle_indices_grid: bool,
) -> Tuple[mx.array, mx.array]:
    return precompute_freqs_cis(
        indices_grid=positions,
        dim=inner_dim,
        out_dtype=mx.float32,
        theta=POSITIONAL_EMBEDDING_THETA,
        max_pos=max_pos,
        use_middle_indices_grid=use_middle_indices_grid,
        num_attention_heads=num_attention_heads,
        rope_type=ROPE_TYPE,
    )


def precompute_cross_rope(
    positions: mx.array,           # (B, n_dims, T) from the per-modality state
    audio_cross_attention_dim: int,
    num_attention_heads: int,
) -> Tuple[mx.array, mx.array]:
    # Only temporal axis; matches MultiModalTransformerArgsPreprocessor.prepare
    temporal = positions[:, 0:1, :]
    return precompute_freqs_cis(
        indices_grid=temporal,
        dim=audio_cross_attention_dim,
        out_dtype=mx.float32,
        theta=POSITIONAL_EMBEDDING_THETA,
        max_pos=[CROSS_PE_MAX_POS],
        use_middle_indices_grid=True,
        num_attention_heads=num_attention_heads,
        rope_type=ROPE_TYPE,
    )


# --- InlinedAVModel: drop-in for X0Model(LTXAVModel) used by the pipeline ----


class InlinedAVModel:
    """Pipeline-compatible wrapper that calls ``transformer_step`` per step.

    ``stage2_harness.py`` swaps ``av_pipeline.transformer = InlinedAVModel(model)``
    when ``LTX_MONO_INLINED=1`` is set.  Pipeline still does spatial upscale,
    sigma loop, modality construction, and Euler step — only the transformer
    forward changes.

    Bit-identical math intent vs the modular path: same weights, same
    operations, same dtype discipline.  Per-stage RoPE is cached
    (positions don't change between steps of a stage).
    """

    def __init__(self, base_model):
        # Pipeline checks ``hasattr(self.transformer, 'velocity_model')`` to
        # detect the X0 wrapper; point it at ourselves so either path works.
        self.velocity_model = self
        self.compute_dtype = base_model.compute_dtype

        # Flat weights extracted from the loaded modular model.
        self.top_weights = extract_top_level_weights(base_model)
        self.block_weights = extract_block_weights(base_model)

        # RoPE caches keyed by positions shape (stable within a stage).
        self._video_rope_cache: dict = {}
        self._audio_rope_cache: dict = {}
        self._video_cross_rope_cache: dict = {}
        self._audio_cross_rope_cache: dict = {}

        # AV CA AdaLN modules — small, not the experiment; reuse.
        self._av_ca_video_ss = base_model.av_ca_video_scale_shift_adaln_single
        self._av_ca_audio_ss = base_model.av_ca_audio_scale_shift_adaln_single
        self._av_ca_a2v_gate = base_model.av_ca_a2v_gate_adaln_single
        self._av_ca_v2a_gate = base_model.av_ca_v2a_gate_adaln_single

        # Shape parameters pulled from the base model so we don't hard-code.
        # Audio dims live as class constants on LTXAVModel (AUDIO_*); video
        # dims live as instance attrs set from the constructor.
        self._audio_cross_attention_dim = base_model.audio_inner_dim
        self._video_num_heads = base_model.num_attention_heads
        self._audio_num_heads = base_model.AUDIO_ATTENTION_HEADS
        self._video_inner_dim = base_model.inner_dim
        self._audio_inner_dim = base_model.audio_inner_dim
        self._video_max_pos = base_model.positional_embedding_max_pos
        # Audio simple preprocessor uses [AUDIO_CROSS_PE_MAX_POS] as max_pos,
        # not a separate audio_positional_embedding_max_pos attribute.
        self._audio_max_pos = [base_model.AUDIO_CROSS_PE_MAX_POS]

    def _stage_rope(self, cache, positions, inner_dim, max_pos, num_heads):
        key = tuple(positions.shape)
        if key not in cache:
            pe = precompute_stage_rope(
                positions, inner_dim, max_pos, num_heads,
                use_middle_indices_grid=True,
            )
            mx.eval(*pe)
            cache[key] = pe
        return cache[key]

    def _cross_rope(self, cache, positions, num_heads):
        key = tuple(positions.shape)
        if key not in cache:
            pe = precompute_cross_rope(
                positions, self._audio_cross_attention_dim, num_heads,
            )
            mx.eval(*pe)
            cache[key] = pe
        return cache[key]

    def __call__(self, video, audio=None, perturbations=None):
        if audio is None:
            raise RuntimeError(
                "InlinedAVModel only supports the audio-on AV path; "
                "fall back to the modular path for video-only runs.",
            )

        v_pe = self._stage_rope(
            self._video_rope_cache, video.positions,
            self._video_inner_dim, self._video_max_pos, self._video_num_heads,
        )
        a_pe = self._stage_rope(
            self._audio_rope_cache, audio.positions,
            self._audio_inner_dim, self._audio_max_pos, self._audio_num_heads,
        )
        v_cross_pe = self._cross_rope(
            self._video_cross_rope_cache, video.positions, self._video_num_heads,
        )
        a_cross_pe = self._cross_rope(
            self._audio_cross_rope_cache, audio.positions, self._audio_num_heads,
        )

        # Cross-modal AdaLN: video conditioned on AUDIO sigma; vice versa.
        v_cross_ss, v_cross_gate = prep_cross_attention_timestep(
            audio.sigma, self._av_ca_video_ss, self._av_ca_a2v_gate,
            self._video_inner_dim, self.compute_dtype,
        )
        a_cross_ss, a_cross_gate = prep_cross_attention_timestep(
            video.sigma, self._av_ca_audio_ss, self._av_ca_v2a_gate,
            self._audio_inner_dim, self.compute_dtype,
        )

        # Inlined forward returns raw velocities.
        v_vel, a_vel = transformer_step(
            video_latent=video.latent,
            audio_latent=audio.latent,
            video_timestep_scalar=video.sigma,
            audio_timestep_scalar=audio.sigma,
            video_context=video.context,
            audio_context=audio.context,
            video_context_mask=video.context_mask,
            audio_context_mask=audio.context_mask,
            video_pe=v_pe,
            audio_pe=a_pe,
            video_cross_pe=v_cross_pe,
            audio_cross_pe=a_cross_pe,
            video_cross_ss_ts=v_cross_ss,
            video_cross_gate_ts=v_cross_gate,
            audio_cross_ss_ts=a_cross_ss,
            audio_cross_gate_ts=a_cross_gate,
            weights=self.top_weights,
            block_weights=self.block_weights,
            compute_dtype=self.compute_dtype,
        )

        # X0 conversion (matches X0Model wrapper math).
        v_ts = (video.timesteps[:, None, None] if video.timesteps.ndim == 1
                else video.timesteps[:, :, None])
        a_ts = (audio.timesteps[:, None, None] if audio.timesteps.ndim == 1
                else audio.timesteps[:, :, None])
        return video.latent - v_ts * v_vel, audio.latent - a_ts * a_vel
