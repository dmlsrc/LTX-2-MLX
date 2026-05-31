"""Fused operations using custom Metal kernels for LTX-2 MLX.

These kernels combine multiple operations into single GPU passes,
reducing memory bandwidth and kernel launch overhead.
"""

import atexit
import os

import mlx.core as mx
import numpy as np


# ── Dispatch probe ─────────────────────────────────────────────────────────
#
# When LTX_FUSED_PROBE=1 is set, every adaln_norm_fused / gated_add_fused
# call increments a counter for either the kernel path or the MLX fallback
# path.  At process exit the counters are printed so you can confirm the
# fused kernels actually fired (and how many times) during a generate.py
# run vs silently routing through the fallback.
#
# Zero overhead when LTX_FUSED_PROBE is unset (single env-var check at
# module load → no-op increments).

_PROBE_ENABLED = bool(os.environ.get("LTX_FUSED_PROBE"))
_PROBE_COUNTS: dict[str, int] = {
    "adaln_kernel": 0,
    "adaln_fallback": 0,
    "gated_add_kernel": 0,
    "gated_add_fallback": 0,
}


def _probe(name: str) -> None:
    if _PROBE_ENABLED:
        _PROBE_COUNTS[name] += 1


def get_dispatch_counts() -> dict[str, int]:
    """Return a copy of the dispatch counters (only nonzero when
    ``LTX_FUSED_PROBE=1`` was set when this module was first imported)."""
    return dict(_PROBE_COUNTS)


def _print_dispatch_summary() -> None:
    """atexit hook: print the dispatch counter summary if any kernel was
    invoked.  No-op when the probe wasn't enabled or no calls happened."""
    if not _PROBE_ENABLED:
        return
    total = sum(_PROBE_COUNTS.values())
    if total == 0:
        return
    print("\n[LTX_FUSED_PROBE] fused-op dispatch summary:")
    for name in (
        "adaln_kernel",
        "adaln_fallback",
        "gated_add_kernel",
        "gated_add_fallback",
    ):
        n = _PROBE_COUNTS[name]
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {name:<22} {n:>8}  ({pct:5.1f}%)")
    # Compute per-op kernel-rate.
    adaln_total = _PROBE_COUNTS["adaln_kernel"] + _PROBE_COUNTS["adaln_fallback"]
    gated_total = (
        _PROBE_COUNTS["gated_add_kernel"] + _PROBE_COUNTS["gated_add_fallback"]
    )
    if adaln_total:
        rate = 100.0 * _PROBE_COUNTS["adaln_kernel"] / adaln_total
        print(f"  adaln kernel-hit rate:   {rate:5.1f}% ({adaln_total} total)")
    if gated_total:
        rate = 100.0 * _PROBE_COUNTS["gated_add_kernel"] / gated_total
        print(f"  gated-add kernel-hit rate: {rate:5.1f}% ({gated_total} total)")


if _PROBE_ENABLED:
    atexit.register(_print_dispatch_summary)

# Fused SiLU-Multiply kernel: silu(a) * b
# Used in SwiGLU and gated MLPs.
#
# Implementation note: explicit float() casts at load + T() at store
# skip a bf16->fp32->bf16 cast roundtrip per element.  Metal's exp()
# on bfloat operands is not a native bf16 op -- it internally casts
# to fp32, runs the fp32 exp, casts back.  Doing the math in fp32
# explicitly lets the compiler keep the value in an fp32 register
# through the sigmoid, skipping that roundtrip.  Measured 1.19x
# speedup vs the prior bf16-math kernel at T=14640, C=16384 (Gemma 3
# text encoder hot path); robust across a cache-protocol microbench
# (hot_fixed, rotate, rotate_shuffle, thrash all show E/C ≈ 0.84 with
# stdev < 0.05 ms).
_silu_mul_kernel = mx.fast.metal_kernel(
    name="silu_mul",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
        uint idx = thread_position_in_grid.x;
        if (idx < a_shape[0] * a_shape[1] * a_shape[2]) {
            float val_a = float(a[idx]);
            float val_b = float(b[idx]);
            // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            float silu_a = val_a / (1.0f + exp(-val_a));
            out[idx] = T(silu_a * val_b);
        }
    """,
)

# Fused GELU-Multiply kernel: gelu_approx(a) * b
# Used in some gated architectures
_gelu_mul_kernel = mx.fast.metal_kernel(
    name="gelu_mul",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
        uint idx = thread_position_in_grid.x;
        if (idx < a_shape[0] * a_shape[1] * a_shape[2]) {
            T val_a = a[idx];
            T val_b = b[idx];
            // GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const T sqrt_2_over_pi = T(0.7978845608028654);
            const T coeff = T(0.044715);
            T inner = sqrt_2_over_pi * (val_a + coeff * val_a * val_a * val_a);
            T gelu_a = T(0.5) * val_a * (T(1.0) + tanh(inner));
            out[idx] = gelu_a * val_b;
        }
    """,
)


def silu_mul(a: mx.array, b: mx.array) -> mx.array:
    """
    Fused SiLU activation with element-wise multiply: silu(a) * b

    This is more efficient than separate silu() and multiply() calls
    as it avoids intermediate memory allocation and extra kernel launch.

    Used in SwiGLU: down(silu(gate(x)) * up(x))

    Args:
        a: Input to SiLU activation
        b: Input to multiply with SiLU output

    Returns:
        silu(a) * b
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Ensure contiguous for kernel
    a = mx.contiguous(a)
    b = mx.contiguous(b)

    # Flatten for simple 1D indexing, will reshape back
    original_shape = a.shape
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    # Pad shape to 3D for kernel (kernel expects 3D shape array)
    n = a_flat.size
    kernel_shape = (n, 1, 1)
    a_3d = a_flat.reshape(kernel_shape)
    b_3d = b_flat.reshape(kernel_shape)

    outputs = _silu_mul_kernel(
        inputs=[a_3d, b_3d],
        template=[("T", a.dtype)],
        output_shapes=[kernel_shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
    )

    return outputs[0].reshape(original_shape)


def gelu_mul(a: mx.array, b: mx.array) -> mx.array:
    """
    Fused GELU activation (tanh approx) with element-wise multiply: gelu(a) * b

    Args:
        a: Input to GELU activation
        b: Input to multiply with GELU output

    Returns:
        gelu_approx(a) * b
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Ensure contiguous for kernel
    a = mx.contiguous(a)
    b = mx.contiguous(b)

    # Flatten for simple 1D indexing
    original_shape = a.shape
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    n = a_flat.size
    kernel_shape = (n, 1, 1)
    a_3d = a_flat.reshape(kernel_shape)
    b_3d = b_flat.reshape(kernel_shape)

    outputs = _gelu_mul_kernel(
        inputs=[a_3d, b_3d],
        template=[("T", a.dtype)],
        output_shapes=[kernel_shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
    )

    return outputs[0].reshape(original_shape)


# Fused Interleaved RoPE kernel
# Combines reshape, rotation, and multiply in one pass
_interleaved_rope_kernel = mx.fast.metal_kernel(
    name="interleaved_rope",
    input_names=["x", "cos_freq", "sin_freq"],
    output_names=["out"],
    source="""
        // x shape: (batch, seq, dim) where dim is even
        // cos/sin shape: (batch, seq, dim) - pre-broadcasted
        // Each pair (x[2i], x[2i+1]) is rotated together

        uint idx = thread_position_in_grid.x;
        uint total = x_shape[0] * x_shape[1] * x_shape[2];

        if (idx < total) {
            // Get position in output
            uint dim = x_shape[2];
            uint seq_dim = x_shape[1] * dim;

            uint batch_idx = idx / seq_dim;
            uint remainder = idx % seq_dim;
            uint seq_idx = remainder / dim;
            uint dim_idx = remainder % dim;

            // Check if this is even or odd index in the pair
            uint pair_idx = dim_idx / 2;
            bool is_even = (dim_idx % 2) == 0;

            // Calculate indices for the pair
            uint even_idx = batch_idx * seq_dim + seq_idx * dim + pair_idx * 2;
            uint odd_idx = even_idx + 1;

            float x_even = float(x[even_idx]);
            float x_odd = float(x[odd_idx]);
            float cos_val = float(cos_freq[idx]);
            float sin_val = float(sin_freq[idx]);

            // Interleaved rotation: out_even = x_even * cos - x_odd * sin
            //                       out_odd = x_odd * cos + x_even * sin
            if (is_even) {
                out[idx] = T(x_even * cos_val - x_odd * sin_val);
            } else {
                out[idx] = T(x_odd * cos_val + x_even * sin_val);
            }
        }
    """,
)


def interleaved_rope(
    x: mx.array, cos_freqs: mx.array, sin_freqs: mx.array
) -> mx.array:
    """
    Apply interleaved rotary position embeddings using fused kernel.

    This fuses the reshape, rotation computation, and application into one kernel,
    avoiding intermediate memory allocations.

    Args:
        x: Input tensor of shape (..., dim) where dim is even
        cos_freqs: Cosine frequencies, broadcastable to x shape
        sin_freqs: Sine frequencies, broadcastable to x shape

    Returns:
        Tensor with rotary embeddings applied
    """
    original_shape = x.shape
    original_ndim = x.ndim

    # Flatten to 3D: (batch, seq, dim)
    if x.ndim == 2:
        x = x[None, :, :]
        cos_freqs = cos_freqs[None, :, :] if cos_freqs.ndim == 2 else cos_freqs
        sin_freqs = sin_freqs[None, :, :] if sin_freqs.ndim == 2 else sin_freqs
    elif x.ndim > 3:
        # Flatten batch dimensions
        batch_dims = x.shape[:-2]
        batch_size = int(np.prod(batch_dims))
        x = x.reshape(batch_size, x.shape[-2], x.shape[-1])
        cos_freqs = mx.broadcast_to(cos_freqs, x.shape)
        sin_freqs = mx.broadcast_to(sin_freqs, x.shape)

    # Ensure shapes match
    cos_freqs = mx.broadcast_to(cos_freqs, x.shape)
    sin_freqs = mx.broadcast_to(sin_freqs, x.shape)

    x = mx.contiguous(x)
    cos_freqs = mx.contiguous(cos_freqs)
    sin_freqs = mx.contiguous(sin_freqs)

    n = x.size
    outputs = _interleaved_rope_kernel(
        inputs=[x, cos_freqs, sin_freqs],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
    )

    result = outputs[0]

    # Restore original shape
    if original_ndim == 2:
        result = result[0]
    elif original_ndim > 3:
        result = result.reshape(original_shape)

    return result


# ── Fused AdaLN: rms_norm(x) * (1 + scale) + shift ─────────────────────────
#
# Replaces the ``_adaln_inline`` MLX path in transformer.py.  Production
# T2V shape is x bf16 (B, T, C), scale/shift fp32 (B, 1, C) broadcast
# across T (uniform_mask=True path through modality_from_state).
#
# Registered for both supported transformer dims:
#   - C=4096 (video, BLOCK=512, VPT=2 → 8 elem/thread, 16 SG/tg)
#   - C=2048 (audio, BLOCK=256, VPT=2 → 8 elem/thread, 8 SG/tg)
# Same algorithmic structure for both; only the constants differ.
# Other shapes fall back to the stock MLX expression (I2V per-token
# scale/shift, non-bf16 x, unsupported C, etc.).
#
# Video variant validated on M1 Max via a separate cache-protocol
# microbench: ~2.0-2.1x speedup over mx.compile under all four cache
# protocols at T=14640, C=4096; 60% of peak DRAM bandwidth (~240 GB/s).
# Audio variant added 2026-05-27 by analogy; bench has not been redone
# at C=2048 but the layout (8 elem/thread, single-barrier reduction)
# is the same shape-portable pattern that survived the C=3584 → C=4096
# shape correction.  Verified by parity tests; perf will surface in the
# end-to-end A/B.

_ADALN_FUSED_EPS = 1e-6  # hardcoded; matches transformer.py norm_eps default

# Map C → (BLOCK, VPT).  BLOCK * VPT * 4 == C; VPT=2 keeps 8 elements/thread
# across both variants (the layout that won the C=4096 bench).
_ADALN_FUSED_CONFIGS: dict[int, tuple[int, int]] = {
    4096: (512, 2),   # video transformer
    2048: (256, 2),   # audio transformer
}


def _build_adaln_norm_kernel(c: int, block: int, vpt: int) -> mx.fast.metal_kernel:
    """Build the AdaLN fused kernel for a specific C / BLOCK / VPT layout."""
    assert block * vpt * 4 == c, (
        f"BLOCK ({block}) * VPT ({vpt}) * 4 must equal C ({c}); "
        "kernel tiling needs to cover the row exactly."
    )
    return mx.fast.metal_kernel(
        name=f"adaln_norm_fused_c{c}",
        input_names=["x", "scale", "shift"],
        output_names=["out"],
        source=f"""
            // One threadgroup per row of x.  BLOCK={block} threads,
            // VPT={vpt} bfloat4 vectors of x per thread (= {vpt * 4} bf16
            // elements) and matching float4 vectors of scale/shift.
            // scale/shift are per-channel fp32 (broadcast across all T rows
            // from the (B, 1, C) AdaLN output); x and out are bf16 (T, C={c}).

            const uint row    = threadgroup_position_in_grid.x;
            const uint tid    = thread_index_in_threadgroup;
            const int  C_     = x_shape[1];
            const float EPS_F = {_ADALN_FUSED_EPS}f;
            const float INV_C = 1.0f / float(C_);

            device const bfloat4* x_v     = (device const bfloat4*)(x + row * C_);
            device const float4*  scale_v = (device const float4*)(scale);
            device const float4*  shift_v = (device const float4*)(shift);
            device       bfloat4* out_v   = (device       bfloat4*)(out + row * C_);

            // Phase 1: register-cache x, accumulate sum-of-squares in fp32.
            float xs[{vpt * 4}];
            float local_sum = 0.0f;
            #pragma unroll
            for (int v = 0; v < {vpt}; ++v) {{
                bfloat4 chunk = x_v[tid * {vpt} + v];
                float a = float(chunk.x);
                float b = float(chunk.y);
                float c = float(chunk.z);
                float d = float(chunk.w);
                xs[v*4 + 0] = a;
                xs[v*4 + 1] = b;
                xs[v*4 + 2] = c;
                xs[v*4 + 3] = d;
                local_sum += a*a + b*b + c*c + d*d;
            }}

            // Simdgroup reduce → per-SG partials in TGM, single barrier.
            threadgroup float tg_partials[{block // 32}];
            float sg_sum = simd_sum(local_sum);
            if ((tid & 31u) == 0) {{
                tg_partials[tid / 32] = sg_sum;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Every thread reads all partials, reduces locally.  No second barrier.
            float total = 0.0f;
            #pragma unroll
            for (int s = 0; s < ({block // 32}); ++s) {{
                total += tg_partials[s];
            }}
            const float inv = rsqrt(total * INV_C + EPS_F);

            // Phase 2: modulate from cached registers, vectorized stores.
            // fp32 scale/shift reads land directly without bfloat→float casts.
            #pragma unroll
            for (int v = 0; v < {vpt}; ++v) {{
                float4 sc = scale_v[tid * {vpt} + v];
                float4 sh = shift_v[tid * {vpt} + v];
                bfloat4 result;
                result.x = bfloat(xs[v*4 + 0] * inv * (1.0f + sc.x) + sh.x);
                result.y = bfloat(xs[v*4 + 1] * inv * (1.0f + sc.y) + sh.y);
                result.z = bfloat(xs[v*4 + 2] * inv * (1.0f + sc.z) + sh.z);
                result.w = bfloat(xs[v*4 + 3] * inv * (1.0f + sc.w) + sh.w);
                out_v[tid * {vpt} + v] = result;
            }}
        """,
        ensure_row_contiguous=True,
    )


# Build kernels for every supported C at import time.  Each kernel is a
# closed-over (BLOCK, VPT) instance; the dispatcher in ``adaln_norm_fused``
# picks the right one based on ``x.shape[-1]``.
_adaln_norm_fused_kernels: dict[int, mx.fast.metal_kernel] = {
    c: _build_adaln_norm_kernel(c, block, vpt)
    for c, (block, vpt) in _ADALN_FUSED_CONFIGS.items()
}


def _adaln_norm_mlx(
    x: mx.array, scale: mx.array, shift: mx.array, eps: float
) -> mx.array:
    """Stock MLX AdaLN — matches ``transformer.py::_adaln_inline``.

    Used as fallback when the fused kernel's shape/dtype gate doesn't
    match (I2V per-token scale/shift, unsupported C, non-bf16 x, etc.).
    """
    normed = mx.fast.rms_norm(x, None, eps)
    return (normed * (1 + scale) + shift).astype(x.dtype)


def _adaln_t2v_broadcast_compatible(
    x: mx.array, scale: mx.array, shift: mx.array
) -> bool:
    """Check whether (x, scale, shift) match the T2V broadcast pattern
    the fused kernels expect, for any of the supported ``C`` values.

    Returns True iff:
      - x is 3D bfloat16 with x.shape[-1] in ``_ADALN_FUSED_CONFIGS``
      - scale, shift are fp32 with last dim == x.shape[-1]
      - scale, shift have all-singleton leading dims (i.e. (1, 1, C) or
        (C,) up to broadcast) so the kernel's per-channel read is valid

    Any per-token scale/shift (B, T, C) returns False → MLX fallback.
    """
    if x.dtype != mx.bfloat16 or x.ndim != 3:
        return False
    c = x.shape[-1]
    if c not in _adaln_norm_fused_kernels:
        return False
    if scale.dtype != mx.float32 or shift.dtype != mx.float32:
        return False
    if scale.shape[-1] != c or shift.shape[-1] != c:
        return False
    # All leading dims must be 1 — covers (C,), (1, C), (1, 1, C).
    for d in scale.shape[:-1]:
        if d != 1:
            return False
    for d in shift.shape[:-1]:
        if d != 1:
            return False
    return True


def adaln_norm_fused(
    x: mx.array, scale: mx.array, shift: mx.array, eps: float
) -> mx.array:
    """Fused RMSNorm + AdaLN modulation: ``rms_norm(x) * (1+scale) + shift``.

    Supports both video (C=4096) and audio (C=2048) production T2V shapes:
    x bf16 (B, T, C) with scale/shift fp32 with last dim C and all-singleton
    leading dims (the (B, 1, C) broadcast pattern produced by
    ``modality_from_state`` when ``uniform_mask=True``).

    Falls back to the stock MLX expression (matching ``_adaln_inline``)
    for any other shape/dtype combination, so the function is a drop-in
    replacement that auto-selects the kernel path when applicable.

    Args:
        x: Input activations, shape (B, T, C).
        scale: AdaLN scale, broadcastable to x.
        shift: AdaLN shift, broadcastable to x.
        eps: RMSNorm epsilon.

    Returns:
        ``(rms_norm(x) * (1 + scale) + shift).astype(x.dtype)``.
    """
    if eps != _ADALN_FUSED_EPS:
        _probe("adaln_fallback")
        return _adaln_norm_mlx(x, scale, shift, eps)
    if not _adaln_t2v_broadcast_compatible(x, scale, shift):
        _probe("adaln_fallback")
        return _adaln_norm_mlx(x, scale, shift, eps)

    _probe("adaln_kernel")
    B, T, C = x.shape
    block, _vpt = _ADALN_FUSED_CONFIGS[C]
    kernel = _adaln_norm_fused_kernels[C]

    # Kernel expects (rows, C) layout; flatten B*T together (works for B=1).
    x_2d = mx.contiguous(x.reshape(B * T, C))
    scale_1d = mx.contiguous(scale.reshape(-1))
    shift_1d = mx.contiguous(shift.reshape(-1))

    rows = B * T
    grid = (rows * block, 1, 1)
    threadgroup = (block, 1, 1)
    outputs = kernel(
        inputs=[x_2d, scale_1d, shift_1d],
        output_shapes=[(rows, C)],
        output_dtypes=[mx.bfloat16],
        grid=grid,
        threadgroup=threadgroup,
    )
    return outputs[0].reshape(B, T, C)


# ── Fused gated add: residual + branch * gate ──────────────────────────────
#
# Replaces the ``_residual_gate_inline`` MLX path in transformer.py.
# Production T2V shape: residual bf16 (B, T, C), branch bf16 (B, T, C),
# gate fp32 (B, 1, C) broadcast across T.  Kernel computes
# ``out = branch * gate + residual``.
#
# Registered for both supported transformer dims (same configs as AdaLN):
#   - C=4096 (video, BLOCK=512, VPT=2 → 8 elem/thread, 16 SG/tg)
#   - C=2048 (audio, BLOCK=256, VPT=2 → 8 elem/thread, 8 SG/tg)
# Other shapes fall back to the stock MLX expression.
#
# Video variant validated on M1 Max: 1.26-1.30x speedup over mx.compile
# under all four cache protocols at T=14640, C=4096; 72% of peak DRAM
# bandwidth (~288 GB/s).  Audio variant added 2026-05-27 by analogy.

# Same (BLOCK, VPT) configs as AdaLN — these are bandwidth-bound kernels
# with the same per-row tiling story.
_GATED_ADD_FUSED_CONFIGS: dict[int, tuple[int, int]] = {
    4096: (512, 2),   # video transformer
    2048: (256, 2),   # audio transformer
}


def _build_gated_add_kernel(c: int, block: int, vpt: int) -> mx.fast.metal_kernel:
    """Build the gated_add fused kernel for a specific C / BLOCK / VPT layout."""
    assert block * vpt * 4 == c, (
        f"BLOCK ({block}) * VPT ({vpt}) * 4 must equal C ({c}); "
        "kernel tiling needs to cover the row exactly."
    )
    return mx.fast.metal_kernel(
        name=f"gated_add_fused_c{c}",
        input_names=["branch", "gate", "residual"],
        output_names=["out"],
        source=f"""
            // One threadgroup per row.  BLOCK={block} threads,
            // VPT={vpt} vectors/thread per dtype.  No reductions —
            // pure elementwise FMA over C={c} bf16 elements per row.

            const uint row    = threadgroup_position_in_grid.x;
            const uint tid    = thread_index_in_threadgroup;
            const int  C_     = branch_shape[1];

            device const bfloat4* branch_v   = (device const bfloat4*)(branch + row * C_);
            device const float4*  gate_v     = (device const float4*)(gate);
            device const bfloat4* residual_v = (device const bfloat4*)(residual + row * C_);
            device       bfloat4* out_v      = (device       bfloat4*)(out + row * C_);

            #pragma unroll
            for (int v = 0; v < {vpt}; ++v) {{
                bfloat4 br  = branch_v[tid * {vpt} + v];
                float4  g   = gate_v[tid * {vpt} + v];
                bfloat4 res = residual_v[tid * {vpt} + v];
                bfloat4 result;
                // fp32 accumulator inside the FMA for precision; bf16 output.
                result.x = bfloat(g.x * float(br.x) + float(res.x));
                result.y = bfloat(g.y * float(br.y) + float(res.y));
                result.z = bfloat(g.z * float(br.z) + float(res.z));
                result.w = bfloat(g.w * float(br.w) + float(res.w));
                out_v[tid * {vpt} + v] = result;
            }}
        """,
        ensure_row_contiguous=True,
    )


_gated_add_fused_kernels: dict[int, mx.fast.metal_kernel] = {
    c: _build_gated_add_kernel(c, block, vpt)
    for c, (block, vpt) in _GATED_ADD_FUSED_CONFIGS.items()
}


def _gated_add_mlx(
    residual: mx.array, branch: mx.array, gate: mx.array
) -> mx.array:
    """Stock MLX gated add — matches ``transformer.py::_residual_gate_inline``."""
    return (residual + branch * gate).astype(residual.dtype)


def _gated_add_t2v_broadcast_compatible(
    residual: mx.array, branch: mx.array, gate: mx.array
) -> bool:
    """Check whether (residual, branch, gate) match the T2V broadcast
    pattern the fused gated_add kernels expect, for any supported ``C``.

    Returns True iff:
      - residual, branch are 3D bf16 with last dim in ``_GATED_ADD_FUSED_CONFIGS``
        and ``residual.shape == branch.shape``
      - gate is fp32 with last dim == residual.shape[-1] and all-singleton
        leading dims
    """
    if residual.dtype != mx.bfloat16 or branch.dtype != mx.bfloat16:
        return False
    if residual.ndim != 3:
        return False
    c = residual.shape[-1]
    if c not in _gated_add_fused_kernels:
        return False
    if residual.shape != branch.shape:
        return False
    if gate.dtype != mx.float32 or gate.shape[-1] != c:
        return False
    for d in gate.shape[:-1]:
        if d != 1:
            return False
    return True


def gated_add_fused(
    residual: mx.array, branch: mx.array, gate: mx.array
) -> mx.array:
    """Fused gated add: ``residual + branch * gate``, cast to residual.dtype.

    Supports both video (C=4096) and audio (C=2048) production T2V shapes:
    residual/branch bf16 (B, T, C), gate fp32 with last dim C and all-
    singleton leading dims.  Falls back to the stock MLX expression for
    any other shape/dtype.

    Argument order matches ``_residual_gate_inline(x, residual, gate)``
    where x ≡ residual_arg, residual ≡ branch_arg, gate ≡ gate_arg.

    Args:
        residual: Accumulated residual stream, shape (B, T, C).
        branch: Sub-layer output to be gated (attn_out or ff_out).
        gate: Per-channel gate scalar, broadcastable to residual.

    Returns:
        ``(residual + branch * gate).astype(residual.dtype)``.
    """
    if not _gated_add_t2v_broadcast_compatible(residual, branch, gate):
        _probe("gated_add_fallback")
        return _gated_add_mlx(residual, branch, gate)

    _probe("gated_add_kernel")
    B, T, C = residual.shape
    block, _vpt = _GATED_ADD_FUSED_CONFIGS[C]
    kernel = _gated_add_fused_kernels[C]

    branch_2d = mx.contiguous(branch.reshape(B * T, C))
    residual_2d = mx.contiguous(residual.reshape(B * T, C))
    gate_1d = mx.contiguous(gate.reshape(-1))

    rows = B * T
    grid = (rows * block, 1, 1)
    threadgroup = (block, 1, 1)
    outputs = kernel(
        inputs=[branch_2d, gate_1d, residual_2d],
        output_shapes=[(rows, C)],
        output_dtypes=[mx.bfloat16],
        grid=grid,
        threadgroup=threadgroup,
    )
    return outputs[0].reshape(B, T, C)
