"""Microbenchmark: ALU throughput per dtype on M1 Max GPU.

The question: does Apple7 (M1 Max) have a dedicated INT8 dot-product instruction
that beats FP16 MAC throughput?

We compare 5 kernels, each running a tight MAC loop with all threads:
  1. FP32 scalar MAC: `acc = fma(a, b, acc)` per iter
  2. FP16 scalar MAC: `acc = fma((half)a, (half)b, acc)` per iter
  3. INT8 scalar MAC: `acc += (int)a * (int)b` per iter (INT32 accumulator)
  4. INT8 packed dp4a: `acc += dot(char4 a, char4 b)` per iter (4 MACs / iter)
  5. BF16 scalar MAC for completeness (Metal 3.1+)

Hypothesis: if M1 has dedicated INT8 hardware, kernel 4 yields ~4x the
MAC/sec of kernel 2.  If not, all kernels run at the same per-iteration
rate and INT8 has no throughput advantage.

This is the empirical answer to the dot4I8Packed feasibility question
that drove the perf-research thread.
"""
from __future__ import annotations

import time

import mlx.core as mx

# Each thread runs ITERS_PER_THREAD MAC operations.  TOTAL_THREADS threads run in parallel.
ITERS_PER_THREAD = 1_000_000
TOTAL_THREADS = 32 * 1024  # 1024 SIMD-groups of 32 threads


def _build_kernel(name: str, dtype_t: str, init_expr: str, mac_expr: str, store_expr: str):
    """Build a microbench kernel with given inner-loop expression."""
    source = f"""
        uint tid = thread_position_in_grid.x;
        if (tid >= TOTAL_THREADS) return;
        {dtype_t} acc = {init_expr};
        {dtype_t} a = ({dtype_t})((tid & 0x3f) + 1);
        {dtype_t} b = ({dtype_t})(((tid >> 1) & 0x3f) + 1);
        for (uint i = 0; i < ITERS; ++i) {{
            {mac_expr};
        }}
        {store_expr};
    """
    return source


def _build_int8_packed_kernel():
    """INT8 4-way dot product, manually unrolled (Metal has no dot(char4) intrinsic)."""
    source = """
        uint tid = thread_position_in_grid.x;
        if (tid >= TOTAL_THREADS) return;
        int acc = (int)tid;
        char4 a = char4((char)((tid & 0xf) + 1),
                        (char)(((tid >> 4) & 0xf) + 1),
                        (char)(((tid >> 8) & 0xf) + 1),
                        (char)(((tid >> 12) & 0xf) + 1));
        char4 b = char4((char)((tid & 0xf) + 2),
                        (char)(((tid >> 4) & 0xf) + 2),
                        (char)(((tid >> 8) & 0xf) + 2),
                        (char)(((tid >> 12) & 0xf) + 2));
        for (uint i = 0; i < ITERS; ++i) {
            int d = (int)a.x * (int)b.x
                  + (int)a.y * (int)b.y
                  + (int)a.z * (int)b.z
                  + (int)a.w * (int)b.w;
            acc += d;
            // Break optimization: mutate one component each iter
            a.x = (char)((acc & 0xf) + 1);
        }
        out[tid] = acc;
    """
    return source


def _bench_int_kernel(label: str, source: str, iters: int, total_threads: int,
                     macs_per_iter: int, dtype: mx.Dtype, warmup: int = 2, runs: int = 5):
    """Run a kernel benchmark with INT-typed output."""
    safe_name = "".join(c if c.isalnum() else "_" for c in label)
    kernel = mx.fast.metal_kernel(
        name=safe_name,
        input_names=["dummy"],
        output_names=["out"],
        source=source.replace("ITERS", str(iters)).replace("TOTAL_THREADS", str(total_threads)),
    )
    # dummy input (kernel needs at least one)
    dummy = mx.zeros((1,), dtype=mx.int32)

    def run():
        out = kernel(
            inputs=[dummy],
            grid=(total_threads, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(total_threads,)],
            output_dtypes=[dtype],
        )[0]
        mx.eval(out)
        return out

    # Warmup
    for _ in range(warmup):
        run()

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    mean_ms = 1000 * sum(times) / len(times)
    min_ms = 1000 * min(times)
    total_macs = iters * total_threads * macs_per_iter
    gops = total_macs / (min(times) * 1e9)
    tops = gops / 1000
    print(f"  {label:<32s}  best={min_ms:7.2f} ms  mean={mean_ms:7.2f} ms  "
          f"throughput = {tops:6.2f} TOps/s ({gops:.1f} GOps/s)")
    return tops


def main() -> None:
    print("=== ALU throughput microbench ===")
    print(f"  threads = {TOTAL_THREADS:,}, iters per thread = {ITERS_PER_THREAD:,}")
    print(f"  total MACs per kernel = {ITERS_PER_THREAD * TOTAL_THREADS:,}")
    print()

    # FP32 scalar MAC
    src = _build_kernel(
        "fp32_mac", "float",
        init_expr="(float)tid * 0.5f",
        mac_expr="acc = metal::fma(a, b, acc); a = acc * 0.0001f + a",  # break optimization
        store_expr="((device float*)out)[tid] = acc",
    )
    _bench_int_kernel("FP32 scalar fma", src, ITERS_PER_THREAD, TOTAL_THREADS,
                       macs_per_iter=1, dtype=mx.float32)

    # FP16 scalar MAC
    src = _build_kernel(
        "fp16_mac", "half",
        init_expr="(half)((float)tid * 0.5f)",
        mac_expr="acc = metal::fma(a, b, acc); a = acc * (half)0.0001 + a",
        store_expr="((device half*)out)[tid] = acc",
    )
    _bench_int_kernel("FP16 scalar fma", src, ITERS_PER_THREAD, TOTAL_THREADS,
                       macs_per_iter=1, dtype=mx.float16)

    # BF16 scalar MAC
    src = _build_kernel(
        "bf16_mac", "bfloat",
        init_expr="(bfloat)((float)tid * 0.5f)",
        mac_expr="acc = metal::fma(a, b, acc); a = acc * (bfloat)0.0001 + a",
        store_expr="((device bfloat*)out)[tid] = acc",
    )
    _bench_int_kernel("BF16 scalar fma", src, ITERS_PER_THREAD, TOTAL_THREADS,
                       macs_per_iter=1, dtype=mx.bfloat16)

    # INT8 scalar MAC (1 mul + 1 add per iter, INT32 accumulator)
    src = _build_kernel(
        "int8_mac", "int",  # accumulator dtype
        init_expr="(int)tid",
        mac_expr="acc += ((int)((tid & 0x7f) + (i & 0x3f) + 1)) * ((int)(((tid >> 1) & 0x7f) + 2))",
        store_expr="((device int*)out)[tid] = acc",
    )
    _bench_int_kernel("INT8 scalar mac (int32 acc)", src, ITERS_PER_THREAD, TOTAL_THREADS,
                       macs_per_iter=1, dtype=mx.int32)

    # INT8 packed dot (4 MACs per iter via dot(char4, char4))
    src = _build_int8_packed_kernel()
    _bench_int_kernel("INT8 dot(char4,char4) [4 MAC/iter]", src,
                       ITERS_PER_THREAD, TOTAL_THREADS,
                       macs_per_iter=4, dtype=mx.int32)

    print()
    print("Interpretation:")
    print("  If INT8 packed has >> FP16 MACs/sec: M1 has dedicated INT8 hardware")
    print("  If INT8 packed ~ FP16 MACs/sec or LOWER: no dedicated INT8 path on M1")
    print("    (and the 21 TFlops/s claim in PERFORMANCE_NOTES is wrong)")


if __name__ == "__main__":
    main()
