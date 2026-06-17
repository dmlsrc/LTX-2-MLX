# MLX float64: real-double vs silent-float32 functions

Short version: on Apple Silicon, MLX has **no float64 on the GPU at all**
(Metal has no `double`), and on the **CPU backend a handful of transcendental
functions silently compute in float32 even when the array dtype is float64**.
The result array still reports `dtype=float64`, so the precision loss is
invisible unless you check the values.

Verified on MLX 0.31.2 (the version this repo pins). The same behavior is in
the MLX `main` source - an untagged snapshot ~85 commits past the v0.31.2 tag
(its `version.h` reads 0.32.0 as the next in-development version, not a
release). This may change in a future MLX release; re-check before relying on it.

## Silently float32 (CPU, float64 input)

`sin`, `cos`, `exp`, `sigmoid`, `erf`, `erfinv`.

- Accuracy is ~1e-7 relative (float32), not ~1e-16.
- `exp` additionally leaks its float32 overflow edge into float64: it returns
  `inf` for any input > 88, even though float64 `exp` is finite out to ~709.
  So `mx.exp(mx.array(90.0, dtype=mx.float64))` is `inf` - wrong; the true
  value 1.2e39 fits in float64.
- Tell-tale: `mx.sin(pi)` returns ~8.74e-08 (the float32 result) instead of
  the true ~1.2e-16.

## Genuinely float64 (CPU)

`tan`, `tanh`, `log`, `log2`, `log10`, `log1p`, `expm1`, `sqrt`, `rsqrt`,
`reciprocal`, `abs`, `square`, `negate`, `sign`, `floor`, `ceil`, `round`,
all comparisons and reductions, and all arithmetic (`+ - * / **`).

Note the asymmetry - it is per-kernel, not a category rule:

- `exp` is degraded but `expm1` is fine.
- `sin`/`cos` are degraded but `tan`/`tanh` are fine.
- `erf` is degraded but `log` is fine.

## Why

`mlx/backend/cpu/simd/math.h` implements `exp`, `sincos` (sin/cos), `erf`, and
`erfinv` as float32-only SIMD polynomial approximations whose first line
narrows the input to `Simd<float, N>` regardless of the element type.
Everything else routes through `base_simd.h` (`std::`) or `accelerate_simd.h`
(Apple `asd::`), which keep `double`.

## What to do

- Do not use MLX CPU float64 as a high-precision reference/oracle for
  `sin`/`cos`/`exp`/`erf` (RoPE angle tables, softmax/exp reductions, bf16/fp16
  parity baselines, torch-float64 comparisons). You will silently get float32.
- Compute those in NumPy or the stdlib `math` module (both true float64) and
  move the result into MLX.
- Concrete example in this repo: the BWE vocoder's Hann-windowed sinc filter
  (`LTX_2_MLX/model/audio_vae/vocoder.py`, `UpSample1d`) is built once at init
  with the stdlib `math` module for exactly this reason - `mx.sin`/`mx.cos`
  cannot reproduce the float64 taps, and the result is frozen to float32 anyway.

## How to check a function yourself

Compare the MLX float64 output against the NumPy float64 result and the NumPy
float32 result. If it tracks the float32 result, the kernel downcast:

```python
import numpy as np, mlx.core as mx
mx.set_default_device(mx.cpu)
x = np.array([0.1, 1.234, 21.77], dtype=np.float64)
g = np.array(mx.cos(mx.array(x, dtype=mx.float64)))
print(np.max(np.abs(g - np.cos(x))))                                 # ~1e-7 if degraded
print(np.max(np.abs(g - np.cos(x.astype(np.float32)).astype(np.float64))))  # ~0 if degraded
```
