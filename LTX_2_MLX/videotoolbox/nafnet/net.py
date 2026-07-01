"""MLX port of NAFNet (Nonlinear Activation Free Network), Chen et al., ECCV 2022 --
"Simple Baselines for Image Restoration" (arXiv 2204.04676). Reimplemented from
basicsr/models/archs/NAFNet_arch.py as a spec; no upstream code run.

A residual U-Net of NAFBlocks. Each NAFBlock has no nonlinear activations: it uses
SimpleGate (split channels in half, multiply) instead of GELU/ReLU, and a Simplified
Channel Attention (global-avg-pool -> 1x1 conv -> multiply). Two residual sub-blocks
with learnable per-channel scales beta/gamma:
  x = norm1(inp); x = conv1(1x1) -> conv2(3x3 depthwise) -> SimpleGate -> x*SCA(x)
      -> conv3(1x1);  y = inp + beta*x
  x = conv4(1x1, on norm2(y)) -> SimpleGate -> conv5(1x1);  out = y + gamma*x
The net predicts a global residual: out = inp + body(inp), so a strength dial scales
that residual (1.0 = full, <1 = a light detail/deblur pass).

Config (width, enc/middle/dec block counts) is inferred from the checkpoint. Layout:
MLX-native NHWC. Every conv weight -> (O,kH,kW,I) (the 3x3 depthwise is (C,1,3,3) ->
(C,3,3,1), used with groups=C); beta/gamma (1,C,1,1) -> (1,1,1,C); LayerNorm weight/bias
stay (C,). Upsample = 1x1 conv (no bias) + PixelShuffle(2), padding is zero (F.pad
default), matching the reference.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# NAFNet task variants (same arch, different training + width); not bundled (see
# weights/README.md). width64 = quality, width32 = ~4x faster per frame.
_VARIANTS = {
    "gopro": "nafnet_gopro_width64.safetensors",     # motion deblur (GoPro), width 64
    "gopro32": "nafnet_gopro_width32.safetensors",   # motion deblur, width 32 (fast)
    "sidd": "nafnet_sidd_width64.safetensors",       # real-noise denoise (SIDD), width 64
    "sidd32": "nafnet_sidd_width32.safetensors",     # denoise, width 32
    "reds": "nafnet_reds_width64.safetensors",       # video restore (REDS), width 64
}
_DEFAULT_VARIANT = "gopro"
_MODEL_ZOO = "https://github.com/megvii-research/NAFNet/blob/main/docs/GoPro.md"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Variant token (gopro / gopro32 / sidd / sidd32 / reds) or a path; falls back to
    $NAFNET_WEIGHTS. Not bundled (the checkpoints are 68-464MB) -- a miss raises a
    download + conversion hint (see weights/README.md)."""
    if spec is None or spec == "":
        spec = os.environ.get("NAFNET_WEIGHTS")
    try:
        return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nNAFNet weights are not bundled in the repo (68-464MB each). Download a\n"
            f"checkpoint from the NAFNet model zoo ({_MODEL_ZOO}) and convert it:\n"
            f"  python scripts/pth_to_safetensors.py NAFNet-GoPro-width64.pth -o "
            f"{_WEIGHTS_DIR / _VARIANTS['gopro']} --strip-prefix ''\n"
            f"or point $NAFNET_WEIGHTS / --nafnet-weights at an existing .safetensors."
        ) from None


def load_params(path: str | Path | None = None, dtype: Any = mx.float32) -> dict:
    """Load + lay out the checkpoint. Conv weights -> NHWC (O,kH,kW,I) (the depthwise
    conv2 too, used with groups=C); beta/gamma -> (1,1,1,C); LayerNorm weight/bias and
    biases stay 1-D. Cast to `dtype` -- default float32, NOT fp16: NAFNet's SimpleGate
    multiplies two channel halves, so magnitudes square and overflow fp16's 65504 in this
    deep (31-block) net -> NaN. bf16 is safe (fp32 range) but its conv is ~1.4x slower
    than fp32 on M1, so fp32 is both the safe and the fast choice here."""
    w = mx.load(str(resolve_weights(path)))
    p: dict = {}
    for k, v in w.items():
        if k.endswith(".beta") or k.endswith(".gamma"):
            a = mx.reshape(v, (1, 1, 1, v.size))       # (1,C,1,1) -> (1,1,1,C)
        elif v.ndim == 4:
            a = mx.transpose(v, (0, 2, 3, 1))          # (O,I,kH,kW) -> (O,kH,kW,I)
        else:
            a = v
        p[k] = a.astype(dtype)
    return p


def _config(p: dict) -> tuple:
    """(width, enc_blk_nums, middle_blk_num, dec_blk_nums) inferred from the weights."""
    width = int(p["intro.weight"].shape[0])
    n_stages = sum(1 for k in p if k.startswith("downs.") and k.endswith(".weight"))

    def _count(prefix: str) -> int:
        idx = {int(k[len(prefix):].split(".")[0]) for k in p if k.startswith(prefix)}
        return (max(idx) + 1) if idx else 0

    enc = tuple(_count(f"encoders.{i}.") for i in range(n_stages))
    dec = tuple(_count(f"decoders.{i}.") for i in range(n_stages))
    return width, enc, _count("middle_blks."), dec


def _conv(x: Any, p: dict, key: str, stride: int = 1, pad: int = 0, groups: int = 1) -> Any:
    return mx.conv2d(x, p[f"{key}.weight"], stride=stride, padding=pad, groups=groups) + p[f"{key}.bias"]


def _layernorm(x: Any, w: Any, b: Any, eps: float = 1e-6) -> Any:
    """Channel-wise LayerNorm2d (over the last NHWC axis): (x-mu)/sqrt(var+eps)*w + b,
    var biased. Done in float32 -- the per-location reduction over C coarsens in fp16."""
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps)
    return (y * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


def _simplegate(x: Any) -> Any:
    c = x.shape[-1] // 2
    return x[..., :c] * x[..., c:]


def _sca(x: Any, p: dict, key: str) -> Any:
    """Simplified channel attention: global avg pool -> 1x1 conv -> per-channel scale."""
    pooled = mx.mean(x, axis=(1, 2), keepdims=True)          # (N,1,1,C)
    return _conv(pooled, p, key)


def _naf_block(x: Any, p: dict, prefix: str) -> Any:
    inp = x
    y = _layernorm(x, p[f"{prefix}.norm1.weight"], p[f"{prefix}.norm1.bias"])
    y = _conv(y, p, f"{prefix}.conv1")                       # 1x1, c -> 2c
    y = _conv(y, p, f"{prefix}.conv2", pad=1, groups=y.shape[-1])   # 3x3 depthwise
    y = _simplegate(y)                                       # 2c -> c
    y = y * _sca(y, p, f"{prefix}.sca.1")
    y = _conv(y, p, f"{prefix}.conv3")                       # 1x1, c -> c
    x = inp + y * p[f"{prefix}.beta"]
    z = _layernorm(x, p[f"{prefix}.norm2.weight"], p[f"{prefix}.norm2.bias"])
    z = _conv(z, p, f"{prefix}.conv4")                       # 1x1, c -> 2c
    z = _simplegate(z)                                       # 2c -> c
    z = _conv(z, p, f"{prefix}.conv5")                       # 1x1, c -> c
    return x + z * p[f"{prefix}.gamma"]


def _pixel_shuffle(x: Any, r: int) -> Any:
    """NHWC PixelShuffle(r): (N,H,W,C*r*r) -> (N,H*r,W*r,C), channel = c*r*r + i*r + j."""
    n, h, w, crr = x.shape
    c = crr // (r * r)
    x = mx.reshape(x, (n, h, w, c, r, r))
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))                 # N, H, r_h, W, r_w, C
    return mx.reshape(x, (n, h * r, w * r, c))


def _pad(x: Any, m: int) -> Any:
    """Zero-pad bottom/right to a multiple of m (F.pad default, matching the reference)."""
    _, h, w, _ = x.shape
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = mx.pad(x, [(0, 0), (0, ph), (0, pw), (0, 0)])
    return x


def nafnet(inp: Any, p: dict, cfg: tuple | None = None, strength: float = 1.0) -> Any:
    """Restore one batch. inp: (N,H,W,3) in [0,1]. strength scales the predicted residual
    (out = inp + strength*body(inp); 1.0 = full, <1 = light). Returns (N,H,W,3)."""
    if cfg is None:
        cfg = _config(p)
    width, enc_nums, mid_num, dec_nums = cfg
    n_stages = len(enc_nums)
    h, w = inp.shape[1], inp.shape[2]
    dt = p["intro.weight"].dtype
    x_pad = _pad(inp.astype(dt), 2 ** n_stages)

    x = _conv(x_pad, p, "intro", pad=1)                     # 3x3, 3 -> width
    encs = []
    for i in range(n_stages):
        for b in range(enc_nums[i]):
            x = _naf_block(x, p, f"encoders.{i}.{b}")
        encs.append(x)
        x = _conv(x, p, f"downs.{i}", stride=2)             # 2x2 stride-2, chan -> 2*chan
    for b in range(mid_num):
        x = _naf_block(x, p, f"middle_blks.{b}")
    for i in range(n_stages):
        x = mx.conv2d(x, p[f"ups.{i}.0.weight"], padding=0)  # 1x1, no bias
        x = _pixel_shuffle(x, 2)                            # -> chan/2 at 2x
        x = x + encs[n_stages - 1 - i]
        for b in range(dec_nums[i]):
            x = _naf_block(x, p, f"decoders.{i}.{b}")
    residual = _conv(x, p, "ending", pad=1)                 # 3x3, width -> 3
    out = x_pad + strength * residual
    return out[:, :h, :w, :]


_COMPILE_CACHE: dict = {}


def make_forward(p: dict, strength: float = 1.0, cfg: tuple | None = None, compile: bool = True):
    """Per-frame forward x -> restored image for a fixed strength, mx.compiled once per
    checkpoint + strength (pair with a capped MLX cache, which the harness sets)."""
    if cfg is None:
        cfg = _config(p)

    def run(x):
        return nafnet(x, p, cfg=cfg, strength=strength)

    if not compile:
        return run
    key = (id(p), float(strength), cfg)
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        fn = mx.compile(run)
        _COMPILE_CACHE[key] = fn
    return fn


if __name__ == "__main__":
    p = load_params(dtype=mx.float32)
    cfg = _config(p)
    print(f"loaded NAFNet: width={cfg[0]} enc={cfg[1]} middle={cfg[2]} dec={cfg[3]}")
    mx.random.seed(0)
    x = mx.clip(mx.random.uniform(shape=(1, 64, 96, 3)), 0, 1)
    mx.eval(x)
    out = nafnet(x, p, cfg)
    mx.eval(out)
    print(f"{tuple(x.shape)} -> {tuple(out.shape)}, residual mean {float(mx.mean(mx.abs(out - x))):.4f}")
