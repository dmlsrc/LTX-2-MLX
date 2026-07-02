"""MLX port of the SAFMN super-resolution family (Sun et al., "Spatially-Adaptive
Feature Modulation for Efficient Image Super-Resolution", ICCV 2023, and its
challenge-winning successors). Reimplemented from the reference architectures as a
spec; no upstream code is bundled. Two variants, auto-detected from the checkpoint:

- "real" (SAFMN-L / Real_SAFMN++, 1st place AIM 2025 Efficient Perceptual SR):
  dim 96 x 16 AttBlocks; channel LayerNorm, 4-level SAFM pyramid (max-pool /2^i,
  per-level 3x3 depthwise, nearest upsample, 1x1 aggregate, GELU gate) and a CCM
  FFN (3x3 expand + GELU + 1x1). Perceptual/GAN-trained 4x -- pair it AFTER
  deblock/denoise so it does not amplify compression artifacts.
- "light" (light_SAFMN++, 1st place fidelity track, AIS 2024 Real-Time 4K SR of
  compressed AVIF): dim 32 x 2 blocks; no norms, no biases, SimpleSAFM (single /8
  max-pool level, 3x3 depthwise, bilinear upsample, GELU gate) + CCM. Fidelity-
  trained on compressed content; ~45x fewer parameters.

Layout: MLX-native NHWC; conv weights -> (O,kH,kW,I) at load; the depthwise convs
run as the 9-tap shift-add (their channel counts of 24/16 fail MLX's depthwise-gate
C%16 check -- see docs/VSR_PERFORMANCE_NOTES.md). Input is replicate-padded to a
multiple of 8 (the deepest pooling level) and the output cropped back.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..compile_cache import cached as _cached
from ..vsr_blocks import resize as _resize_bilinear
from ..weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# Not bundled (download + convert; see weights/README.md).
_VARIANTS = {
    "real": "safmn_l_real.safetensors",      # SAFMN-L, perceptual 4x (AIM 2025)
    "light": "light_safmnpp.safetensors",    # light_SAFMN++, fidelity 4x (AIS 2024)
}
_DEFAULT_VARIANT = "real"
_REPO = "https://github.com/sunny2109/SAFMN"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Variant token (real / light) or a path; falls back to $SAFMN_WEIGHTS."""
    if spec is None or spec == "":
        spec = os.environ.get("SAFMN_WEIGHTS")
    try:
        return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nSAFMN weights are not bundled. Download a checkpoint from the SAFMN "
            f"repo ({_REPO}) and convert it (see weights/README.md), or point "
            f"$SAFMN_WEIGHTS / --safmn-weights at an existing .safetensors."
        ) from None


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: conv weights -> NHWC (O,kH,kW,I); the depthwise
    weights (C,1,3,3) -> (C,3,3,1) for the shift-add; LayerNorm weight/bias stay 1-D."""
    w = mx.load(str(resolve_weights(path)))
    p: dict = {}
    for k, v in w.items():
        if v.ndim == 4:
            a = mx.transpose(v, (0, 2, 3, 1))
        else:
            a = v
        p[k] = a.astype(dtype)
    if "to_feat.weight" not in p:
        raise ValueError("not a SAFMN checkpoint (missing to_feat.weight)")
    return p


def _config(p: dict) -> tuple:
    """(variant, dim, n_blocks, scale) inferred from the weights."""
    variant = "real" if "feats.0.norm1.weight" in p else "light"
    dim = int(p["to_feat.weight"].shape[0])
    i = 0
    while f"feats.{i}.conv1.proj.weight" in p or f"feats.{i}.norm1.weight" in p:
        i += 1
    scale = int(round((p["to_img.0.weight"].shape[0] / 3) ** 0.5))
    return variant, dim, i, scale


def _gelu(x: Any) -> Any:
    """Exact GELU (erf form), matching torch nn.GELU()."""
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _conv(x: Any, p: dict, key: str, pad: int = 0) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=1, padding=pad)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _dw3x3(x: Any, p: dict, key: str) -> Any:
    """3x3 depthwise (pad 1) as 9 shifted per-channel-scaled adds. The variants'
    depthwise channel counts (24 / 16) fail MLX's depthwise-gate C%16 check, and the
    shift-add is never pathological (see nafnet)."""
    w = p[f"{key}.weight"]
    h, wd = x.shape[1], x.shape[2]
    xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    acc = None
    for i in range(3):
        for j in range(3):
            t = xp[:, i:i + h, j:j + wd, :] * w[:, i, j, 0]
            acc = t if acc is None else acc + t
    b = p.get(f"{key}.bias")
    return acc if b is None else acc + b


def _layernorm(x: Any, w: Any, b: Any, eps: float = 1e-6) -> Any:
    """Channel LayerNorm (torch channels_first): (x-mu)/sqrt(var+eps)*w + b over the
    channel axis, biased var, fp32 reduction. Manual on purpose -- mx.fast.layer_norm
    is transformer-shaped and slower for small-C/many-row norms."""
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps)
    return (y * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


def _maxpool(x: Any, k: int) -> Any:
    """k x k max pool, stride k (H, W multiples of k -- guaranteed by the pad-to-8).
    Matches torch adaptive_max_pool2d to (H/k, W/k) exactly under divisibility."""
    n, h, w, c = x.shape
    return mx.max(x.reshape(n, h // k, k, w // k, k, c), axis=(2, 4))


def _nearest_up(x: Any, r: int) -> Any:
    n, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, r, w, r, c))
    return x.reshape(n, h * r, w * r, c)


def _pixel_shuffle(x: Any, r: int) -> Any:
    n, h, w, cr = x.shape
    c = cr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _replicate_pad(x: Any, m: int) -> Any:
    """Replicate-pad bottom/right so H, W are multiples of m."""
    _, h, w, _ = x.shape
    ph, pw = (-h) % m, (-w) % m
    if ph:
        x = mx.concatenate([x, mx.broadcast_to(x[:, h - 1:h, :, :], (x.shape[0], ph, x.shape[2], x.shape[3]))], axis=1)
    if pw:
        x = mx.concatenate([x, mx.broadcast_to(x[:, :, w - 1:w, :], (x.shape[0], x.shape[1], pw, x.shape[3]))], axis=2)
    return x


# ---- "real" variant blocks (SAFMN-L) ----------------------------------------
def _safm(x: Any, p: dict, pre: str) -> Any:
    """4-level spatially-adaptive feature modulation: chunk channels into 4, level i
    max-pools by 2^i, runs a per-level depthwise 3x3, nearest-upsamples back; the
    concatenated levels pass a 1x1 aggregate and gate x via GELU."""
    c4 = x.shape[-1] // 4
    outs = []
    for i in range(4):
        xc = x[..., i * c4:(i + 1) * c4]
        if i == 0:
            s = _dw3x3(xc, p, f"{pre}.mfr.0")
        else:
            s = _maxpool(xc, 2 ** i)
            s = _dw3x3(s, p, f"{pre}.mfr.{i}")
            s = _nearest_up(s, 2 ** i)
        outs.append(s)
    out = _conv(mx.concatenate(outs, axis=-1), p, f"{pre}.aggr")
    return _gelu(out) * x


def _ccm(x: Any, p: dict, pre: str) -> Any:
    return _conv(_gelu(_conv(x, p, f"{pre}.ccm.0", pad=1)), p, f"{pre}.ccm.2")


def _att_block_real(x: Any, p: dict, i: int) -> Any:
    x = _safm(_layernorm(x, p[f"feats.{i}.norm1.weight"], p[f"feats.{i}.norm1.bias"]),
              p, f"feats.{i}.safm") + x
    x = _ccm(_layernorm(x, p[f"feats.{i}.norm2.weight"], p[f"feats.{i}.norm2.bias"]),
             p, f"feats.{i}.ccm") + x
    return x


# ---- "light" variant blocks (light_SAFMN++) ----------------------------------
def _simple_safm(x: Any, p: dict, pre: str) -> Any:
    """Single-level modulation: 3x3 proj, split channels; one half max-pools to
    (H/8, W/8), runs a depthwise 3x3, bilinear-upsamples back and GELU-gates its
    source; concat with the other half, GELU, 1x1 out. No biases."""
    h, w = x.shape[1], x.shape[2]
    y = _conv(x, p, f"{pre}.proj", pad=1)
    d = y.shape[-1] // 2
    x0, x1 = y[..., :d], y[..., d:]
    s = _maxpool(x0, 8)
    s = _dw3x3(s, p, f"{pre}.dwconv")
    s = _resize_bilinear(s, h, w, False)
    s = _gelu(s) * x0
    z = mx.concatenate([x1, s], axis=-1)
    return _conv(_gelu(z), p, f"{pre}.out")


def _att_block_light(x: Any, p: dict, i: int) -> Any:
    y = _simple_safm(x, p, f"feats.{i}.conv1")
    y = _conv(_gelu(_conv(y, p, f"feats.{i}.conv2.conv.0", pad=1)), p, f"feats.{i}.conv2.conv.2")
    return y                                     # no per-block residual in the light net


def safmn(x: Any, p: dict, cfg: tuple | None = None) -> Any:
    """Upscale one batch. x: (N,H,W,3) in [0,1] -> (N, scale*H, scale*W, 3)."""
    if cfg is None:
        cfg = _config(p)
    variant, _dim, n_blocks, scale = cfg
    dt = p["to_feat.weight"].dtype
    n, h, w, _ = x.shape
    xp = _replicate_pad(x.astype(dt), 8)
    feat = _conv(xp, p, "to_feat", pad=1)
    y = feat
    for i in range(n_blocks):
        y = _att_block_real(y, p, i) if variant == "real" else _att_block_light(y, p, i)
    y = y + feat
    out = _pixel_shuffle(_conv(y, p, "to_img.0", pad=1), scale)
    return mx.clip(out[:, :h * scale, :w * scale, :], 0.0, 1.0)


_COMPILE_CACHE: dict = {}


def make_forward(p: dict, cfg: tuple | None = None, compile: bool = True):
    """Per-frame forward x -> upscaled image, mx.compiled once per checkpoint."""
    if cfg is None:
        cfg = _config(p)

    def run(x):
        return safmn(x, p, cfg=cfg)

    if not compile:
        return run
    return _cached(_COMPILE_CACHE, (id(p), cfg), lambda: mx.compile(run))


if __name__ == "__main__":
    p = load_params()
    cfg = _config(p)
    print(f"loaded SAFMN: variant={cfg[0]} dim={cfg[1]} blocks={cfg[2]} scale={cfg[3]}x")
    mx.random.seed(0)
    x = mx.clip(mx.random.uniform(shape=(1, 64, 96, 3)), 0, 1)
    mx.eval(x)
    out = safmn(x, p, cfg)
    mx.eval(out)
    print(f"{tuple(x.shape)} -> {tuple(out.shape)}, finite={bool(mx.all(mx.isfinite(out)))}")
