"""MLX port of ESC-Real (Lee et al., "Emulating Self-attention with Convolution for
Efficient Image Super-Resolution", ICCV 2025) -- the real-world ESCRealM generator.
Reimplemented from the reference architecture as a spec; no upstream code is bundled.

Per-frame 4x upscaler trained with the high-order Real-ESRGAN degradation (blur
included), so it is video-appropriate. Two checkpoints: "gan" (perceptual) and "mse"
(fidelity), same architecture. Structure (dim 64, 10 blocks in the released weights):

  proj 3x3 -> n_blocks x Block -> last 3x3 + feature skip + skip branch(input)
  -> nearest+conv upsample tail -> RGB

  Block: LN -> ConvFFN -> +window attention (32x32 windows, additive relative-
  position bias) -> conv_blocks x [LN -> ConvFFN -> partial large-kernel conv
  (static 13x13 on the first pdim=16 channels via a geo-ensembled shared filter,
  plus a per-frame DYNAMIC 3x3 depthwise whose taps are predicted from the pooled
  features)] -> LN -> conv 3x3 -> +skip.

Port notes (see docs/VSR_PERFORMANCE_NOTES.md for the general rules):
- The released checkpoint's upsample tail is Upsample-FIRST (params at to_img
  indices 1,4,6,8: up2 -> conv -> lrelu, x2, then conv -> lrelu -> conv); the
  current upstream source builds conv-first -- the weights win.
- The relative-position bias is expanded from the (heads, (2w-1)^2) table into a
  dense (heads, w^2, w^2) additive mask once at load; attention runs through
  mx.fast.scaled_dot_product_attention.
- The skip branch's 7x7 depthwise uses REFLECT padding (padding_mode='reflect').
- Depthwise 3x3s (ConvFFN dwc, dynamic conv) run as the 9-tap shift-add; the
  13x13 partial conv is a dense 16->16 conv on the good implicit-GEMM path.
- The geo-ensemble of the 13x13 filter (mean over 8 flips/rotations) is folded
  once at load.

Layout: MLX-native NHWC; conv weights -> (O,kH,kW,I) at load.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..compile_cache import cached as _cached
from ..weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# Not bundled (download + convert; see weights/README.md).
_VARIANTS = {
    "gan": "esc_real_x4_gan.safetensors",    # perceptual (Real-ESRGAN-style GAN)
    "mse": "esc_real_x4_mse.safetensors",    # fidelity twin
}
_DEFAULT_VARIANT = "gan"
_REPO = "https://github.com/dslisleedh/ESC"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Variant token (gan / mse) or a path; falls back to $ESC_WEIGHTS."""
    if spec is None or spec == "":
        spec = os.environ.get("ESC_WEIGHTS")
    try:
        return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nESC weights are not bundled. Download ESC_Real_X4_GAN.pth / "
            f"ESC_Real_X4_MSE.pth from the ESC releases ({_REPO}/releases) and convert "
            f"(see weights/README.md), or point $ESC_WEIGHTS / --esc-weights at an "
            f"existing .safetensors."
        ) from None


def _rot90_cw(k: Any) -> Any:
    """torch.rot90(k, -1, [2, 3]) on (O,I,kh,kw): out[..., i, j] = in[..., K-1-j, i]."""
    return mx.transpose(k, (0, 1, 3, 2))[..., ::-1]


def _geo_ensemble(k: Any) -> Any:
    """Mean over the 8 flip/rotation views of the shared large-kernel filter."""
    k_h = k[..., ::-1]
    k_v = k[..., ::-1, :]
    k_hv = k[..., ::-1, ::-1]
    k_r = _rot90_cw(k)
    k_rh = k_r[..., ::-1]
    k_rv = k_r[..., ::-1, :]
    k_rhv = k_r[..., ::-1, ::-1]
    return (k + k_h + k_v + k_hv + k_r + k_rh + k_rv + k_rhv) / 8.0


def _expand_rpe(table: Any, heads: int) -> Any:
    """(heads, (2w-1)^2) relative-position table -> dense (heads, w^2, w^2) bias."""
    side = int(round(table.shape[1] ** 0.5))          # 2w-1
    w = (side + 1) // 2
    pos = mx.arange(w * w)
    qh, qw = pos // w, pos % w
    rel_h = qh[None, :] - qh[:, None] + w - 1         # k - q, transposed below
    rel_w = qw[None, :] - qw[:, None] + w - 1
    idx = (rel_h * side + rel_w).T                    # [q, k] with rel = k - q
    return table[:, idx.reshape(-1)].reshape(heads, w * w, w * w)


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: conv weights -> NHWC (O,kH,kW,I); the 13x13
    filter is geo-ensembled here (fold-once); each attention's relative-position
    table is expanded to a dense additive bias under '...attn.bias_dense'."""
    w = mx.load(str(resolve_weights(path)))
    p: dict = {}
    for k, v in w.items():
        if k == "plk_filter":
            v = _geo_ensemble(v.astype(mx.float32)).astype(v.dtype)
        if v.ndim == 4:
            a = mx.transpose(v, (0, 2, 3, 1))
        else:
            a = v
        p[k] = a.astype(dtype)
    if "proj.weight" not in p or "blocks.0.attn.relative_position_bias" not in p:
        raise ValueError("not an ESCRealM checkpoint")
    i = 0
    while f"blocks.{i}.attn.relative_position_bias" in p:
        t = p[f"blocks.{i}.attn.relative_position_bias"]
        p[f"blocks.{i}.attn.bias_dense"] = _expand_rpe(t.astype(mx.float32), t.shape[0]).astype(dtype)
        i += 1
    return p


def _config(p: dict) -> tuple:
    """(dim, n_blocks, conv_blocks, heads, window, pdim, scale) from the weights."""
    dim = int(p["proj.weight"].shape[0])
    n_blocks = 0
    while f"blocks.{n_blocks}.attn.to_qkv.weight" in p:
        n_blocks += 1
    conv_blocks = 0
    while f"blocks.0.pconvs.{conv_blocks}.plk.dwc_proj.1.weight" in p:
        conv_blocks += 1
    heads = int(p["blocks.0.attn.relative_position_bias"].shape[0])
    side = int(round(p["blocks.0.attn.relative_position_bias"].shape[1] ** 0.5))
    window = (side + 1) // 2
    pdim = int(p["plk_filter"].shape[0])
    ups = sum(1 for k in ("to_img.1.weight", "to_img.4.weight") if k in p)
    return dim, n_blocks, conv_blocks, heads, window, pdim, 2 ** ups


def _lrelu(x: Any, slope: float = 0.2) -> Any:
    return mx.where(x >= 0, x, x * slope)


def _gelu(x: Any) -> Any:
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _conv(x: Any, p: dict, key: str, pad: int = 0) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=1, padding=pad)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _ln(x: Any, p: dict, key: str, eps: float = 1e-6) -> Any:
    """Channel LayerNorm (fp32 reduction; manual on purpose -- see the perf notes)."""
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps)
    return (y * p[f"{key}.weight"].astype(mx.float32)
            + p[f"{key}.bias"].astype(mx.float32)).astype(x.dtype)


def _dw3x3(x: Any, w: Any, b: Any = None) -> Any:
    """3x3 depthwise (pad 1) as the 9-tap shift-add; w is (C,3,3,1) or per-frame
    computed taps shaped (3,3,C)."""
    h, wd = x.shape[1], x.shape[2]
    xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    acc = None
    for i in range(3):
        for j in range(3):
            wij = w[:, i, j, 0] if w.ndim == 4 else w[i, j]
            t = xp[:, i:i + h, j:j + wd, :] * wij
            acc = t if acc is None else acc + t
    return acc if b is None else acc + b


def _conv_ffn(x: Any, p: dict, pre: str) -> Any:
    """gelu(1x1 expand) -> gelu(3x3 depthwise) + residual -> 1x1 aggregate."""
    y = _gelu(_conv(x, p, f"{pre}.proj"))
    y = _gelu(_dw3x3(y, p[f"{pre}.dwc.weight"], p[f"{pre}.dwc.bias"])) + y
    return _conv(y, p, f"{pre}.aggr")


def _plk_attn(x: Any, p: dict, pre: str, pdim: int) -> Any:
    """Partial large-kernel conv: on the first pdim channels, a static 13x13 conv
    (geo-ensembled shared filter) plus a per-frame dynamic 3x3 depthwise whose taps
    come from the pooled features; the remaining channels pass through."""
    x1 = x[..., :pdim]
    # Global pool in fp32: summing H*W (~400k at 480p) fp16 values overflows 65504.
    pooled = mx.mean(x1.astype(mx.float32), axis=(1, 2), keepdims=True).astype(x1.dtype)
    dk = _gelu(_conv(pooled, p, f"{pre}.plk.dwc_proj.1"))
    dk = _conv(dk, p, f"{pre}.plk.dwc_proj.3")                      # (N,1,1,pdim*9)
    dk = dk.reshape(pdim, 3, 3).transpose(1, 2, 0)                  # (3,3,pdim), N==1
    stat = mx.conv2d(x1, p["plk_filter"], stride=1, padding=6)      # 13x13, 16->16
    y1 = stat + _dw3x3(x1, dk)
    y = mx.concatenate([y1, x[..., pdim:]], axis=-1)
    return _conv(y, p, f"{pre}.aggr")


def _window_attn(x: Any, p: dict, pre: str, heads: int, window: int) -> Any:
    """32x32 window attention with the dense additive relative-position bias."""
    n, h, w, dim = x.shape
    ph, pw = (-h) % window, (-w) % window
    if ph or pw:   # reflect-pad bottom/right to window multiples ([::-1] mirrors)
        if ph:
            x = mx.concatenate([x, x[:, h - 1 - ph:h - 1, :, :][:, ::-1, :, :]], axis=1)
        if pw:
            x = mx.concatenate([x, x[:, :, w - 1 - pw:w - 1, :][:, :, ::-1, :]], axis=2)
    hp, wp = x.shape[1], x.shape[2]
    nh, nw = hp // window, wp // window
    hd = dim // heads
    qkv = _conv(x, p, f"{pre}.to_qkv")                              # (n,hp,wp,3*dim)
    # channel layout (qkv, heads, hd); window-partition to (3, n*nh*nw, heads, win^2, hd)
    qkv = qkv.reshape(n, nh, window, nw, window, 3, heads, hd)
    qkv = qkv.transpose(5, 0, 1, 3, 6, 2, 4, 7)                     # 3,n,nh,nw,heads,wh,ww,hd
    qkv = qkv.reshape(3, n * nh * nw, heads, window * window, hd)
    q, k, v = qkv[0], qkv[1], qkv[2]
    bias = p[f"{pre}.bias_dense"][None]                             # (1,heads,N,N)
    # Manual attention: mx.fast SDPA with a dense additive mask at head_dim 16 lands
    # on a slow path (~5x slower than these two batched GEMMs + a precise softmax).
    scores = (q * hd ** -0.5) @ k.transpose(0, 1, 3, 2) + bias
    o = mx.softmax(scores, axis=-1, precise=True) @ v
    o = o.reshape(n, nh, nw, heads, window, window, hd)
    o = o.transpose(0, 1, 4, 2, 5, 3, 6).reshape(n, hp, wp, dim)
    o = o[:, :h, :w, :]
    return _conv(o, p, f"{pre}.to_out")


def _block(x: Any, p: dict, i: int, cfg: tuple) -> Any:
    _dim, _nb, conv_blocks, heads, window, pdim, _s = cfg
    skip = x
    x = _conv_ffn(_ln(x, p, f"blocks.{i}.ln_proj"), p, f"blocks.{i}.proj")
    x = x + _window_attn(_ln(x, p, f"blocks.{i}.ln_attn"), p, f"blocks.{i}.attn",
                         heads, window)
    for j in range(conv_blocks):
        y = _conv_ffn(_ln(x, p, f"blocks.{i}.lns.{j}"), p, f"blocks.{i}.convffns.{j}")
        x = x + _plk_attn(y, p, f"blocks.{i}.pconvs.{j}", pdim)
    x = _conv(_ln(x, p, f"blocks.{i}.ln_out"), p, f"blocks.{i}.conv_out", pad=1)
    return x + skip


def _skip_branch(x: Any, p: dict) -> Any:
    """Input skip: 1x1 expand -> 7x7 depthwise (REFLECT pad) -> lrelu -> 1x1."""
    y = _conv(x, p, "skip.0")
    # reflect-pad 3 then a 7x7 depthwise via conv2d (C%16==0 -> the dedicated path)
    h, w = y.shape[1], y.shape[2]
    yp = mx.concatenate([y[:, 1:4, :, :][:, ::-1, :, :], y,
                         y[:, h - 4:h - 1, :, :][:, ::-1, :, :]], axis=1)
    yp = mx.concatenate([yp[:, :, 1:4, :][:, :, ::-1, :], yp,
                         yp[:, :, w - 4:w - 1, :][:, :, ::-1, :]], axis=2)
    y = mx.conv2d(yp, p["skip.1.weight"], stride=1, padding=0,
                  groups=y.shape[-1]) + p["skip.1.bias"]
    return _conv(_lrelu(y), p, "skip.3")


def esc(x: Any, p: dict, cfg: tuple | None = None) -> Any:
    """Upscale one frame. x: (1,H,W,3) in [0,1] -> (1, scale*H, scale*W, 3)."""
    if cfg is None:
        cfg = _config(p)
    dt = p["proj.weight"].dtype
    x = x.astype(dt)
    feat = _conv(x, p, "proj", pad=1)
    skip = feat
    y = feat
    for i in range(cfg[1]):
        y = _block(y, p, i, cfg)
    y = _conv(y, p, "last", pad=1) + skip + _skip_branch(x, p)
    # nearest+conv tail, Upsample-first per the released weights (params at 1,4,6,8)
    y = _lrelu(_conv(_up2(y), p, "to_img.1", pad=1))
    y = _lrelu(_conv(_up2(y), p, "to_img.4", pad=1))
    y = _lrelu(_conv(y, p, "to_img.6", pad=1))
    y = _conv(y, p, "to_img.8", pad=1)
    return mx.clip(y, 0.0, 1.0)


def _up2(x: Any) -> Any:
    n, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, 2, w, 2, c))
    return x.reshape(n, h * 2, w * 2, c)


_COMPILE_CACHE: dict = {}


def make_forward(p: dict, cfg: tuple | None = None, compile: bool = True):
    """Per-frame forward x -> upscaled image, mx.compiled once per checkpoint."""
    if cfg is None:
        cfg = _config(p)

    def run(x):
        return esc(x, p, cfg=cfg)

    if not compile:
        return run
    return _cached(_COMPILE_CACHE, (id(p), cfg), lambda: mx.compile(run))


if __name__ == "__main__":
    p = load_params()
    cfg = _config(p)
    print(f"loaded ESC-Real: dim={cfg[0]} blocks={cfg[1]}x{cfg[2]} heads={cfg[3]} "
          f"window={cfg[4]} pdim={cfg[5]} scale={cfg[6]}x")
    mx.random.seed(0)
    x = mx.clip(mx.random.uniform(shape=(1, 64, 96, 3)), 0, 1)
    mx.eval(x)
    out = esc(x, p, cfg)
    mx.eval(out)
    print(f"{tuple(x.shape)} -> {tuple(out.shape)}, finite={bool(mx.all(mx.isfinite(out)))}")
