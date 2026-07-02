"""MLX port of RealViformer (Zhang and Yao, "RealViformer: Investigating Attention
for Real-World Video Super-Resolution", ECCV 2024). Reimplemented from the reference
architecture as a spec; no upstream code is bundled.

A CAUSAL recurrent real-world 4x video upscaler: per frame, a shallow transformer
extracts features; the previous frame's propagated features are flow-warped (SpyNet)
and merged via cross CHANNEL attention (the paper's finding: channel attention is
more degradation-robust than spatial attention); a Restormer-style U-Net (MDTA
channel attention + GDFN, with channel-compressed attention and the paper's masked
attention variant) refines; a pixel-shuffle tail adds onto the bilinear 4x base.
Unidirectional recurrence means the harness can STREAM it frame by frame -- no
window buffering; reset() at cuts.

Released config (auto-detected from the checkpoint): num_feat 48, blocks [2,3,4,1],
heads [1,2,4], ffn expansion 2.66, squeeze 4 (attention runs on compressed channels),
masked attention on, BiasFree LayerNorm (no mean subtraction, eps 1e-5). The
checkpoint's vestigial 'attn_merge.attn.masktemp' is unused by the reference
inference (it loads strict=False); it is dropped at load here.

The SpyNet inside is BasicSR's (keys 'spynet.basic_module.N.basic_module.{0,2,..}');
its semantics (resize-up to ceil-32, avg-pool pyramid, border-warp, align-corners
flow upsampling, resize-back with scale correction) match vsr_blocks.spynet_flow
exactly, so the keys are remapped to the mmagic naming at load and the shared
compiled + gate-padded implementation is reused.

fp16 with fp32 islands where reductions span the full spatial extent: the q/k
L2-normalization over H*W tokens (a fp16 sum of ~400k squares overflows 65504 --
the same class as the ESC global-pool overflow) and the LayerNorm reductions.
Layout: MLX-native NHWC; conv weights -> (O,kH,kW,I) at load.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..compile_cache import cached as _cached
from ..vsr_blocks import compiled_spynet_flow, flow_warp, pad_spynet_gates, resize
from ..weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# Not bundled (download + convert; see weights/README.md).
_VARIANTS = {"x4": "realviformer_x4.safetensors"}
_DEFAULT_VARIANT = "x4"
_REPO = "https://github.com/Yuehan717/RealViformer"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Variant token (x4) or a path; falls back to $REALVIFORMER_WEIGHTS."""
    if spec is None or spec == "":
        spec = os.environ.get("REALVIFORMER_WEIGHTS")
    try:
        return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nRealViformer weights are not bundled. Download the released model "
            f"from the RealViformer repo ({_REPO}) and convert (see weights/README.md), "
            f"or point $REALVIFORMER_WEIGHTS / --realviformer-weights at an existing "
            f".safetensors."
        ) from None


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint. Conv weights -> NHWC; Linear weights stay
    (out,in); the vestigial attn_merge masktemp is dropped; BasicSR SpyNet keys are
    remapped to the mmagic naming so vsr_blocks.spynet_flow drives them; SpyNet
    mean/std -> NHWC broadcast shape; the SpyNet first-conv gate pad is applied."""
    w = mx.load(str(resolve_weights(path)))
    p: dict = {}
    for k, v in w.items():
        if k.endswith(".masktemp"):
            continue                                   # vestigial (reference loads strict=False)
        if k in ("spynet.mean", "spynet.std"):
            p[k] = v.reshape(1, 1, 1, 3).astype(dtype)
            continue
        if k.startswith("spynet.basic_module."):
            parts = k.split(".")                       # spynet.basic_module.L.basic_module.J.{weight,bias}
            parts[4] = f"{int(parts[4]) // 2}"         # BasicSR Sequential idx -> mmagic module idx
            k = ".".join(parts[:5]) + ".conv." + parts[5]
        if v.ndim == 4:
            a = mx.transpose(v, (0, 2, 3, 1))
        else:
            a = v
        p[k] = a.astype(dtype)
    if "shallow_extraction.0.weight" not in p:
        raise ValueError("not a RealViformer checkpoint")
    pad_spynet_gates(p)
    return p


def _config(p: dict) -> tuple:
    """(num_feat, (enc1, enc2, latent, refine) block counts) from the weights."""
    nf = int(p["shallow_extraction.0.weight"].shape[0])

    def _count(prefix: str) -> int:
        i = 0
        while f"{prefix}.{i}.norm2.body.weight" in p:
            i += 1
        return i

    return nf, (_count("encoder_level1"), _count("encoder_level2"),
                _count("latent"), _count("refinement"))


def _lrelu(x: Any, slope: float = 0.1) -> Any:
    return mx.where(x >= 0, x, x * slope)


def _gelu(x: Any) -> Any:
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _conv(x: Any, p: dict, key: str, pad: int = 0) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=1, padding=pad)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _dw3x3(x: Any, p: dict, key: str) -> Any:
    """3x3 depthwise (pad 1) as the 9-tap shift-add (bias-free in this net)."""
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


def _ln_biasfree(x: Any, p: dict, key: str) -> Any:
    """BiasFree LayerNorm over channels: x / sqrt(var + 1e-5) * w. NO mean
    subtraction (the variance is still computed about the mean); fp32 reduction."""
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = xf * mx.rsqrt(var + 1e-5)
    return (y * p[f"{key}.body.weight"].astype(mx.float32)).astype(x.dtype)


def _heads_split(t: Any, heads: int) -> Any:
    """(n,h,w,C) -> (n, heads, C/heads, h*w) channel-major tokens."""
    n, h, w, c = t.shape
    t = t.reshape(n, h * w, heads, c // heads)
    return t.transpose(0, 2, 3, 1)


def _l2norm_tokens(t: Any) -> Any:
    """L2-normalize along the token axis in fp32: the sum of ~H*W fp16 squares
    overflows 65504 at video sizes."""
    tf = t.astype(mx.float32)
    return (tf * mx.rsqrt(mx.sum(tf * tf, axis=-1, keepdims=True) + 1e-12)).astype(t.dtype)


def _linear(x: Any, p: dict, key: str) -> Any:
    y = x @ p[f"{key}.weight"].T
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _mdta(x: Any, p: dict, pre: str, masked: bool) -> Any:
    """Multi-DConv-head transposed (channel) attention, with the paper's masked
    variant: a per-channel sigmoid gate predicted from the attention map's row
    max/avg statistics."""
    n, h, w, c = x.shape
    heads = p[f"{pre}.temperature"].shape[0]
    qkv = _dw3x3(_conv(x, p, f"{pre}.qkv"), p, f"{pre}.qkv_dwconv")
    q, k, v = (qkv[..., :c], qkv[..., c:2 * c], qkv[..., 2 * c:])
    q = _l2norm_tokens(_heads_split(q, heads))
    k = _l2norm_tokens(_heads_split(k, heads))
    v = _heads_split(v, heads)
    attn = (q @ k.transpose(0, 1, 3, 2)) * p[f"{pre}.temperature"]   # (n,heads,ch,ch)
    if masked:
        mx_d = mx.max(attn, axis=-1, keepdims=True)
        av_d = mx.mean(attn, axis=-1, keepdims=True)
        m = _linear(mx.concatenate([mx_d, av_d], axis=-1), p, f"{pre}.linear1")  # (n,hd,ch,1)
        m = _linear(_gelu(m).transpose(0, 1, 3, 2), p, f"{pre}.linear2")         # (n,hd,1,ch)
        mask = mx.sigmoid(m.transpose(0, 1, 3, 2))
    attn = mx.softmax(attn, axis=-1, precise=True)
    out = attn @ v                                                    # (n,heads,ch,hw)
    if masked:
        out = out * mask
    out = out.transpose(0, 3, 1, 2).reshape(n, h, w, c)
    return _conv(out, p, f"{pre}.project_out")


def _gdfn(x: Any, p: dict, pre: str) -> Any:
    y = _dw3x3(_conv(x, p, f"{pre}.project_in"), p, f"{pre}.dwconv")
    half = y.shape[-1] // 2
    y = _gelu(y[..., :half]) * y[..., half:]
    return _conv(y, p, f"{pre}.project_out")


def _tblock(x: Any, p: dict, pre: str, masked: bool) -> Any:
    """TransformerBlock; attention runs on channel-compressed features when the
    checkpoint has a compress conv (squeeze_factor > 1)."""
    if f"{pre}.compress.weight" in p:
        z = _conv(x, p, f"{pre}.compress", pad=1)
        x = x + _conv(_mdta(_ln_biasfree(z, p, f"{pre}.norm1"), p, f"{pre}.attn", masked),
                      p, f"{pre}.expand", pad=1)
    else:
        x = x + _mdta(_ln_biasfree(x, p, f"{pre}.norm1"), p, f"{pre}.attn", masked)
    return x + _gdfn(_ln_biasfree(x, p, f"{pre}.norm2"), p, f"{pre}.ffn")


def _tblocks(x: Any, p: dict, prefix: str, count: int, masked: bool) -> Any:
    for i in range(count):
        x = _tblock(x, p, f"{prefix}.{i}", masked)
    return x


def _pixel_shuffle(x: Any, r: int) -> Any:
    n, h, w, cr = x.shape
    c = cr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _pixel_unshuffle(x: Any, r: int) -> Any:
    n, h, w, c = x.shape
    x = x.reshape(n, h // r, r, w // r, r, c)
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))
    return x.reshape(n, h // r, w // r, c * r * r)


def _attn_merge(xs: Any, yp: Any, p: dict) -> Any:
    """Cross channel attention merging the shallow features (query) with the
    warped propagated features (key/value), then a gated conv FFN on the concat."""
    xn = _ln_biasfree(xs, p, "attn_merge.norm_q")
    yn = _ln_biasfree(yp, p, "attn_merge.norm_kv")
    heads = p["attn_merge.attn.temperature"].shape[0]
    kv = _dw3x3(_conv(yn, p, "attn_merge.attn.kv"), p, "attn_merge.attn.kv_dwconv")
    c = xs.shape[-1]
    k, v = kv[..., :c], kv[..., c:]
    q = _conv(xn, p, "attn_merge.attn.q", pad=1)
    qh = _l2norm_tokens(_heads_split(q, heads))
    kh = _l2norm_tokens(_heads_split(k, heads))
    vh = _heads_split(v, heads)
    attn = mx.softmax((qh @ kh.transpose(0, 1, 3, 2)) * p["attn_merge.attn.temperature"],
                      axis=-1, precise=True)
    o = (attn @ vh).transpose(0, 3, 1, 2).reshape(xs.shape)
    o = _conv(o, p, "attn_merge.attn.project_out")
    o = _ln_biasfree(o, p, "attn_merge.norm_out")
    y = _conv(mx.concatenate([xs, o], axis=-1), p, "attn_merge.ffn.0")
    y = _gelu(_dw3x3(y, p, "attn_merge.ffn.1"))
    return _conv(y, p, "attn_merge.ffn.3")


def _unet(x: Any, p: dict, blocks: tuple, masked: bool = True) -> tuple:
    """The Restormer U-Net body -> (upsample-tail input, next feat_prop)."""
    e1 = _tblocks(x, p, "encoder_level1", blocks[0], masked)
    d12 = _pixel_unshuffle(_conv(e1, p, "down1_2.body.0", pad=1), 2)
    e2 = _tblocks(d12, p, "encoder_level2", blocks[1], masked)
    d23 = _pixel_unshuffle(_conv(e2, p, "down2_3.body.0", pad=1), 2)
    lat = _tblocks(d23, p, "latent", blocks[2], masked)
    u32 = _pixel_shuffle(_conv(lat, p, "up3_2.body.0", pad=1), 2)
    d2 = _conv(mx.concatenate([u32, e2], axis=-1), p, "reduce_chan_level2")
    d2 = _tblocks(d2, p, "decoder_level2", blocks[1], masked)
    u21 = _pixel_shuffle(_conv(d2, p, "up2_1.body.0", pad=1), 2)
    d1 = _tblocks(mx.concatenate([u21, e1], axis=-1), p, "decoder_level1", blocks[0], masked)
    out = _tblocks(d1, p, "refinement", blocks[3], masked)
    return out, _conv(out, p, "compress")


def _reconstruct(out: Any, x_i: Any, p: dict) -> Any:
    y = _lrelu(_pixel_shuffle(_conv(out, p, "upconv1", pad=1), 2))
    y = _lrelu(_pixel_shuffle(_conv(y, p, "upconv2", pad=1), 2))
    y = _lrelu(_conv(y, p, "conv_hr", pad=1))
    y = _conv(y, p, "conv_last", pad=1)
    base = resize(x_i, x_i.shape[1] * 4, x_i.shape[2] * 4, False)
    return mx.clip(y + base, 0.0, 1.0)


def step_first(x_i: Any, p: dict, cfg: tuple) -> tuple:
    """First frame of a clip: zero prior features, no flow. -> (sr, feat_prop)."""
    nf, blocks = cfg
    dt = p["shallow_extraction.0.weight"].dtype
    x_i = x_i.astype(dt)
    shallow = _tblock(_conv(x_i, p, "shallow_extraction.0", pad=1), p,
                      "shallow_extraction.1", masked=False)
    prop = _attn_merge(shallow, mx.zeros_like(shallow), p)
    out, feat_prop = _unet(prop, p, blocks)
    return _reconstruct(out, x_i, p), feat_prop


def step_next(x_i: Any, x_prev: Any, feat_prop: Any, p: dict, cfg: tuple) -> tuple:
    """Subsequent frames: flow-warp the propagated features, merge, refine."""
    nf, blocks = cfg
    dt = p["shallow_extraction.0.weight"].dtype
    x_i = x_i.astype(dt)
    shallow = _tblock(_conv(x_i, p, "shallow_extraction.0", pad=1), p,
                      "shallow_extraction.1", masked=False)
    flow = compiled_spynet_flow(p, x_i, x_prev.astype(dt))
    warped = flow_warp(feat_prop, flow, "zeros")
    prop = _attn_merge(shallow, warped, p)
    out, feat_prop = _unet(prop, p, blocks)
    return _reconstruct(out, x_i, p), feat_prop


_COMPILE_CACHE: dict = {}


def make_steps(p: dict, cfg: tuple | None = None, compile: bool = True) -> tuple:
    """(first_step, next_step) compiled once per checkpoint. compiled_spynet_flow
    nests its own compile, so the flow stays OUTSIDE the compiled next-step: the
    step closes over p and takes (x_i, warped-ready inputs)."""
    if cfg is None:
        cfg = _config(p)

    def first(x_i):
        return step_first(x_i, p, cfg)

    def merge_and_refine(x_i, warped):
        dt = p["shallow_extraction.0.weight"].dtype
        x_i = x_i.astype(dt)
        shallow = _tblock(_conv(x_i, p, "shallow_extraction.0", pad=1), p,
                          "shallow_extraction.1", masked=False)
        prop = _attn_merge(shallow, warped, p)
        out, feat_prop = _unet(prop, p, cfg[1])
        return _reconstruct(out, x_i, p), feat_prop

    if not compile:
        return first, merge_and_refine
    f = _cached(_COMPILE_CACHE, (id(p), cfg, "first"), lambda: mx.compile(first))
    n = _cached(_COMPILE_CACHE, (id(p), cfg, "next"), lambda: mx.compile(merge_and_refine))
    return f, n


if __name__ == "__main__":
    p = load_params()
    cfg = _config(p)
    print(f"loaded RealViformer: nf={cfg[0]} blocks={cfg[1]}")
    mx.random.seed(0)
    a = mx.clip(mx.random.uniform(shape=(1, 64, 96, 3)), 0, 1)
    b = mx.clip(mx.random.uniform(shape=(1, 64, 96, 3)), 0, 1)
    mx.eval(a, b)
    s0, fp = step_first(a, p, cfg)
    s1, fp = step_next(b, a, fp, p, cfg)
    mx.eval(s0, s1)
    print(f"{tuple(a.shape)} -> {tuple(s1.shape)}, finite={bool(mx.all(mx.isfinite(s1)))}")
