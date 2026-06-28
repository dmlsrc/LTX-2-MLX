"""MLX BasicVSR++ x4 net (port of OpenMMLab mmagic basicvsr_plusplus_net).

Built incrementally. This stage has the weight loader, the NHWC helpers
(bilinear sample / flow_warp / resize / avg-pool), and SPyNet (the optical-flow
pyramid). The propagation branches + recurrent forward build on top.

Convention: MLX-native NHWC throughout. Conv weights are transposed to MLX's
(O,kH,kW,I) at load; the deformable-conv weight stays torch NCHW (O,I,kH,kW) for
deform_conv2d. Flow is (N,H,W,2) = (x-offset, y-offset), matching flow_warp.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

try:
    from .deform_conv import deform_conv2d
except ImportError:   # running net.py directly as a script
    from deform_conv import deform_conv2d

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Bundled 4x-SR checkpoints (all is_low_res_input=True). reds4/vimeo90k_bi are
# c64n7 (7.3M); ntire_vsr is c128n25 (44M, NTIRE'21). Block counts auto-detect.
_VARIANTS = {
    "reds4": "basicvsrpp_reds4.safetensors",
    "vimeo90k_bi": "basicvsrpp_vimeo90k_bi.safetensors",
    "vimeo90k_bd": "basicvsrpp_vimeo90k_bd.safetensors",
    "ntire_vsr": "basicvsrpp_ntire_vsr.safetensors",
}


def default_weights_path(variant: str = "reds4") -> Path:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown basicvsrpp variant {variant!r}; choose from {list(_VARIANTS)}")
    return _WEIGHTS_DIR / _VARIANTS[variant]


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: conv weights -> NHWC, DCN weight kept NCHW,
    SPyNet mean/std -> NHWC, step_counter dropped, all cast to `dtype`.

    Default fp16 halves activation memory and is ~1.5x faster; the deformable
    conv still runs its im2col/GEMM in fp32 internally for safety. Pass
    dtype=mx.float32 to match the bit-exact validation reference.

    A checkpoint over GitHub's 100MB file limit (ntire_vsr) ships split into
    `<stem>.shardNN<suffix>` pieces; if the single file is absent, the shards are
    loaded and merged."""
    src = Path(path or default_weights_path())
    if src.exists():
        w = mx.load(str(src))
    else:
        shards = sorted(src.parent.glob(f"{src.stem}.shard*{src.suffix}"))
        if not shards:
            raise FileNotFoundError(f"{src} (no file, and no {src.stem}.shard*{src.suffix} shards)")
        w = {}
        for s in shards:
            w.update(mx.load(str(s)))
    p: dict = {}
    for k, v in w.items():
        if k == "step_counter":
            continue
        if k in ("spynet.mean", "spynet.std"):
            a = v.reshape(1, 1, 1, 3)
        elif v.ndim == 4 and not (
            k.startswith("deform_align.") and k.endswith(".weight") and "conv_offset" not in k
        ):
            a = mx.transpose(v, (0, 2, 3, 1))   # (O,I,kH,kW) torch -> (O,kH,kW,I) MLX
        else:
            a = v
        p[k] = a.astype(dtype)
    return p


# ---- NHWC building blocks --------------------------------------------------
def relu(x: Any) -> Any:
    return mx.maximum(x, 0)


def lrelu(x: Any, slope: float = 0.1) -> Any:
    return mx.where(x >= 0, x, x * slope)


def conv(x: Any, p: dict, key: str, stride: int = 1, pad: int = 1, groups: int = 1) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=stride, padding=pad, groups=groups)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _bilinear(x: Any, sy: Any, sx: Any, pad: str = "border") -> Any:
    """Sample x (N,H,W,C) at (sy,sx) (each (N,oH,oW)) -> (N,oH,oW,C). 'border'
    clamps out-of-range to the edge; 'zeros' returns 0 outside."""
    n, h, w, c = x.shape
    oh, ow = sy.shape[1], sy.shape[2]
    y0 = mx.floor(sy)
    x0 = mx.floor(sx)
    ly = (sy - y0)[..., None]
    lx = (sx - x0)[..., None]
    y0i = y0.astype(mx.int32)
    x0i = x0.astype(mx.int32)
    flat = x.reshape(n, h * w, c)

    def g(yi: Any, xi: Any) -> Any:
        idx = (mx.clip(yi, 0, h - 1) * w + mx.clip(xi, 0, w - 1)).reshape(n, oh * ow, 1)
        v = mx.take_along_axis(flat, mx.broadcast_to(idx, (n, oh * ow, c)), axis=1).reshape(n, oh, ow, c)
        if pad == "zeros":
            valid = ((yi >= 0) & (yi <= h - 1) & (xi >= 0) & (xi <= w - 1)).astype(x.dtype)
            v = v * valid[..., None]
        return v

    v00 = g(y0i, x0i)
    v01 = g(y0i, x0i + 1)
    v10 = g(y0i + 1, x0i)
    v11 = g(y0i + 1, x0i + 1)
    out = (1 - ly) * (1 - lx) * v00 + (1 - ly) * lx * v01 + ly * (1 - lx) * v10 + ly * lx * v11
    return out.astype(x.dtype)   # fp32 grid weights would otherwise upcast features


def flow_warp(x: Any, flow: Any, pad: str = "zeros") -> Any:
    """Warp x (N,H,W,C) by flow (N,H,W,2): out[p] = x[p + flow[p]]."""
    n, h, w, _ = x.shape
    gy, gx = mx.meshgrid(mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32), indexing="ij")
    sx = gx[None] + flow[..., 0]
    sy = gy[None] + flow[..., 1]
    return _bilinear(x, sy, sx, pad)


def resize(x: Any, oh: int, ow: int, align_corners: bool) -> Any:
    """Bilinear resize NHWC x to (oh, ow) (edge-clamped), matching torch's
    align_corners True/False coordinate maps."""
    n, h, w, _ = x.shape
    if align_corners:
        ry = (h - 1) / (oh - 1) if oh > 1 else 0.0
        rx = (w - 1) / (ow - 1) if ow > 1 else 0.0
        sy1 = mx.arange(oh, dtype=mx.float32) * ry
        sx1 = mx.arange(ow, dtype=mx.float32) * rx
    else:
        sy1 = (mx.arange(oh, dtype=mx.float32) + 0.5) * (h / oh) - 0.5
        sx1 = (mx.arange(ow, dtype=mx.float32) + 0.5) * (w / ow) - 0.5
    sy = mx.broadcast_to(sy1.reshape(1, oh, 1), (n, oh, ow))
    sx = mx.broadcast_to(sx1.reshape(1, 1, ow), (n, oh, ow))
    return _bilinear(x, sy, sx, "border")


def _avgpool2(x: Any) -> Any:
    """2x2 average pool, stride 2 (input dims even)."""
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))


# ---- SPyNet ----------------------------------------------------------------
def _spynet_basic_module(x: Any, p: dict, lvl: int) -> Any:
    base = f"spynet.basic_module.{lvl}.basic_module"
    for j in (0, 1, 2, 3):
        x = relu(conv(x, p, f"{base}.{j}.conv", pad=3))
    return conv(x, p, f"{base}.4.conv", pad=3)


def spynet_flow(p: dict, ref: Any, supp: Any) -> Any:
    """Optical flow from ref to supp; both (N,H,W,3) in [0,1]. -> (N,H,W,2)."""
    n, h, w, _ = ref.shape
    w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
    h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
    ref = resize(ref, h_up, w_up, False)
    supp = resize(supp, h_up, w_up, False)
    mean, std = p["spynet.mean"], p["spynet.std"]
    refs = [(ref - mean) / std]
    supps = [(supp - mean) / std]
    for _ in range(5):
        refs.append(_avgpool2(refs[-1]))
        supps.append(_avgpool2(supps[-1]))
    refs = refs[::-1]
    supps = supps[::-1]
    flow = mx.zeros((n, h_up // 32, w_up // 32, 2), dtype=ref.dtype)   # keep flow in the feature dtype
    for lvl in range(6):
        flow_up = flow if lvl == 0 else resize(flow, flow.shape[1] * 2, flow.shape[2] * 2, True) * 2.0
        warped = flow_warp(supps[lvl], flow_up, "border")
        inp = mx.concatenate([refs[lvl], warped, flow_up], axis=-1)   # (N,h,w,8)
        flow = flow_up + _spynet_basic_module(inp, p, lvl)
    flow = resize(flow, h, w, False)
    return mx.stack([flow[..., 0] * (w / w_up), flow[..., 1] * (h / h_up)], axis=-1)


# ---- residual blocks + pixel-shuffle ---------------------------------------
def _resblock(x: Any, p: dict, key: str) -> Any:
    """ResidualBlockNoBN: x + conv2(relu(conv1(x))), res_scale 1."""
    return x + conv(relu(conv(x, p, f"{key}.conv1")), p, f"{key}.conv2")


def _resblocks_with_input(x: Any, p: dict, prefix: str) -> Any:
    # Block count is read from the checkpoint, so c64n7 (7) and c128n25 (25) and
    # the 15-block restoration variants all load without a hardcoded count.
    x = lrelu(conv(x, p, f"{prefix}.main.0"))
    i = 0
    while f"{prefix}.main.2.{i}.conv1.weight" in p:
        x = _resblock(x, p, f"{prefix}.main.2.{i}")
        i += 1
    return x


def _pixel_shuffle(x: Any, r: int) -> Any:
    """(N,H,W,C*r^2) -> (N,H*r,W*r,C), torch PixelShuffle channel order."""
    n, h, w, c4 = x.shape
    c = c4 // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _pixelshuffle_pack(x: Any, p: dict, prefix: str, r: int = 2) -> Any:
    return _pixel_shuffle(conv(x, p, f"{prefix}.upsample_conv"), r)


# ---- second-order deformable alignment -------------------------------------
def _flow_yx_tiled(flow: Any, reps: int) -> Any:
    """flow (N,H,W,2)=[x,y] -> [y,x] tiled `reps` times along the channel axis."""
    return mx.tile(mx.concatenate([flow[..., 1:2], flow[..., 0:1]], axis=-1), (1, 1, 1, reps))


def _deform_align(feat_cat: Any, cond: Any, flow1: Any, flow2: Any, p: dict,
                  key: str, max_res: float = 10.0) -> Any:
    """SecondOrderDeformableAlignment: predict offsets/mask from cond+flows, add
    the flows (the deformable offset is relative to the optical flow), then a
    modulated deform conv on feat_cat (NHWC <-> NCHW only at the DCN call)."""
    extra = mx.concatenate([cond, flow1, flow2], axis=-1)
    o = lrelu(conv(extra, p, f"{key}.conv_offset.0"))
    o = lrelu(conv(o, p, f"{key}.conv_offset.2"))
    o = lrelu(conv(o, p, f"{key}.conv_offset.4"))
    o = conv(o, p, f"{key}.conv_offset.6")               # (N,H,W,27*dg)
    o1, o2, mask = mx.split(o, 3, axis=-1)               # dg*K*K each
    off = max_res * mx.tanh(mx.concatenate([o1, o2], axis=-1))
    off1, off2 = mx.split(off, 2, axis=-1)
    off1 = off1 + _flow_yx_tiled(flow1, off1.shape[-1] // 2)
    off2 = off2 + _flow_yx_tiled(flow2, off2.shape[-1] // 2)
    offset = mx.concatenate([off1, off2], axis=-1)       # (N,H,W,dg*2*K*K)
    mask = mx.sigmoid(mask)
    dg = o.shape[-1] // 27
    out = deform_conv2d(
        mx.transpose(feat_cat, (0, 3, 1, 2)), mx.transpose(offset, (0, 3, 1, 2)),
        p[f"{key}.weight"], p.get(f"{key}.bias"), mx.transpose(mask, (0, 3, 1, 2)),
        stride=1, padding=1, dilation=1, deform_groups=dg,
    )
    return mx.transpose(out, (0, 2, 3, 1)).astype(feat_cat.dtype)   # DCN runs fp32 inside


# ---- recurrent forward -----------------------------------------------------
def _compute_flows(frames: list, p: dict) -> tuple:
    """flows_forward[i] = flow(i+1 -> i); flows_backward[i] = flow(i -> i+1).

    Each flow is materialized as computed: SPyNet upsizes to a multiple of 32 and
    builds a 6-level pyramid, so holding all 2*(T-1) of them as one lazy graph
    spikes memory; per-flow eval keeps only the small (H,W,2) results alive."""
    fb, ff = [], []
    for i in range(len(frames) - 1):
        b = spynet_flow(p, frames[i], frames[i + 1])
        f = spynet_flow(p, frames[i + 1], frames[i])
        mx.eval(b, f)
        fb.append(b)
        ff.append(f)
    return ff, fb


def _propagate(feats: dict, flows: list, module: str, p: dict) -> dict:
    nf = len(feats["spatial"])
    frame_idx = list(range(nf))
    flow_idx = list(range(-1, nf - 1))
    if "backward" in module:
        frame_idx = frame_idx[::-1]
        flow_idx = frame_idx
    n, h, w, mid = feats["spatial"][0].shape
    feat_prop = mx.zeros((n, h, w, mid), dtype=feats["spatial"][0].dtype)
    out: list = []
    for i, idx in enumerate(frame_idx):
        feat_current = feats["spatial"][idx]
        if i > 0:
            flow_n1 = flows[flow_idx[i]]
            cond_n1 = flow_warp(feat_prop, flow_n1)
            feat_n2 = mx.zeros_like(feat_prop)
            flow_n2 = mx.zeros_like(flow_n1)
            cond_n2 = mx.zeros_like(cond_n1)
            if i > 1:
                feat_n2 = out[-2]
                flow_n2 = flow_n1 + flow_warp(flows[flow_idx[i - 1]], flow_n1)
                cond_n2 = flow_warp(feat_n2, flow_n2)
            cond = mx.concatenate([cond_n1, feat_current, cond_n2], axis=-1)
            feat_prop = _deform_align(mx.concatenate([feat_prop, feat_n2], axis=-1),
                                      cond, flow_n1, flow_n2, p, f"deform_align.{module}")
        feat = [feat_current] + [feats[k][idx] for k in feats if k not in ("spatial", module)] + [feat_prop]
        feat_prop = feat_prop + _resblocks_with_input(mx.concatenate(feat, axis=-1), p, f"backbone.{module}")
        # Materialize each step so the recurrent graph (and the large transient
        # DCN im2col columns) frees per frame instead of accumulating the whole
        # clip's forward into one lazy graph - that peaks memory catastrophically.
        mx.eval(feat_prop)
        out.append(feat_prop)
    if "backward" in module:
        out = out[::-1]
    feats[module] = out
    return feats


def _upsample(frames: list, feats: dict, p: dict) -> list:
    outs = []
    for i in range(len(feats["spatial"])):
        hr = [feats["spatial"][i]] + [feats[k][i] for k in feats if k != "spatial"]
        hr = _resblocks_with_input(mx.concatenate(hr, axis=-1), p, "reconstruction")
        hr = lrelu(_pixelshuffle_pack(hr, p, "upsample1"))
        hr = lrelu(_pixelshuffle_pack(hr, p, "upsample2"))
        hr = lrelu(conv(hr, p, "conv_hr"))
        hr = conv(hr, p, "conv_last")
        _, fh, fw, _ = frames[i].shape
        out_frame = hr + resize(frames[i], fh * 4, fw * 4, False)
        mx.eval(out_frame)   # free each frame's upsample graph before the next
        outs.append(out_frame)
    return outs


def upscale(frames: list, p: dict) -> list:
    """Upscale an LR clip 4x. frames: list of (N,H,W,3) f32 [0,1]; out: same len,
    each (N,4H,4W,3). Bidirectional + second-order, so the whole clip is needed."""
    dt = p["conv_last.weight"].dtype
    frames = [f.astype(dt) for f in frames]
    spatial = []
    for f in frames:
        s = _resblocks_with_input(f, p, "feat_extract")
        mx.eval(s)                       # materialize per frame, not all at once
        spatial.append(s)
    feats: dict = {"spatial": spatial}
    ff, fb = _compute_flows(frames, p)   # evals each flow internally
    for it in (1, 2):
        for direction in ("backward", "forward"):
            mod = f"{direction}_{it}"
            feats = _propagate(feats, fb if direction == "backward" else ff, mod, p)
            mx.eval(*feats[mod])
    return _upsample(frames, feats, p)


if __name__ == "__main__":
    p = load_params()
    mx.random.seed(0)
    frames = [mx.clip(mx.random.uniform(shape=(1, 48, 64, 3)), 0, 1) for _ in range(5)]
    mx.eval(*frames)
    outs = upscale(frames, p)
    mx.eval(*outs)
    print(f"upscale: {len(outs)} frames, 48x64 -> {outs[0].shape[1]}x{outs[0].shape[2]}, "
          f"center range [{float(mx.min(outs[2])):.3f}, {float(mx.max(outs[2])):.3f}]")
