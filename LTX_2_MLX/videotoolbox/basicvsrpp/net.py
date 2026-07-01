"""MLX BasicVSR++ x4 net (port of OpenMMLab mmagic basicvsr_plusplus_net).

The shared BasicVSR backbone (conv/activation helpers, bilinear sample / flow_warp
/ resize, SPyNet, residual blocks, pixel-shuffle) lives in ../vsr_blocks; this
module adds the BasicVSR++-specific pieces: the weight loader, the second-order
deformable alignment, and the bidirectional recurrent forward.

Convention: MLX-native NHWC throughout. Conv weights are transposed to MLX's
(O,kH,kW,I) at load; the deformable-conv weight stays torch NCHW (O,I,kH,kW) for
deform_conv2d. Flow is (N,H,W,2) = (x-offset, y-offset), matching flow_warp.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from ..deform_conv import deform_conv2d
from ..vsr_blocks import (
    _compute_flows,
    _pixelshuffle_pack,
    _resblocks_with_input,
    compiled_resblocks,
    conv,
    flow_warp,
    lrelu,
    resize,
)
from ..weights import resolve_weights as _resolve_weights

# Per-checkpoint compiled reconstruction/upsample tail (keyed by id(p)).
_UPSAMPLE_COMPILE_CACHE: dict = {}


def _compiled_upsample(p: dict):
    """Compiled reconstruction resblocks + pixel-shuffle upsample tail -> HR residual.
    The cheap base resize + clip stay in the loop. Pure, byte-identical (profiled)."""
    fn = _UPSAMPLE_COMPILE_CACHE.get(id(p))
    if fn is None:
        def step(hr):
            hr = _resblocks_with_input(hr, p, "reconstruction")
            hr = lrelu(_pixelshuffle_pack(hr, p, "upsample1"))
            hr = lrelu(_pixelshuffle_pack(hr, p, "upsample2"))
            hr = lrelu(conv(hr, p, "conv_hr"))
            return conv(hr, p, "conv_last")
        fn = mx.compile(step)
        _UPSAMPLE_COMPILE_CACHE[id(p)] = fn
    return fn

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


def resolve_weights(spec: Any = None) -> Path:
    """Bundled variant token (reds4/vimeo90k_bi/vimeo90k_bd/ntire_vsr) or a path."""
    return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, "reds4")


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: conv weights -> NHWC, DCN weight kept NCHW,
    SPyNet mean/std -> NHWC, step_counter dropped, all cast to `dtype`.

    Default fp16 halves activation memory and is ~1.5x faster; the deformable
    conv still runs its im2col/GEMM in fp32 internally for safety. Pass
    dtype=mx.float32 to match the bit-exact validation reference."""
    src = Path(path or default_weights_path())
    w = mx.load(str(src))
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


_DEFORM_COMPILE_CACHE: dict = {}


def _compiled_deform_align(p: dict, key: str):
    """_deform_align (offset convs + the deform_conv kernel), compiled + cached per
    (checkpoint, module). ~1.02x byte-identical: the custom kernel is one big dispatch
    with nothing to fuse, but the offset conv stack around it fuses. Keyed by (id(p), key)."""
    fn = _DEFORM_COMPILE_CACHE.get((id(p), key))
    if fn is None:
        fn = mx.compile(lambda fc, c, f1, f2: _deform_align(fc, c, f1, f2, p, key))
        _DEFORM_COMPILE_CACHE[(id(p), key)] = fn
    return fn


# ---- recurrent forward -----------------------------------------------------
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
            feat_prop = _compiled_deform_align(p, f"deform_align.{module}")(
                mx.concatenate([feat_prop, feat_n2], axis=-1), cond, flow_n1, flow_n2)
        feat = [feat_current] + [feats[k][idx] for k in feats if k not in ("spatial", module)] + [feat_prop]
        feat_prop = feat_prop + compiled_resblocks(mx.concatenate(feat, axis=-1), p, f"backbone.{module}")
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
        residual = _compiled_upsample(p)(mx.concatenate(hr, axis=-1))
        _, fh, fw, _ = frames[i].shape
        # Clip the terminal SR to [0,1]: the residual overshoots slightly at edges
        # (ringing) and this frame goes straight to the encoder. Not fed back into
        # the recurrence, so clipping here is safe.
        out_frame = mx.clip(residual + resize(frames[i], fh * 4, fw * 4, False), 0.0, 1.0)
        mx.eval(out_frame)   # free each frame's upsample graph before the next
        outs.append(out_frame)
    return outs


def upscale(frames: list, p: dict) -> list:
    """Upscale an LR clip 4x. frames: list of (N,H,W,3) f32 [0,1]; out: same len,
    each (N,4H,4W,3). Bidirectional + second-order, so the whole clip is needed."""
    dt = p["conv_last.weight"].dtype
    # Clip to [0,1] - the model trained on uint8-derived [0,1] LR, but the
    # RGBAHalf decode can overshoot (~[-0.07, 1.04]); keep input in-distribution.
    frames = [mx.clip(f, 0.0, 1.0).astype(dt) for f in frames]
    spatial = []
    for f in frames:
        s = compiled_resblocks(f, p, "feat_extract")
        mx.eval(s)                       # materialize per frame, not all at once
        spatial.append(s)
    feats: dict = {"spatial": spatial}
    ff, fb = _compute_flows(frames, p)   # evals each flow internally
    for it in (1, 2):
        for direction in ("backward", "forward"):
            mod = f"{direction}_{it}"
            feats = _propagate(feats, fb if direction == "backward" else ff, mod, p)
            # _propagate already mx.eval's each step internally (see net.py:176), so every
            # element of feats[mod] is materialized here -- no extra sync barrier needed.
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
