"""Shared BasicVSR backbone building blocks (MLX, NHWC).

Generic primitives used by BOTH the BasicVSR++ and RealBasicVSR ports: conv /
activation helpers, bilinear sampling + flow-warp + resize, the SPyNet optical-flow
pyramid, residual blocks, and pixel-shuffle. They live here so neither upscaler
depends on the other -- RealBasicVSR previously reached into basicvsrpp/net.py for
these, which coupled two sibling architectures.

Convention: NHWC throughout. Conv weights are pre-transposed to MLX's (O,kH,kW,I)
by each net's load_params; flow is (N,H,W,2) = (x-offset, y-offset), matching
flow_warp.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from .compile_cache import cached as _cached


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
def pad_spynet_gates(p: dict) -> None:
    """Zero-pad each SPyNet basic module's FIRST conv from 8 to 16 input channels
    (in place, same key). C=8 fails MLX's implicit-GEMM gate (C<=4 or C%16==0,
    mlx conv.cpp) so the finest-level 7x7 conv ran on the ~2.5x-slower general
    kernel. spynet_flow appends matching zero channels to the module input, so the
    math is exact (zero columns x zero channels). Call after load_params; a no-op
    when the keys are absent or already padded."""
    for lvl in range(6):
        k = f"spynet.basic_module.{lvl}.basic_module.0.conv.weight"
        if k in p and p[k].shape[-1] == 8:
            w = p[k]
            p[k] = mx.concatenate(
                [w, mx.zeros((*w.shape[:3], 8), dtype=w.dtype)], axis=-1)


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
    # Gate-padded first conv (pad_spynet_gates): append zero channels to match.
    inp_pad = p["spynet.basic_module.0.basic_module.0.conv.weight"].shape[-1] - 8
    for lvl in range(6):
        flow_up = flow if lvl == 0 else resize(flow, flow.shape[1] * 2, flow.shape[2] * 2, True) * 2.0
        warped = flow_warp(supps[lvl], flow_up, "border")
        parts = [refs[lvl], warped, flow_up]                          # (N,h,w,8)
        if inp_pad:
            parts.append(mx.zeros((*refs[lvl].shape[:3], inp_pad), dtype=ref.dtype))
        inp = mx.concatenate(parts, axis=-1)
        flow = flow_up + _spynet_basic_module(inp, p, lvl)
    flow = resize(flow, h, w, False)
    return mx.stack([flow[..., 0] * (w / w_up), flow[..., 1] * (h / h_up)], axis=-1)


_SPYNET_COMPILE_CACHE: dict = {}


def compiled_spynet_flow(p: dict, ref: Any, supp: Any) -> Any:
    """spynet_flow, mx.compiled + cached per checkpoint (~1.1x).

    The reference runs SPyNet in fp32; this port runs it in fp16, and compiling the
    fp16 path reorders ops so the flow shifts ~0.02 vs op-by-op (fp32 reorders <3e-4).
    That moves the final SR by <=0.012 max / ~6e-4 mean on [0,1] -- fp16 noise on a net
    that is already an fp16 approximation of the fp32 reference. Keyed by id(p)."""
    fn = _cached(_SPYNET_COMPILE_CACHE, id(p), lambda: mx.compile(lambda r, s: spynet_flow(p, r, s)))
    return fn(ref, supp)


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


_RESBLOCKS_COMPILE_CACHE: dict = {}


def compiled_resblocks(x: Any, p: dict, prefix: str) -> Any:
    """_resblocks_with_input, mx.compiled + cached per (checkpoint, prefix).

    The resblock stack fuses for ~1.05-1.07x over the op-by-op path, pure and
    byte-identical (profiled, MLX cache capped). For the recurrent VSR loops, where
    the stack runs many times per frame. Keyed by (id(p), prefix); the cache entry
    closes over p so its id stays stable. Do NOT call this from inside another
    mx.compile'd step (it would nest compiles) -- call _resblocks_with_input there.
    """
    fn = _cached(_RESBLOCKS_COMPILE_CACHE, (id(p), prefix),
                 lambda: mx.compile(lambda x: _resblocks_with_input(x, p, prefix)))
    return fn(x)


def _pixel_shuffle(x: Any, r: int) -> Any:
    """(N,H,W,C*r^2) -> (N,H*r,W*r,C), torch PixelShuffle channel order."""
    n, h, w, c4 = x.shape
    c = c4 // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _pixelshuffle_pack(x: Any, p: dict, prefix: str, r: int = 2) -> Any:
    return _pixel_shuffle(conv(x, p, f"{prefix}.upsample_conv"), r)


def _compute_flows(frames: list, p: dict) -> tuple:
    """flows_forward[i] = flow(i+1 -> i); flows_backward[i] = flow(i -> i+1).

    Each flow is materialized as computed: SPyNet upsizes to a multiple of 32 and
    builds a 6-level pyramid, so holding all 2*(T-1) of them as one lazy graph
    spikes memory; per-flow eval keeps only the small (H,W,2) results alive."""
    fb, ff = [], []
    for i in range(len(frames) - 1):
        b = compiled_spynet_flow(p, frames[i], frames[i + 1])
        f = compiled_spynet_flow(p, frames[i + 1], frames[i])
        mx.eval(b, f)
        fb.append(b)
        ff.append(f)
    return ff, fb
