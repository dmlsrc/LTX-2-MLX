"""MLX Real-ESRGAN / ESRGAN per-frame super-resolution: RRDBNet + SRVGGNetCompact.

A feedforward conv SR network: no optical flow, no recurrence, so it has none of
the flow-propagation ghosting of the BasicVSR family (each frame is independent).
Operates on NHWC arrays in [0,1] RGB.

Covers every basicsr-format RRDBNet checkpoint -- RealESRGAN_x4plus (general
real-world, scale 4), RealESRNet_x4plus (MSE-trained, softer/no GAN
hallucination), RealESRGAN_x2plus (scale 2), the official ESRGAN_x4 -- and
normalizes the old-ESRGAN key names (RRDB_trunk/trunk_conv/upconv/HRconv) used by
BSRGAN and the original ESRGAN release. Scale and block count auto-detect from
the checkpoint, so all variants load with no code change. The tiny
SRVGGNetCompact checkpoints (realesr-general-x4v3, ~12x smaller and far faster
than RRDBNet) are also supported and auto-detected by the absence of conv_first.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from ..compile_cache import cached as _cached
from ..weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_IN_CH_TO_SCALE = {3: 4, 12: 2, 48: 1}   # conv_first in-channels -> output scale

# Checkpoints by short token; only 'general' (+ its wdn companion) is bundled -- the rest
# are download + convert (see weights/README.md). Scale and block count auto-detect, so
# every variant loads with no code change (x2plus is 2x, the others 4x).
_VARIANTS = {
    "general":    "realesr_general_x4v3.safetensors",       # SRVGG, fast, gentle (default)
    "x4plus":     "realesrgan_x4plus.safetensors",          # RRDBNet, GAN, crisp
    "realesrnet": "realesrnet_x4plus.safetensors",          # RRDBNet, MSE, faithful
    "bsrgan":     "bsrgan_x4.safetensors",                  # RRDBNet, BSRGAN degradation
    "bsrnet":     "bsrnet_x4.safetensors",                  # RRDBNet, BSRNet (no GAN)
    "x2plus":     "realesrgan_x2plus.safetensors",          # RRDBNet, 2x
    "anime":      "realesrgan_x4plus_anime_6b.safetensors",  # RRDBNet, 6 blocks, anime
    "animevideo": "realesr_animevideov3.safetensors",       # SRVGG, anime video
    "esrgan":     "esrgan_x4.safetensors",                  # original ESRGAN x4 (DF2K+OST)
}
_DEFAULT_VARIANT = "general"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Bundled variant token (general/x4plus/realesrnet/bsrgan/bsrnet) or a path."""
    return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)


def wdn_path_for(weights: str | Path | None) -> Path:
    """Locate the 'wdn' companion beside a realesr-general-x4v3 checkpoint.

    The denoise dial (dni) blends general + wdn; only realesr-general-x4v3 ships a
    wdn variant. Raises if the sibling file is absent.
    """
    src = Path(weights or default_weights_path())
    name = src.name.replace("general_x4v3", "general_wdn_x4v3")
    wdn = src.with_name(name)
    if wdn == src or not wdn.exists():
        raise FileNotFoundError(
            f"--realesrgan-denoise < 1 needs the wdn companion beside "
            f"{src.name} (expected {name}); only realesr-general-x4v3 has one")
    return wdn


def _canonical_key(k: str) -> str:
    """Map old-ESRGAN (BSRGAN / original ESRGAN) keys to basicsr names."""
    if "RRDB_trunk." in k:
        k = k.replace("RRDB_trunk.", "body.").replace(".RDB", ".rdb")
    k = k.replace("trunk_conv.", "conv_body.")
    k = k.replace("upconv1.", "conv_up1.").replace("upconv2.", "conv_up2.")
    k = k.replace("HRconv.", "conv_hr.")
    return k


def load_params(path: str | Path | None = None, dtype: Any = mx.float16,
                wdn_path: str | Path | None = None,
                denoise_strength: float = 1.0) -> dict:
    """Load RRDBNet/SRVGG weights, normalize keys, and transpose conv kernels.

    For realesr-general-x4v3, ``wdn_path`` + ``denoise_strength`` < 1 applies the
    Real-ESRGAN dni blend ``s*general + (1-s)*wdn``. Per upstream, higher
    denoise_strength = stronger denoise (smoother); lower keeps more of the
    sharper/noisier wdn model (more real-world texture/grain). 1.0 = pure general.
    """
    src = Path(path or default_weights_path())
    w = mx.load(str(src))
    s = float(denoise_strength)
    if wdn_path is not None and s < 1.0:
        wdn = mx.load(str(wdn_path))
        if set(wdn) != set(w):
            raise ValueError(f"wdn weights {wdn_path} do not match {src}")
        w = {k: w[k] * s + wdn[k] * (1.0 - s) for k in w}
    p: dict = {}
    for k, v in w.items():
        k = _canonical_key(k)
        if v.ndim == 4:                      # (O,I,kH,kW) -> (O,kH,kW,I)
            v = mx.transpose(v, (0, 2, 3, 1))
        p[k] = v.astype(dtype)
    if "conv_first.weight" not in p and "body.0.weight" not in p:
        raise ValueError(f"{src} is not an RRDBNet / SRVGGNetCompact checkpoint")
    _restack_rdb_weights(p)
    return p


def _restack_rdb_weights(p: dict) -> None:
    """Regroup each ResidualDenseBlock's conv weights by INPUT segment (see _rdb).

    A conv over a channel concat equals the sum of convs of each segment with the
    matching weight slice, and every dense conv's x / x1 / x2 / x3 segment convolves
    the same tensor -- so the five convs' slices restack into one conv per PRODUCED
    tensor: x -> all five x-slices (64->224), x1 -> the four x1-slices (32->160), and
    so on. Pure weight reordering (bit-identical values), done once at load; the
    originals stay in p for introspection. No-op for SRVGG (no rdb keys)."""
    def restack_one(pre: str) -> None:
        W = [p[f"{pre}.conv{k}.weight"] for k in (1, 2, 3, 4, 5)]       # (O,3,3,I)
        nf = W[0].shape[-1]                                             # feature ch (64)
        gc = W[0].shape[0]                                              # growth ch (32)

        def seg(k: int, s: int) -> Any:
            return W[k][..., nf + s * gc: nf + (s + 1) * gc]

        p[f"{pre}.stack_x.weight"] = mx.concatenate(
            [W[0]] + [W[k][..., :nf] for k in (1, 2, 3, 4)], axis=0)          # 64 -> 4gc+nf
        p[f"{pre}.stack_x1.weight"] = mx.concatenate(
            [seg(k, 0) for k in (1, 2, 3, 4)], axis=0)                        # gc -> 3gc+nf
        p[f"{pre}.stack_x2.weight"] = mx.concatenate(
            [seg(k, 1) for k in (2, 3, 4)], axis=0)                           # gc -> 2gc+nf
        p[f"{pre}.stack_x3.weight"] = mx.concatenate(
            [seg(k, 2) for k in (3, 4)], axis=0)                              # gc -> gc+nf
        p[f"{pre}.stack_x4.weight"] = seg(4, 3)                               # gc -> nf

    i = 0
    while f"body.{i}.rdb1.conv1.weight" in p:
        for j in (1, 2, 3):
            restack_one(f"body.{i}.rdb{j}")
        i += 1


def conv(x: Any, p: dict, key: str) -> Any:
    return mx.conv2d(x, p[f"{key}.weight"], stride=1, padding=1) + p[f"{key}.bias"]


def lrelu(x: Any, slope: float = 0.2) -> Any:
    return mx.where(x >= 0, x, x * slope)


def _nearest_upsample(x: Any, r: int) -> Any:
    """NHWC nearest-neighbor r-x upsample (torch F.interpolate mode='nearest')."""
    n, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, r, w, r, c))
    return x.reshape(n, h * r, w * r, c)


def _pixel_shuffle(x: Any, r: int) -> Any:
    """NHWC pixel-shuffle (torch PixelShuffle): (n,h,w,c*r^2) -> (n,h*r,w*r,c)."""
    n, h, w, cr = x.shape
    c = cr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _pixel_unshuffle(x: Any, r: int) -> Any:
    """NHWC inverse pixel-shuffle, basicsr channel order (c, r_h, r_w)."""
    n, h, w, c = x.shape
    x = x.reshape(n, h // r, r, w // r, r, c)
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))          # n, h/r, w/r, c, r_h, r_w
    return x.reshape(n, h // r, w // r, c * r * r)


def _rdb(x: Any, p: dict, prefix: str) -> Any:
    """ResidualDenseBlock: 5 dense convs, lrelu(0.2), 0.2 residual scale.

    Computed concat-free from the load-time restacked weights (_restack_rdb_weights):
    conv-over-concat = sum of per-segment convs, so each produced tensor is convolved
    ONCE with all its consumers' weight slices and the pre-activations are recombined
    by slicing. Exactly the same math, reshaped for MLX: the incremental dense concats
    (a full copy each) disappear, and the x-stack (64 -> 4gc+nf = 224) crosses MLX's
    winograd gate (C+O >= 256, conv.cpp), where the original thin 32-out growth convs
    ran implicit GEMM at ~half throughput. ~1.35x on the dense body (measured). The
    recombination sums run in fp32: the split rounds each partial to fp16 where the
    original GEMM accumulated all of K in fp32, so summing partials in fp32 keeps the
    only extra rounding at the conv outputs themselves."""
    gc = p[f"{prefix}.conv1.bias"].size          # growth channels (32)
    nf = p[f"{prefix}.conv5.bias"].size          # feature channels (64)
    dt = x.dtype
    f32 = mx.float32

    def b(k: int) -> Any:
        return p[f"{prefix}.conv{k}.bias"].astype(f32)

    ax = mx.conv2d(x, p[f"{prefix}.stack_x.weight"], stride=1, padding=1)
    x1 = lrelu((ax[..., 0:gc].astype(f32) + b(1)).astype(dt))
    a1 = mx.conv2d(x1, p[f"{prefix}.stack_x1.weight"], stride=1, padding=1)
    x2 = lrelu((ax[..., gc:2 * gc].astype(f32) + a1[..., 0:gc].astype(f32)
                + b(2)).astype(dt))
    a2 = mx.conv2d(x2, p[f"{prefix}.stack_x2.weight"], stride=1, padding=1)
    x3 = lrelu((ax[..., 2 * gc:3 * gc].astype(f32) + a1[..., gc:2 * gc].astype(f32)
                + a2[..., 0:gc].astype(f32) + b(3)).astype(dt))
    a3 = mx.conv2d(x3, p[f"{prefix}.stack_x3.weight"], stride=1, padding=1)
    x4 = lrelu((ax[..., 3 * gc:4 * gc].astype(f32) + a1[..., 2 * gc:3 * gc].astype(f32)
                + a2[..., gc:2 * gc].astype(f32) + a3[..., 0:gc].astype(f32)
                + b(4)).astype(dt))
    a4 = mx.conv2d(x4, p[f"{prefix}.stack_x4.weight"], stride=1, padding=1)
    x5 = (ax[..., 4 * gc:4 * gc + nf].astype(f32) + a1[..., 3 * gc:3 * gc + nf].astype(f32)
          + a2[..., 2 * gc:2 * gc + nf].astype(f32) + a3[..., gc:gc + nf].astype(f32)
          + a4.astype(f32) + b(5))                   # conv5: no activation
    return (x5 * 0.2).astype(dt) + x


def _rrdb(x: Any, p: dict, i: int) -> Any:
    out = _rdb(x, p, f"body.{i}.rdb1")
    out = _rdb(out, p, f"body.{i}.rdb2")
    out = _rdb(out, p, f"body.{i}.rdb3")
    return out * 0.2 + x


def _num_blocks(p: dict) -> int:
    i = 0
    while f"body.{i}.rdb1.conv1.weight" in p:
        i += 1
    return i


def scale_of(p: dict) -> int:
    if "conv_first.weight" in p:                       # RRDBNet
        return _IN_CH_TO_SCALE[p["conv_first.weight"].shape[-1]]
    last = max(int(k.split(".")[1]) for k in p         # SRVGG: last conv -> 3*scale^2 out
               if k.startswith("body.") and p[k].ndim == 4)
    return int(round((p[f"body.{last}.weight"].shape[0] / 3) ** 0.5))


def _upscale_rrdbnet(x: Any, p: dict) -> Any:
    """RRDBNet forward. Pure (no eval): run via _compiled_forward so the 23 dense
    blocks fuse into one graph -- both faster and LOWER peak memory than a per-block
    eval at realistic sizes (5.4 vs 6.6 GB at 512px), since MLX's whole-graph planner
    reuses buffers better than manual per-block eviction."""
    scale = scale_of(p)
    nb = _num_blocks(p)
    if scale == 2:
        feat = _pixel_unshuffle(x, 2)
    elif scale == 1:
        feat = _pixel_unshuffle(x, 4)
    else:
        feat = x
    feat = conv(feat, p, "conv_first")
    body = feat
    for i in range(nb):
        body = _rrdb(body, p, i)
    feat = feat + conv(body, p, "conv_body")
    feat = lrelu(conv(_nearest_upsample(feat, 2), p, "conv_up1"))
    feat = lrelu(conv(_nearest_upsample(feat, 2), p, "conv_up2"))
    out = conv(lrelu(conv(feat, p, "conv_hr")), p, "conv_last")
    return mx.clip(out, 0.0, 1.0)


def _upscale_srvgg(x: Any, p: dict) -> Any:
    """SRVGGNetCompact forward: conv/PReLU stack, then pixel-shuffle + nearest residual."""
    out = x
    i = 0
    while f"body.{i}.weight" in p:
        w = p[f"body.{i}.weight"]
        if w.ndim == 4:
            out = conv(out, p, f"body.{i}")
        else:                                          # per-channel PReLU (NHWC: last axis)
            out = mx.where(out >= 0, out, out * w)
        i += 1
    scale = scale_of(p)
    return mx.clip(_pixel_shuffle(out, scale) + _nearest_upsample(x, scale), 0.0, 1.0)


_COMPILE_CACHE: dict = {}


def _compiled_forward(p: dict):
    """Compile the (pure, shape-stable) forward once per loaded checkpoint and reuse
    it for every frame. Both arches fuse into a single graph for ~1.25-1.3x over the
    op-by-op path with byte-identical output (profiled 128-512px). For RRDBNet the
    fused graph is also LOWER peak memory than the old per-block eval (5.4 vs 6.6 GB
    at 512px) -- MLX's whole-graph planner reuses buffers better than manual eviction.

    Keyed by id(p); the cache entry closes over p, so p stays alive and its id stays
    stable. Best paired with a capped MLX buffer cache (mx.set_cache_limit, as
    generate.py sets) so per-frame allocation churn does not grow into swap --
    uncapped, MLX hoards freed buffers as RSS.
    """
    forward = _upscale_rrdbnet if "conv_first.weight" in p else _upscale_srvgg
    return _cached(_COMPILE_CACHE, id(p), lambda: mx.compile(lambda x: forward(x, p)))


def upscale_frame(x: Any, p: dict) -> Any:
    """Upscale one NHWC frame (n,h,w,3) in [0,1] with whichever arch is loaded."""
    return _compiled_forward(p)(x)


def upscale(frames: list, p: dict) -> list:
    """Upscale a list of NHWC frames independently (per-frame, no temporal path)."""
    dt = next(iter(p.values())).dtype
    out = []
    for f in frames:
        x = mx.clip(f, 0.0, 1.0).astype(dt)
        if x.ndim == 3:
            x = x[None]
        y = upscale_frame(x, p)
        mx.eval(y)
        out.append(y)
    return out


if __name__ == "__main__":
    params = load_params()
    print(f"loaded RRDBNet: scale={scale_of(params)}x, blocks={_num_blocks(params)}")
    clip = [mx.random.uniform(shape=(1, 48, 64, 3))]
    mx.eval(*clip)
    sr = upscale(clip, params)
    print(f"upscale: {tuple(clip[0].shape)} -> {tuple(sr[0].shape)}")
