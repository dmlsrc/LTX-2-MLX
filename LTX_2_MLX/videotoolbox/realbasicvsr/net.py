"""MLX RealBasicVSR x4 net.

This is the real-world VSR sibling of BasicVSR++: an iterative image-cleaning
front-end followed by the first-order BasicVSR recurrent core. The checkpoint is
loaded in NHWC layout, but follows the OpenMMLab key names after stripping
``generator_ema.`` at conversion time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

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

# Per-checkpoint compiled step functions (built lazily, keyed by id(p)). The clean
# step and the forward output step are pure and shape-stable, so mx.compile fuses
# each into one graph (~1.06x / ~1.15x, byte-identical, profiled with the MLX cache
# capped). spynet is intentionally left uncompiled -- compiling it drifts the fp16
# flow by ~0.02, which would break bit-exactness.
_CLEAN_COMPILE_CACHE: dict = {}
_FWD_COMPILE_CACHE: dict = {}


def _compiled_clean(p: dict):
    fn = _CLEAN_COMPILE_CACHE.get(id(p))
    if fn is None:
        fn = mx.compile(
            lambda f: conv(_resblocks_with_input(f, p, "image_cleaning.0"), p, "image_cleaning.1"))
        _CLEAN_COMPILE_CACHE[id(p)] = fn
    return fn


def _compiled_fwd(p: dict):
    """Compiled forward output step: (frame, warped feat_prop, backward feat) ->
    (new feat_prop, HR residual). The cheap base-resize + residual scale + clip stay
    outside so residual_strength is not baked into the graph."""
    fn = _FWD_COMPILE_CACHE.get(id(p))
    if fn is None:
        def step(frame, feat_prop, backward_feat):
            fp = _resblocks_with_input(
                mx.concatenate([frame, feat_prop], axis=-1), p, "basicvsr.forward_resblocks")
            out = mx.concatenate([backward_feat, fp], axis=-1)
            out = lrelu(conv(out, p, "basicvsr.fusion", pad=0))
            out = lrelu(_pixelshuffle_pack(out, p, "basicvsr.upsample1"))
            out = lrelu(_pixelshuffle_pack(out, p, "basicvsr.upsample2"))
            out = lrelu(conv(out, p, "basicvsr.conv_hr"))
            return fp, conv(out, p, "basicvsr.conv_last")
        fn = mx.compile(step)
        _FWD_COMPILE_CACHE[id(p)] = fn
    return fn

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_DEFAULT_WEIGHTS = "realbasicvsr_x4.safetensors"


def default_weights_path() -> Path:
    return _WEIGHTS_DIR / _DEFAULT_WEIGHTS


def _load_safetensors(path: Path) -> dict:
    if path.exists():
        return mx.load(str(path))
    shards = sorted(path.parent.glob(f"{path.stem}.shard*{path.suffix}"))
    if not shards:
        raise FileNotFoundError(
            f"{path} (no file, and no {path.stem}.shard*{path.suffix} shards)"
        )
    w = {}
    for s in shards:
        w.update(mx.load(str(s)))
    return w


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load RealBasicVSR weights and transpose torch conv kernels for MLX.

    Files converted with ``scripts/pth_to_safetensors.py --only-prefix
    generator_ema. --strip-prefix generator_ema.`` load directly. For robustness,
    this also accepts safetensors that still contain ``generator_ema.`` or
    ``generator.`` prefixes and chooses EMA when present.
    """
    src = Path(path or default_weights_path())
    w = _load_safetensors(src)
    keys = tuple(w.keys())
    has_clean_keys = any(k.startswith(("image_cleaning.", "basicvsr.")) for k in keys)
    selected_prefix = ""
    if any(k.startswith("generator_ema.") for k in keys):
        selected_prefix = "generator_ema."
    elif not has_clean_keys and any(k.startswith("generator.") for k in keys):
        selected_prefix = "generator."

    p: dict = {}
    for k, v in w.items():
        if selected_prefix:
            if not k.startswith(selected_prefix):
                continue
            k = k[len(selected_prefix):]
        elif k.startswith(("generator.", "generator_ema.")):
            continue
        if k == "step_counter":
            continue
        if k.startswith("basicvsr.spynet."):
            k = k[len("basicvsr."):]
        if k in ("spynet.mean", "spynet.std"):
            a = v.reshape(1, 1, 1, 3)
        elif v.ndim == 4:
            a = mx.transpose(v, (0, 2, 3, 1))
        else:
            a = v
        p[k] = a.astype(dtype)

    required = ("image_cleaning.0.main.0.weight", "basicvsr.conv_last.weight", "spynet.mean")
    missing = [k for k in required if k not in p]
    if missing:
        raise ValueError(f"{src} is not a RealBasicVSR generator checkpoint; missing {missing}")
    return p


def _clean(frames: list, p: dict, dynamic_refine_thres: float, max_iters: int) -> list:
    thres = float(dynamic_refine_thres) / 255.0
    cleaned = frames
    for _ in range(max(0, int(max_iters))):
        next_frames = []
        means = []
        for f in cleaned:
            residue = _compiled_clean(p)(f)
            nf = f + residue
            mean_abs = mx.mean(mx.abs(residue.astype(mx.float32)))
            mx.eval(nf, mean_abs)
            next_frames.append(nf)
            means.append(mean_abs)
        cleaned = next_frames
        if not means or float(mx.mean(mx.stack(means))) < thres:
            break
    return cleaned


_FC_LO, _FC_HI = 1.0, 4.0   # fwd-bwd error ratio: <=LO fully trusts, >=HI drops


def _flow_consistency_mask(flow_use: Any, flow_rev: Any, dtype: Any) -> Any:
    """Soft forward-backward occlusion mask in [0,1] for a warp by ``flow_use``.

    1.0 where ``flow_use`` and the reverse flow ``flow_rev`` round-trip (reliable
    propagation), ramping to 0.0 where they disagree -- occlusions and fast-motion
    flow errors (disocclusions, moving edges) that smear RealBasicVSR's recurrent
    features. Catches gross flow failure only; it cannot catch consistent-but-wrong
    flow (e.g. a specular highlight that optical flow tracks self-consistently to
    the wrong place). Smooth form of the Sundaram et al. (2010) criterion
    ``|f + warp(f_rev, f)|^2 < a1 (|f|^2 + |warp|^2) + a2``. The
    arithmetic runs in fp32 then casts to ``dtype`` so it never upcasts an fp16
    feature path.
    """
    fu = flow_use.astype(mx.float32)
    rev_w = flow_warp(flow_rev, flow_use).astype(mx.float32)
    s = fu + rev_w
    err2 = s[..., 0] ** 2 + s[..., 1] ** 2
    mag2 = fu[..., 0] ** 2 + fu[..., 1] ** 2 + rev_w[..., 0] ** 2 + rev_w[..., 1] ** 2
    r = err2 / (0.01 * mag2 + 0.5)
    m = mx.clip((_FC_HI - r) / (_FC_HI - _FC_LO), 0.0, 1.0)
    return m[..., None].astype(dtype)


def _basicvsr(frames: list, p: dict, residual_strength: float,
              flow_consistency: float = 0.0) -> list:
    n, h, w, _ = frames[0].shape
    mid = int(p["basicvsr.backward_resblocks.main.0.bias"].shape[0])
    dt = frames[0].dtype

    flows_forward, flows_backward = _compute_flows(frames, p)

    # Precompute soft occlusion masks (flow-only, independent of feat_prop). With
    # strength s, mask = 1 - s*(1 - m_raw): s=0 reproduces the reference (no
    # masking), s=1 fully drops propagation where the flow round-trip fails.
    use_mask = flow_consistency > 0.0
    mask_b: list = []
    mask_f: list = []
    if use_mask:
        s = float(flow_consistency)
        for i in range(len(frames) - 1):
            mb = _flow_consistency_mask(flows_backward[i], flows_forward[i], dt)
            mf = _flow_consistency_mask(flows_forward[i], flows_backward[i], dt)
            mask_b.append(1.0 - s * (1.0 - mb))
            mask_f.append(1.0 - s * (1.0 - mf))
            mx.eval(mask_b[-1], mask_f[-1])

    feat_prop = mx.zeros((n, h, w, mid), dtype=dt)
    backward_feats: list[Any] = [None] * len(frames)
    for i in range(len(frames) - 1, -1, -1):
        if i < len(frames) - 1:
            feat_prop = flow_warp(feat_prop, flows_backward[i])
            if use_mask:
                feat_prop = feat_prop * mask_b[i]
        feat_prop = compiled_resblocks(
            mx.concatenate([frames[i], feat_prop], axis=-1),
            p,
            "basicvsr.backward_resblocks",
        )
        mx.eval(feat_prop)
        backward_feats[i] = feat_prop

    feat_prop = mx.zeros((n, h, w, mid), dtype=dt)
    outs = []
    for i, frame in enumerate(frames):
        if i > 0:
            feat_prop = flow_warp(feat_prop, flows_forward[i - 1])
            if use_mask:
                feat_prop = feat_prop * mask_f[i - 1]
        feat_prop, residual = _compiled_fwd(p)(frame, feat_prop, backward_feats[i])
        base = resize(frame, h * 4, w * 4, False)
        out = mx.clip(base + residual * float(residual_strength), 0.0, 1.0)
        mx.eval(out, feat_prop)
        outs.append(out)
    return outs


def upscale(
    frames: list,
    p: dict,
    *,
    dynamic_refine_thres: float = 5.0,
    clean_iters: int = 3,
    residual_strength: float = 1.0,
    flow_consistency: float = 0.0,
) -> list:
    """Upscale an LR clip 4x.

    ``frames`` is a list of NHWC arrays shaped ``(N,H,W,3)`` in [0,1]. The
    GAN RealBasicVSR configs use ``dynamic_refine_thres=5`` for test-time
    iterative cleaning; pass 255 to force a single cleaning pass. Values below
    1 for ``residual_strength`` attenuate GAN/pixel-shuffle artifacts while
    keeping the 4x bilinear base.
    """
    if not frames:
        return []
    dt = p["basicvsr.conv_last.weight"].dtype
    frames = [mx.clip(f, 0.0, 1.0).astype(dt) for f in frames]
    cleaned = _clean(frames, p, dynamic_refine_thres, clean_iters)
    return _basicvsr(cleaned, p, residual_strength, flow_consistency)


if __name__ == "__main__":
    params = load_params()
    mx.random.seed(0)
    clip = [mx.random.uniform(shape=(1, 64, 64, 3)) for _ in range(2)]
    mx.eval(*clip)
    sr = upscale(clip, params, clean_iters=1)
    mx.eval(*sr)
    print(f"upscale: {len(sr)} frames, 64x64 -> {sr[0].shape[1]}x{sr[0].shape[2]}")
