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

try:
    from ..basicvsrpp.net import (
        _compute_flows,
        _pixelshuffle_pack,
        _resblocks_with_input,
        conv,
        flow_warp,
        lrelu,
        resize,
    )
except ImportError:   # running net.py directly as a script
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "basicvsrpp"))
    from net import (  # type: ignore
        _compute_flows,
        _pixelshuffle_pack,
        _resblocks_with_input,
        conv,
        flow_warp,
        lrelu,
        resize,
    )

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
            residue = conv(_resblocks_with_input(f, p, "image_cleaning.0"), p, "image_cleaning.1")
            nf = f + residue
            mean_abs = mx.mean(mx.abs(residue.astype(mx.float32)))
            mx.eval(nf, mean_abs)
            next_frames.append(nf)
            means.append(mean_abs)
        cleaned = next_frames
        if not means or float(mx.mean(mx.stack(means))) < thres:
            break
    return cleaned


def _basicvsr(frames: list, p: dict, residual_strength: float) -> list:
    n, h, w, _ = frames[0].shape
    mid = int(p["basicvsr.backward_resblocks.main.0.bias"].shape[0])
    feat_prop = mx.zeros((n, h, w, mid), dtype=frames[0].dtype)

    flows_forward, flows_backward = _compute_flows(frames, p)

    backward_feats: list[Any] = [None] * len(frames)
    for i in range(len(frames) - 1, -1, -1):
        if i < len(frames) - 1:
            feat_prop = flow_warp(feat_prop, flows_backward[i])
        feat_prop = _resblocks_with_input(
            mx.concatenate([frames[i], feat_prop], axis=-1),
            p,
            "basicvsr.backward_resblocks",
        )
        mx.eval(feat_prop)
        backward_feats[i] = feat_prop

    feat_prop = mx.zeros((n, h, w, mid), dtype=frames[0].dtype)
    outs = []
    for i, frame in enumerate(frames):
        if i > 0:
            feat_prop = flow_warp(feat_prop, flows_forward[i - 1])
        feat_prop = _resblocks_with_input(
            mx.concatenate([frame, feat_prop], axis=-1),
            p,
            "basicvsr.forward_resblocks",
        )
        out = mx.concatenate([backward_feats[i], feat_prop], axis=-1)
        out = lrelu(conv(out, p, "basicvsr.fusion", pad=0))
        out = lrelu(_pixelshuffle_pack(out, p, "basicvsr.upsample1"))
        out = lrelu(_pixelshuffle_pack(out, p, "basicvsr.upsample2"))
        out = lrelu(conv(out, p, "basicvsr.conv_hr"))
        residual = conv(out, p, "basicvsr.conv_last")
        base = resize(frame, h * 4, w * 4, False)
        out = mx.clip(base + residual * float(residual_strength), 0.0, 1.0)
        mx.eval(out)
        outs.append(out)
    return outs


def upscale(
    frames: list,
    p: dict,
    *,
    dynamic_refine_thres: float = 5.0,
    clean_iters: int = 3,
    residual_strength: float = 1.0,
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
    return _basicvsr(cleaned, p, residual_strength)


if __name__ == "__main__":
    params = load_params()
    mx.random.seed(0)
    clip = [mx.random.uniform(shape=(1, 64, 64, 3)) for _ in range(2)]
    mx.eval(*clip)
    sr = upscale(clip, params, clean_iters=1)
    mx.eval(*sr)
    print(f"upscale: {len(sr)} frames, 64x64 -> {sr[0].shape[1]}x{sr[0].shape[2]}")
