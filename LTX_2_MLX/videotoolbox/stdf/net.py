"""MLX port of STDF (Spatio-Temporal Deformable Fusion) for compressed-video
artifact removal -- the MFVQE network of Deng et al. (AAAI 2020).

Reimplemented from the reference net_stdf.py as a spec; this is clean MLX code, not
a wrapper. A U-Net regresses a per-frame deformable offset + modulation mask, then a
modulated deform-conv (DCNv2, the shared videotoolbox deform_conv2d -- verified
against torchvision) fuses a 2*radius+1 frame window; a plain-CNN head predicts a
residual added onto the center frame. Operates on the luma (Y) channel (in_nc=1).

Layout: MLX-native NHWC. Conv weights -> (O,kH,kW,I) at load; ConvTranspose weights
-> (O,kH,kW,I) from torch's (I,O,kH,kW); the deform-conv weight stays torch NCHW
(O,I,kH,kW) for deform_conv2d.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from ..deform_conv import deform_conv2d
from ..weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# Both bundled checkpoints are R3 (7-frame), Y-only; they differ only in training data.
_VARIANTS = {
    "mfqev2": "stdf_mfqev2_r3.safetensors",       # HEVC MFQEv2, multi-QP (general)
    "vimeo90k": "stdf_vimeo90k_r3.safetensors",   # Vimeo90K, HEVC All-Intra QP37
}
_DEFAULT_VARIANT = "mfqev2"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Bundled variant token (mfqev2 / vimeo90k) or a path."""
    return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: Conv weights -> NHWC (O,kH,kW,I); ConvTranspose
    (the 4x4 upsamplers) -> (O,kH,kW,I) from torch (I,O,kH,kW); the deform-conv weight
    stays torch NCHW for deform_conv2d. All cast to `dtype` (default fp16)."""
    w = mx.load(str(resolve_weights(path)))
    p: dict = {}
    for k, v in w.items():
        if k == "ffnet.deform_conv.weight":
            a = v                                       # keep torch NCHW (O,I,kH,kW)
        elif v.ndim == 4 and tuple(v.shape[-2:]) == (4, 4):
            a = mx.transpose(v, (1, 2, 3, 0))           # ConvTranspose (I,O,4,4) -> (O,4,4,I)
        elif v.ndim == 4:
            a = mx.transpose(v, (0, 2, 3, 1))           # Conv (O,I,kH,kW) -> (O,kH,kW,I)
        else:
            a = v
        p[k] = a.astype(dtype)
    return p


def _config(p: dict) -> tuple:
    """(in_nc color count, input_len = 2*radius+1, nb) inferred from the weights."""
    in_nc_total = int(p["ffnet.in_conv.0.weight"].shape[-1])     # NHWC Cin
    in_nc = int(p["qenet.out_conv.weight"].shape[0])             # NHWC Cout
    nb = 1 + sum(1 for k in p if k.startswith("ffnet.dn_conv") and k.endswith(".0.weight"))
    return in_nc, in_nc_total // in_nc, nb


def _relu(x: Any) -> Any:
    return mx.maximum(x, 0)


def _conv(x: Any, p: dict, key: str, stride: int = 1, pad: int = 1) -> Any:
    return mx.conv2d(x, p[f"{key}.weight"], stride=stride, padding=pad) + p[f"{key}.bias"]


def _convt(x: Any, p: dict, key: str) -> Any:
    """ConvTranspose 4x4 stride 2 pad 1 (exact 2x upsample) + bias."""
    return mx.conv_transpose2d(x, p[f"{key}.weight"], stride=2, padding=1) + p[f"{key}.bias"]


def _reflect_pad_to(x: Any, m: int) -> Any:
    """Reflect-pad NHWC x on bottom/right so H,W are multiples of m. The U-Net has
    three /2 stages, so it needs a multiple of 8. [::-1] mirrors (MLX has no flip)."""
    _, h, w, _ = x.shape
    ph, pw = (-h) % m, (-w) % m
    if ph:
        x = mx.concatenate([x, x[:, h - 1 - ph:h - 1, :, :][:, ::-1, :, :]], axis=1)
    if pw:
        x = mx.concatenate([x, x[:, :, w - 1 - pw:w - 1, :][:, :, ::-1, :]], axis=2)
    return x


def _stdf(x: Any, p: dict, nb: int) -> Any:
    """Spatio-temporal deformable fusion. x: (N,H,W,in_nc_total) stacked frames; the
    U-Net regresses offsets+mask, the deform-conv fuses the stacked frames -> features."""
    in_nc_total = x.shape[-1]
    feats = [_relu(_conv(x, p, "ffnet.in_conv.0"))]
    for i in range(1, nb):
        d = _relu(_conv(feats[-1], p, f"ffnet.dn_conv{i}.0", stride=2))
        feats.append(_relu(_conv(d, p, f"ffnet.dn_conv{i}.2")))
    out = _relu(_conv(feats[-1], p, "ffnet.tr_conv.0", stride=2))
    out = _relu(_conv(out, p, "ffnet.tr_conv.2"))
    out = _relu(_convt(out, p, "ffnet.tr_conv.4"))
    for i in range(nb - 1, 0, -1):
        out = _relu(_conv(mx.concatenate([out, feats[i]], axis=-1), p, f"ffnet.up_conv{i}.0"))
        out = _relu(_convt(out, p, f"ffnet.up_conv{i}.2"))
    out = _relu(_conv(out, p, "ffnet.out_conv.0"))
    off_msk = _conv(out, p, "ffnet.offset_mask")                # (N,H,W, in_nc_total*3*9)
    n_off = in_nc_total * 2 * 9
    off = off_msk[..., :n_off]
    msk = mx.sigmoid(off_msk[..., n_off:])
    fused = deform_conv2d(                                       # NHWC -> NCHW for the kernel
        mx.transpose(x, (0, 3, 1, 2)), mx.transpose(off, (0, 3, 1, 2)),
        p["ffnet.deform_conv.weight"], p.get("ffnet.deform_conv.bias"),
        mx.transpose(msk, (0, 3, 1, 2)), stride=1, padding=1, deform_groups=in_nc_total)
    return _relu(mx.transpose(fused, (0, 2, 3, 1)))


def _qe(x: Any, p: dict) -> Any:
    """Plain-CNN quality-enhancement head -> residual map."""
    out = _relu(_conv(x, p, "qenet.in_conv.0"))
    i = 0
    while f"qenet.hid_conv.{i}.weight" in p:
        out = _relu(_conv(out, p, f"qenet.hid_conv.{i}"))
        i += 2
    return _conv(out, p, "qenet.out_conv")


def deblock(frames: list, p: dict, strength: float = 1.0) -> Any:
    """Deblock the center of a (2*radius+1)-frame window. `frames` is a list of
    (N,H,W,in_nc) arrays in [0,1] (in_nc=1 for the bundled Y-only models); returns the
    deblocked center frame (N,H,W,in_nc). The net predicts a residual onto the center;
    `strength` scales that residual (1.0 = full deblock, 0.0 = passthrough) to trade
    artifact removal against softening of fine texture."""
    in_nc, input_len, nb = _config(p)
    radius = (input_len - 1) // 2
    if len(frames) != input_len:
        raise ValueError(f"STDF needs {input_len} frames (radius {radius}); got {len(frames)}")
    dt = next(iter(p.values())).dtype
    # Channel layout matches the reference: all frames of channel 0, then channel 1, ...
    x = mx.concatenate([f.astype(dt) for f in frames], axis=-1) if in_nc == 1 else \
        mx.concatenate([f[..., c:c + 1].astype(dt) for c in range(in_nc) for f in frames], axis=-1)
    h, w = x.shape[1], x.shape[2]
    xp = _reflect_pad_to(x, 8)
    res = _qe(_stdf(xp, p, nb), p)
    centers = mx.concatenate(
        [xp[..., radius + c * input_len:radius + c * input_len + 1] for c in range(in_nc)], axis=-1)
    return (centers + float(strength) * res)[:, :h, :w, :]


if __name__ == "__main__":
    p = load_params()
    in_nc, input_len, nb = _config(p)
    print(f"loaded STDF: in_nc={in_nc}, frames={input_len} (radius {input_len // 2}), nb={nb}")
    mx.random.seed(0)
    frames = [mx.clip(mx.random.uniform(shape=(1, 64, 96, in_nc)), 0, 1) for _ in range(input_len)]
    mx.eval(*frames)
    out = deblock(frames, p)
    mx.eval(out)
    print(f"deblock: {input_len}x{tuple(frames[0].shape)} -> {tuple(out.shape)}, "
          f"residual mean={float(mx.mean(mx.abs(out - frames[input_len // 2]))):.4f}")
