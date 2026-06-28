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

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_DEFAULT_WEIGHTS = "realesr_general_x4v3.safetensors"   # fast/gentle default
_IN_CH_TO_SCALE = {3: 4, 12: 2, 48: 1}   # conv_first in-channels -> output scale


def default_weights_path() -> Path:
    return _WEIGHTS_DIR / _DEFAULT_WEIGHTS


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
    return p


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
    """ResidualDenseBlock: 5 dense convs, lrelu(0.2), 0.2 residual scale."""
    x1 = lrelu(conv(x, p, f"{prefix}.conv1"))
    x2 = lrelu(conv(mx.concatenate([x, x1], axis=-1), p, f"{prefix}.conv2"))
    x3 = lrelu(conv(mx.concatenate([x, x1, x2], axis=-1), p, f"{prefix}.conv3"))
    x4 = lrelu(conv(mx.concatenate([x, x1, x2, x3], axis=-1), p, f"{prefix}.conv4"))
    x5 = conv(mx.concatenate([x, x1, x2, x3, x4], axis=-1), p, f"{prefix}.conv5")
    return x5 * 0.2 + x


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
    """RRDBNet forward. Per-RRDB eval bounds memory across the dense blocks."""
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
        mx.eval(body)                 # dense blocks -> evict transients per block
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


def upscale_frame(x: Any, p: dict) -> Any:
    """Upscale one NHWC frame (n,h,w,3) in [0,1] with whichever arch is loaded."""
    if "conv_first.weight" in p:
        return _upscale_rrdbnet(x, p)
    return _upscale_srvgg(x, p)


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
