"""MLX port of FBCNN (Flexible Blind CNN for JPEG artifact removal), Jiang et al.,
ICCV 2021. Reimplemented from models/network_fbcnn.py as a spec -- clean MLX code, no
upstream code run.

A U-Net (head -> 3 strideconv-down stages -> bottleneck body) with a quality-factor
(QF) predictor on the bottleneck. The predicted QF -- or a manual override -- is
embedded into per-channel FiLM (gamma, beta) parameters that modulate QFAttention
residual blocks in the decoder, so the artifact-removal strength is controllable. The
net predicts the clean image directly (no global residual). Color model: in_nc=out_nc=3,
nc=[64,128,256,512], nb=4 (~72M params).

QF semantics (from the reference): the internal qf is INVERTED JPEG quality -- the
displayed quality factor is (1 - qf) * 100, so a HIGHER qf_input assumes a LOWER-quality
(more compressed) input and removes artifacts more aggressively. qf_input=None runs
blind (uses the per-image predicted qf).

Layout: MLX-native NHWC. Conv weights -> (O,kH,kW,I) at load; the 2x2 stride-2
ConvTranspose upsamplers -> (O,kH,kW,I) from torch (I,O,kH,kW); Linear weights -> (in,out)
from torch (out,in) so forward is x @ W + b.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..weights import resolve_weights as _resolve_weights

# The checkpoints are ~287MB each and are NOT bundled in-repo (free-account space); see
# weights/README.md. fbcnn_color is the RGB one used by FbcnnDeblocker.
_DOWNLOAD_URL = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_VARIANTS = {
    "color": "fbcnn_color.safetensors",            # RGB JPEG (the video default)
    "gray": "fbcnn_gray.safetensors",              # grayscale JPEG
    "gray_double": "fbcnn_gray_double.safetensors",  # grayscale double-JPEG
}
_DEFAULT_VARIANT = "color"

# The only ConvTranspose layers: m_up{1,2,3}.0 (the upsample block of each m_up list).
_UP_KEYS = {"m_up1.0.weight", "m_up2.0.weight", "m_up3.0.weight"}


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Variant token (color / gray / gray_double) or a path; falls back to
    $FBCNN_WEIGHTS. The ~287MB checkpoints are not shipped in-repo, so a miss raises a
    message pointing at the download + conversion (see weights/README.md)."""
    if spec is None or spec == "":
        spec = os.environ.get("FBCNN_WEIGHTS")
    try:
        return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nFBCNN weights are not bundled in the repo (~287MB each). Get one:\n"
            f"  curl -L -o fbcnn_color.pth {_DOWNLOAD_URL}\n"
            f"  python scripts/pth_to_safetensors.py fbcnn_color.pth -o "
            f"{_WEIGHTS_DIR / _VARIANTS['color']} --strip-prefix ''\n"
            f"or point $FBCNN_WEIGHTS / --deblock-weights at an existing .safetensors."
        ) from None


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: Conv -> NHWC (O,kH,kW,I); ConvTranspose upsamplers
    -> (O,kH,kW,I) from torch (I,O,kH,kW); Linear -> (in,out) from torch (out,in). All
    cast to `dtype` (default fp16)."""
    w = mx.load(str(resolve_weights(path)))
    p: dict = {}
    for k, v in w.items():
        if k in _UP_KEYS:
            a = mx.transpose(v, (1, 2, 3, 0))      # ConvTranspose (I,O,kH,kW) -> (O,kH,kW,I)
        elif v.ndim == 4:
            a = mx.transpose(v, (0, 2, 3, 1))      # Conv (O,I,kH,kW) -> (O,kH,kW,I)
        elif v.ndim == 2:
            a = mx.transpose(v, (1, 0))            # Linear (out,in) -> (in,out)
        else:
            a = v
        p[k] = a.astype(dtype)
    return p


def _config(p: dict) -> tuple:
    """(in_nc, nb) inferred from the weights."""
    in_nc = int(p["m_head.weight"].shape[-1])      # NHWC Cin of the head conv
    nb = sum(1 for k in p if k.startswith("m_body_encoder.") and k.endswith(".res.0.weight"))
    return in_nc, nb


def _relu(x: Any) -> Any:
    return mx.maximum(x, 0)


def _conv(x: Any, p: dict, key: str, stride: int = 1, pad: int = 1) -> Any:
    return mx.conv2d(x, p[f"{key}.weight"], stride=stride, padding=pad) + p[f"{key}.bias"]


def _convt(x: Any, p: dict, key: str) -> Any:
    """ConvTranspose 2x2 stride 2 pad 0 (exact 2x upsample) + bias."""
    return mx.conv_transpose2d(x, p[f"{key}.weight"], stride=2, padding=0) + p[f"{key}.bias"]


def _linear(x: Any, p: dict, key: str) -> Any:
    return mx.matmul(x, p[f"{key}.weight"]) + p[f"{key}.bias"]


def _res(x: Any, p: dict, key: str) -> Any:
    """ResBlock 'CRC': x + conv(relu(conv(x)))."""
    r = _conv(_relu(_conv(x, p, f"{key}.res.0")), p, f"{key}.res.2")
    return x + r


def _qfatt(x: Any, p: dict, key: str, gamma: Any, beta: Any) -> Any:
    """QFAttention: x + gamma * conv(relu(conv(x))) + beta, FiLM-modulated per channel."""
    r = _conv(_relu(_conv(x, p, f"{key}.res.0")), p, f"{key}.res.2")
    n, c = gamma.shape
    g = gamma.reshape(n, 1, 1, c)
    b = beta.reshape(n, 1, 1, c)
    return x + g * r + b


def _down(x: Any, p: dict, prefix: str, nb: int) -> Any:
    """nb ResBlocks then a 2x2 stride-2 downsample conv (at index nb)."""
    for i in range(nb):
        x = _res(x, p, f"{prefix}.{i}")
    return _conv(x, p, f"{prefix}.{nb}", stride=2, pad=0)


def _body(x: Any, p: dict, prefix: str, nb: int) -> Any:
    for i in range(nb):
        x = _res(x, p, f"{prefix}.{i}")
    return x


def _qf_pred(x: Any, p: dict, nb: int) -> Any:
    """Bottleneck QF predictor -> scalar qf (N,1) in [0,1] (inverted JPEG quality)."""
    for i in range(nb):
        x = _res(x, p, f"qf_pred.{i}")
    x = mx.mean(x.astype(mx.float32), axis=(1, 2)).astype(p["qf_pred.6.weight"].dtype)  # global avgpool
    x = _relu(_linear(x, p, "qf_pred.6"))
    x = _relu(_linear(x, p, "qf_pred.8"))
    return mx.sigmoid(_linear(x, p, "qf_pred.10"))


def _qf_embed(qf: Any, p: dict) -> Any:
    x = _relu(_linear(qf, p, "qf_embed.0"))
    x = _relu(_linear(x, p, "qf_embed.2"))
    return _relu(_linear(x, p, "qf_embed.4"))


def _film(emb: Any, p: dict, lvl: str) -> tuple:
    g = mx.sigmoid(_linear(emb, p, f"to_gamma_{lvl}.0"))
    b = mx.tanh(_linear(emb, p, f"to_beta_{lvl}.0"))
    return g, b


def _replicate_pad(x: Any, m: int) -> Any:
    """ReplicationPad2d on bottom/right so H,W are multiples of m (3 down stages -> 8)."""
    _, h, w, c = x.shape
    pb, pr = (-h) % m, (-w) % m
    if pb:
        x = mx.concatenate([x, mx.broadcast_to(x[:, h - 1:h, :, :], (x.shape[0], pb, w, c))], axis=1)
    if pr:
        x = mx.concatenate([x, mx.broadcast_to(x[:, :, w - 1:w, :], (x.shape[0], x.shape[1], pr, c))], axis=2)
    return x


def fbcnn(x: Any, p: dict, qf_input: Any = None, nb: int | None = None) -> tuple:
    """Restore one batch. x: (N,H,W,in_nc) in [0,1]. qf_input: None = blind (use the
    predicted qf), else a float or (N,1) in [0,1] model-space (higher = assume lower
    JPEG quality = stronger removal). Returns (restored (N,H,W,in_nc), qf (N,1) predicted)."""
    if nb is None:
        _, nb = _config(p)
    h, w = x.shape[1], x.shape[2]
    dt = p["m_head.weight"].dtype
    x = _replicate_pad(x.astype(dt), 8)

    x1 = _conv(x, p, "m_head")
    x2 = _down(x1, p, "m_down1", nb)
    x3 = _down(x2, p, "m_down2", nb)
    x4 = _down(x3, p, "m_down3", nb)
    xe = _body(x4, p, "m_body_encoder", nb)
    qf = _qf_pred(xe, p, nb)
    xd = _body(xe, p, "m_body_decoder", nb)

    n = x.shape[0]
    if qf_input is None:
        emb_in = qf
    elif isinstance(qf_input, (int, float)):
        emb_in = mx.full((n, 1), float(qf_input), dtype=dt)
    else:
        emb_in = mx.array(qf_input).reshape(n, 1).astype(dt)
    emb = _qf_embed(emb_in, p)
    g3, b3 = _film(emb, p, "3")
    g2, b2 = _film(emb, p, "2")
    g1, b1 = _film(emb, p, "1")

    x = xd + x4
    x = _convt(x, p, "m_up3.0")
    for i in range(1, nb + 1):
        x = _qfatt(x, p, f"m_up3.{i}", g3, b3)
    x = x + x3
    x = _convt(x, p, "m_up2.0")
    for i in range(1, nb + 1):
        x = _qfatt(x, p, f"m_up2.{i}", g2, b2)
    x = x + x2
    x = _convt(x, p, "m_up1.0")
    for i in range(1, nb + 1):
        x = _qfatt(x, p, f"m_up1.{i}", g1, b1)
    x = x + x1
    x = _conv(x, p, "m_tail")
    return x[:, :h, :w, :], qf


if __name__ == "__main__":
    p = load_params(dtype=mx.float32)
    in_nc, nb = _config(p)
    print(f"loaded FBCNN: in_nc={in_nc}, nb={nb}")
    mx.random.seed(0)
    x = mx.clip(mx.random.uniform(shape=(1, 64, 96, in_nc)), 0, 1)
    mx.eval(x)
    out, qf = fbcnn(x, p)
    mx.eval(out, qf)
    print(f"blind: {tuple(x.shape)} -> {tuple(out.shape)}, predicted JPEG QF "
          f"{float((1 - qf[0, 0]) * 100):.1f}, residual mean {float(mx.mean(mx.abs(out - x))):.4f}")
