"""FastDVDnet video denoiser, ported to MLX.

Architecture ported from FastDVDnet (M. Tassano, J. Delon, T. Veit, "FastDVDnet:
Towards Real-Time Deep Video Denoising Without Flow Estimation", CVPR 2020).
Reference implementation is MIT-licensed, Copyright 2024 Matias Tassano. Only the
network structure and the (separately converted) weights are used here; this is a
clean MLX reimplementation of the forward pass.

A two-stage cascade of small U-Net denoising blocks:
  - temp1 (weights shared) denoises three overlapping triplets of a 5-frame
    window: (f0,f1,f2), (f1,f2,f3), (f2,f3,f4) -> three estimates of the center.
  - temp2 fuses those three estimates into the final denoised center frame.
Non-blind: each block also takes a noise map (a constant sigma in [0,1]).

Each DenBlock is InputConv -> 2x downsample -> 2x pixel-shuffle upsample ->
output conv, with skip adds and a final residual (predict noise, subtract from
the center frame). BatchNorm is folded into the preceding conv at load time
(inference only), so the forward pass is just conv + ReLU + pixel-shuffle + adds,
all MLX-native in NHWC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

_BN_EPS = 1e-5   # torch BatchNorm2d default

# One DenBlock's convolutions: (name, conv subkey, bn subkey | None, stride, groups).
# bn None => conv has no following BatchNorm (the pre-pixel-shuffle convs and the
# final output conv); those also have no bias in the reference (bias=False).
_CONVS = (
    ("inc0", "inc.convblock.0", "inc.convblock.1", 1, 3),   # InputCvBlock, grouped
    ("inc3", "inc.convblock.3", "inc.convblock.4", 1, 1),
    ("d0",   "downc0.convblock.0", "downc0.convblock.1", 2, 1),
    ("d0c0", "downc0.convblock.3.convblock.0", "downc0.convblock.3.convblock.1", 1, 1),
    ("d0c3", "downc0.convblock.3.convblock.3", "downc0.convblock.3.convblock.4", 1, 1),
    ("d1",   "downc1.convblock.0", "downc1.convblock.1", 2, 1),
    ("d1c0", "downc1.convblock.3.convblock.0", "downc1.convblock.3.convblock.1", 1, 1),
    ("d1c3", "downc1.convblock.3.convblock.3", "downc1.convblock.3.convblock.4", 1, 1),
    ("u2c0", "upc2.convblock.0.convblock.0", "upc2.convblock.0.convblock.1", 1, 1),
    ("u2c3", "upc2.convblock.0.convblock.3", "upc2.convblock.0.convblock.4", 1, 1),
    ("u2",   "upc2.convblock.1", None, 1, 1),                # -> pixel-shuffle
    ("u1c0", "upc1.convblock.0.convblock.0", "upc1.convblock.0.convblock.1", 1, 1),
    ("u1c3", "upc1.convblock.0.convblock.3", "upc1.convblock.0.convblock.4", 1, 1),
    ("u1",   "upc1.convblock.1", None, 1, 1),                # -> pixel-shuffle
    ("o0",   "outc.convblock.0", "outc.convblock.1", 1, 1),
    ("o3",   "outc.convblock.3", None, 1, 1),                # final conv, no BN/bias
)


def _fold(w: dict, prefix: str, conv_sub: str, bn_sub: str | None, dtype: Any):
    """Return (weight NHWC (O,kH,kW,I/groups), bias (O,) | None) for one conv,
    folding its following BatchNorm (if any) into the weight + bias. The fold is
    computed in float32 and the result cast to dtype (fp16 by default)."""
    W = w[f"{prefix}.{conv_sub}.weight"].astype(mx.float32)   # torch layout (O,I,kH,kW)
    if bn_sub is None:
        return mx.transpose(W, (0, 2, 3, 1)).astype(dtype), None
    g = w[f"{prefix}.{bn_sub}.weight"].astype(mx.float32)
    b = w[f"{prefix}.{bn_sub}.bias"].astype(mx.float32)
    mean = w[f"{prefix}.{bn_sub}.running_mean"].astype(mx.float32)
    var = w[f"{prefix}.{bn_sub}.running_var"].astype(mx.float32)
    scale = g * mx.rsqrt(var + _BN_EPS)                   # (O,)
    Wf = W * scale.reshape(-1, 1, 1, 1)
    bias = b - mean * scale
    return mx.transpose(Wf, (0, 2, 3, 1)).astype(dtype), bias.astype(dtype)


def _load_block(w: dict, prefix: str, dtype: Any) -> dict:
    return {
        name: (*_fold(w, prefix, conv_sub, bn_sub, dtype), stride, groups)
        for name, conv_sub, bn_sub, stride, groups in _CONVS
    }


def load_fastdvdnet(path: str | Path, dtype: Any = mx.float16) -> dict:
    """Load + BN-fold FastDVDnet weights into {'temp1':..., 'temp2':...}.

    Defaults to fp16 (measured identical denoise quality, ~20% faster on M1 than
    fp32); pass dtype=mx.float32 for the reference path."""
    w = mx.load(str(path))
    return {"temp1": _load_block(w, "temp1", dtype), "temp2": _load_block(w, "temp2", dtype)}


def _relu(x: Any) -> Any:
    return mx.maximum(x, 0)


def _cv(x: Any, conv: tuple) -> Any:
    weight, bias, stride, groups = conv
    y = mx.conv2d(x, weight, stride=stride, padding=1, groups=groups)
    return y if bias is None else y + bias


def _pixelshuffle2(x: Any) -> Any:
    """(N,H,W,4C) -> (N,2H,2W,C), matching torch PixelShuffle(2): output channel
    c gathers input channels [4c:4c+4] as a 2x2 block (idx = i*2 + j)."""
    n, h, w, c4 = x.shape
    c = c4 // 4
    x = x.reshape(n, h, w, c, 2, 2)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))              # (n, h, i, w, j, c)
    return x.reshape(n, h * 2, w * 2, c)


def _denblock(p: dict, in0: Any, in1: Any, in2: Any, nm: Any) -> Any:
    """One DenBlock: 3 frames (+ noise map per frame) -> denoised center (in1)."""
    x = mx.concatenate([in0, nm, in1, nm, in2, nm], axis=-1)      # NHWC, 12 ch
    x0 = _relu(_cv(_relu(_cv(x, p["inc0"])), p["inc3"]))          # InputCvBlock
    x1 = _relu(_cv(_relu(_cv(_relu(_cv(x0, p["d0"])), p["d0c0"])), p["d0c3"]))   # downc0
    x2 = _relu(_cv(_relu(_cv(_relu(_cv(x1, p["d1"])), p["d1c0"])), p["d1c3"]))   # downc1
    u2 = _pixelshuffle2(_cv(_relu(_cv(_relu(_cv(x2, p["u2c0"])), p["u2c3"])), p["u2"]))  # upc2
    u1 = _pixelshuffle2(_cv(_relu(_cv(_relu(_cv(x1 + u2, p["u1c0"])), p["u1c3"])), p["u1"]))  # upc1
    out = _cv(_relu(_cv(x0 + u1, p["o0"])), p["o3"])             # OutputCvBlock
    return in1 - out                                            # residual


def _forward(params: dict, frames: list, nm: Any) -> Any:
    """Full FastDVDnet: 5 frames + noise map -> denoised center frame (NHWC).

    Stage 1 runs temp1 on the three overlapping triplets in a single batched call
    (same weights, independent inputs) instead of three sequential passes.
    """
    t1, t2 = params["temp1"], params["temp2"]
    f0, f1, f2, f3, f4 = frames
    in0 = mx.concatenate([f0, f1, f2], axis=0)           # (3, H, W, 3)
    in1 = mx.concatenate([f1, f2, f3], axis=0)
    in2 = mx.concatenate([f2, f3, f4], axis=0)
    nm3 = mx.broadcast_to(nm, (3, *nm.shape[1:]))
    abc = _denblock(t1, in0, in1, in2, nm3)              # (3, H, W, 3) = [a, b, c]
    return _denblock(t2, abc[0:1], abc[1:2], abc[2:3], nm)


def _reflect_pad_to4(x: Any) -> tuple[Any, int, int]:
    """Reflect-pad NHWC x on bottom/right so H and W are multiples of 4 (the two
    /2 stages need it). Returns (padded, pad_h, pad_w)."""
    _, h, w, _ = x.shape
    ph, pw = (-h) % 4, (-w) % 4
    if ph:
        x = mx.concatenate([x, mx.flip(x[:, h - 1 - ph:h - 1, :, :], axis=1)], axis=1)
    if pw:
        x = mx.concatenate([x, mx.flip(x[:, :, w - 1 - pw:w - 1, :], axis=2)], axis=2)
    return x, ph, pw


class FastDVDnet:
    """FastDVDnet denoiser. Call denoise_center(5 frames, sigma) -> center frame.

    Frames are (H,W,3) float32 in [0,1]; sigma is the noise level in [0,1].
    """

    NUM_FRAMES = 5

    def __init__(self, weights_path: str | Path, dtype: Any = mx.float16):
        self.dtype = dtype
        self.params = load_fastdvdnet(weights_path, dtype=dtype)

    def denoise_center(self, frames: list, sigma: float) -> Any:
        """Denoise the center of a 5-frame window. Frames (H,W,3) f32 in [0,1];
        returns the denoised center frame (H,W,3) f32."""
        if len(frames) != self.NUM_FRAMES:
            raise ValueError(f"FastDVDnet needs {self.NUM_FRAMES} frames, got {len(frames)}")
        h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
        batched = [f[None].astype(self.dtype) for f in frames]              # (1,H,W,3)
        padded = [_reflect_pad_to4(f)[0] for f in batched]
        _, hp, wp, _ = padded[0].shape
        nm = mx.full((1, hp, wp, 1), float(sigma), dtype=self.dtype)
        out = mx.clip(_forward(self.params, padded, nm), 0.0, 1.0)
        return out[0, :h, :w, :].astype(mx.float32)


def _strength_to_sigma(strength: float) -> float:
    """Map a [0,1] denoise strength to FastDVDnet's noise sigma. The network was
    trained on AWGN with sigma_255 in roughly [5, 55], so span that range."""
    s = max(0.0, min(1.0, float(strength)))
    return (5.0 + 50.0 * s) / 255.0


# Weights ship inside the package (~10MB each), so no path/env is needed.
_WEIGHTS_DIR = Path(__file__).resolve().parent / "fastdvdnet_weights"
_VARIANTS = {
    # Trained with clipped noise; stays clean on real footage at moderate sigma.
    "clipped": "model_clipped_noise.safetensors",
    # Plain-AWGN model; over-reconstructs clean content above ~sigma 0.04 and
    # shows a faint pixel-shuffle grid there, so it needs a low strength.
    "standard": "model.safetensors",
}


def default_weights_path(variant: str = "clipped") -> Path:
    """Bundled FastDVDnet weights for the given variant ('clipped' is the default
    - it is the better behaved model on real footage)."""
    return _WEIGHTS_DIR / _VARIANTS[variant]


class FastDvdDenoiser:
    """Streaming FastDVDnet with a proper centered 5-frame window.

    FastDVDnet denoises the center of a 5-frame window, so it needs two future
    frames - a 2-frame lookahead. This is a delay line: feed() buffers a frame and
    returns any frames whose two future neighbours have now arrived; flush() drains
    the tail at end of stream. Only the clip's first/last two frames use a
    reflected window (the standard boundary handling, exactly as the reference);
    every interior frame uses its real [t-2 .. t+2] neighbours - no reflection
    hack (that fed the net out-of-distribution symmetric frames and produced a
    pixel-shuffle moire).

    feed(frame, token) and flush() return lists of (denoised_frame, token). Frames
    are (H,W,3) f32 in [0,1]; token is opaque - the harness threads its per-frame
    context (source pixels, ...) through so delayed outputs stay aligned. reset()
    drops state; call flush() first at a scene cut so the window never bridges it.
    """

    LOOKAHEAD = 2   # (5 - 1) // 2: frames of future context the center needs

    def __init__(self, weights_path: str | Path | None = None, strength: float = 0.5,
                 variant: str = "clipped", dtype: Any = mx.float16):
        wp = Path(weights_path) if weights_path else default_weights_path(variant)
        if not wp.is_file():
            raise FileNotFoundError(
                f"FastDVDnet weights not found at {wp}; bundled weights should ship "
                "with the package, or pass an explicit weights_path."
            )
        self.net = FastDVDnet(wp, dtype=dtype)
        self.sigma = _strength_to_sigma(strength)
        self._reset_state()

    def _reset_state(self) -> None:
        self._buf: list = []    # (frame, token) for absolute indices [_base ..]
        self._base = 0          # absolute index of _buf[0]
        self._received = 0
        self._emitted = 0

    def reset(self) -> None:
        self._reset_state()

    def close(self) -> None:
        pass

    @staticmethod
    def _reflect(a: int, last: int) -> int:
        """Reflect index a into [0, last] (boundary handling at clip ends)."""
        if a < 0:
            a = -a
        if a > last:
            a = 2 * last - a
        return max(0, min(last, a))

    def _emit_one(self, last: int) -> tuple:
        e = self._emitted
        win = [self._buf[self._reflect(e + d, last) - self._base][0]
               for d in (-2, -1, 0, 1, 2)]
        tok = self._buf[e - self._base][1]
        out = self.net.denoise_center(win, self.sigma)
        self._emitted += 1
        while self._base < self._emitted - self.LOOKAHEAD:   # drop no-longer-needed
            self._buf.pop(0)
            self._base += 1
        return out, tok

    def feed(self, frame: Any, token: Any = None) -> list:
        """Buffer one input frame; return [(denoised, token), ...] now ready."""
        self._buf.append((frame, token))
        self._received += 1
        last = self._received - 1
        ready = []
        while last - self._emitted >= self.LOOKAHEAD:
            ready.append(self._emit_one(last))
        return ready

    def flush(self) -> list:
        """Drain remaining frames (end-reflected window) at end of stream / cut."""
        last = self._received - 1
        out = []
        while self._emitted <= last:
            out.append(self._emit_one(last))
        self._reset_state()
        return out
