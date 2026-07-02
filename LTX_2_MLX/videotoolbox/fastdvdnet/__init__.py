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
    block = {
        name: (*_fold(w, prefix, conv_sub, bn_sub, dtype), stride, groups)
        for name, conv_sub, bn_sub, stride, groups in _CONVS
    }
    _pad_inc_for_gates(block)
    return block


def _pad_inc_for_gates(block: dict) -> None:
    """Zero-pad inc0's grouped conv outputs onto MLX's grouped implicit-GEMM path.

    inc0 (12 -> 90, groups=3) has 30 output channels per group; the grouped gate
    needs O_per_group%16==0 (mlx conv.cpp), so it fell to the explicit-grouped
    fallback (~4x slower measured at 480p). Pad each group's filters 30 -> 32 with
    zeros (+bias zeros) and insert matching zero input columns into inc3 at the pad
    positions. Exact: the padded channels are 0 (zero filters, zero bias), relu(0)=0,
    and inc3's matching columns are 0 -- nothing changes but the kernel taken."""
    W0, b0, s0, g0 = block["inc0"]                     # (O,3,3,I/g), groups g0
    og = W0.shape[0] // g0                             # out channels per group (30)
    if g0 <= 1 or og % 16 == 0:
        return
    pad = 16 - og % 16                                 # -> 32 per group
    zr = mx.zeros((pad, *W0.shape[1:]), dtype=W0.dtype)
    zb = mx.zeros((pad,), dtype=b0.dtype)
    W0p = mx.concatenate(
        [t for g in range(g0) for t in (W0[g * og:(g + 1) * og], zr)], axis=0)
    b0p = mx.concatenate(
        [t for g in range(g0) for t in (b0[g * og:(g + 1) * og], zb)], axis=0)
    block["inc0"] = (W0p, b0p, s0, g0)
    W3, b3, s3, g3 = block["inc3"]                     # (32,3,3,g0*og)
    zc = mx.zeros((*W3.shape[:3], pad), dtype=W3.dtype)
    W3p = mx.concatenate(
        [t for g in range(g0) for t in (W3[..., g * og:(g + 1) * og], zc)], axis=-1)
    block["inc3"] = (W3p, b3, s3, g3)


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


def _reflect_pad_to4(x: Any) -> tuple[Any, int, int]:
    """Reflect-pad NHWC x on bottom/right so H and W are multiples of 4 (the two
    /2 stages need it). Returns (padded, pad_h, pad_w)."""
    _, h, w, _ = x.shape
    ph, pw = (-h) % 4, (-w) % 4
    if ph:   # [::-1] reverses the mirror slice (MLX has no mx.flip)
        x = mx.concatenate([x, x[:, h - 1 - ph:h - 1, :, :][:, ::-1, :, :]], axis=1)
    if pw:
        x = mx.concatenate([x, x[:, :, w - 1 - pw:w - 1, :][:, :, ::-1, :]], axis=2)
    return x, ph, pw


class FastDVDnet:
    """FastDVDnet denoiser. Call denoise_center(5 frames, sigma) -> center frame.

    Frames are (H,W,3) float32 in [0,1]; sigma is the noise level in [0,1].
    """

    NUM_FRAMES = 5

    def __init__(self, weights_path: str | Path, dtype: Any = mx.float16):
        self.dtype = dtype
        self.params = load_fastdvdnet(weights_path, dtype=dtype)
        # Compile the two stages separately (both pure). Keeping temp1 and temp2
        # as distinct callables lets the streaming denoiser cache temp1 outputs
        # across frames - each stage-1 estimate g(i) is reused by 3 consecutive
        # output frames, so caching cuts temp1 from 3 passes/frame to 1. mx.compile
        # also fuses the elementwise ops between convs and removes Python dispatch;
        # it re-traces per input shape (once per clip resolution) and caches.
        p = self.params
        self._t1 = mx.compile(lambda a, b, c, nm: _denblock(p["temp1"], a, b, c, nm))
        self._t2 = mx.compile(lambda a, b, c, nm: _denblock(p["temp2"], a, b, c, nm))

    def temp1_step(self, prev: Any, cur: Any, nxt: Any, nm: Any) -> Any:
        """temp1 on one triplet of padded frames -> stage-1 estimate of the center."""
        return self._t1(prev, cur, nxt, nm)

    def temp2_step(self, a: Any, b: Any, c: Any, nm: Any) -> Any:
        """temp2 fusing three stage-1 estimates -> denoised (padded) center frame."""
        return self._t2(a, b, c, nm)

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
        g0 = self.temp1_step(padded[0], padded[1], padded[2], nm)
        g1 = self.temp1_step(padded[1], padded[2], padded[3], nm)
        g2 = self.temp1_step(padded[2], padded[3], padded[4], nm)
        out = mx.clip(self.temp2_step(g0, g1, g2, nm), 0.0, 1.0)
        return out[0, :h, :w, :].astype(mx.float32)


def _strength_to_sigma(strength: float) -> float:
    """Map a [0,1] denoise strength to FastDVDnet's noise sigma. The network was
    trained on AWGN with sigma_255 in roughly [5, 55], so span that range."""
    s = max(0.0, min(1.0, float(strength)))
    return (5.0 + 50.0 * s) / 255.0


# Weights ship inside the package (~10MB each), so no path/env is needed.
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
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
    """Streaming FastDVDnet with a proper centered 5-frame window, caching temp1.

    FastDVDnet's output for frame t is temp2(g(t-1), g(t), g(t+1)) where the
    stage-1 estimate g(i) = temp1(frame i-1, i, i+1). Consecutive output frames
    overlap in g, so each g(i) is cached and computed once: that is one temp1 pass
    per output frame instead of three (~1.9x faster), bit-identical to denoising a
    fresh 5-frame window each time.

    A 2-frame lookahead is still needed (g(t+1) needs frame t+2), so this is a
    delay line: feed() buffers a frame and returns any outputs whose inputs have
    all arrived; flush() drains the tail. Only the clip's first/last two frames use
    a reflected window (the standard boundary handling, exactly as the reference);
    every interior frame uses its real neighbours - no reflection hack (that fed
    the net out-of-distribution symmetric frames and produced a pixel-shuffle
    moire).

    feed(frame, token) and flush() return lists of (denoised_frame, token). Frames
    are (H,W,3) f32 in [0,1]; token is opaque - the harness threads its per-frame
    context through so delayed outputs stay aligned. reset() drops state; call
    flush() first at a scene cut so the window never bridges it.
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
        self._buf: list = []    # (padded_frame, token) for abs indices [_base ..]
        self._base = 0          # absolute index of _buf[0]
        self._g: dict = {}      # abs index -> g(i) = temp1 estimate (cached)
        self._nm: Any = None    # padded noise map (set on first frame)
        self._hw: Any = None    # original (h, w) for unpad
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

    def _frame(self, i: int, last: int) -> Any:
        return self._buf[self._reflect(i, last) - self._base][0]

    def _g_at(self, i: int, last: int) -> Any:
        """temp1 estimate g(i), computed once and cached."""
        g = self._g.get(i)
        if g is None:
            g = self.net.temp1_step(
                self._frame(i - 1, last), self._frame(i, last), self._frame(i + 1, last), self._nm)
            mx.eval(g)   # materialize once: each g feeds 3 temp2 calls, don't recompute
            self._g[i] = g
        return g

    def _emit_one(self, last: int) -> tuple:
        t = self._emitted
        out = self.net.temp2_step(
            self._g_at(t - 1, last), self._g_at(t, last), self._g_at(t + 1, last), self._nm)
        h, w = self._hw
        out = mx.clip(out, 0.0, 1.0)[0, :h, :w, :].astype(mx.float32)
        tok = self._buf[t - self._base][1]
        self._emitted += 1
        keep = self._emitted - 2                          # oldest index still needed
        while self._base < keep and self._buf:
            self._buf.pop(0)
            self._base += 1
        for k in [k for k in self._g if k < keep]:
            del self._g[k]
        return out, tok

    def feed(self, frame: Any, token: Any = None) -> list:
        """Buffer one input frame; return [(denoised, token), ...] now ready."""
        padded = _reflect_pad_to4(frame[None].astype(self.net.dtype))[0]
        if self._nm is None:
            _, hp, wp, _ = padded.shape
            self._nm = mx.full((1, hp, wp, 1), float(self.sigma), dtype=self.net.dtype)
            self._hw = (int(frame.shape[0]), int(frame.shape[1]))
        self._buf.append((padded, token))
        self._received += 1
        last = self._received - 1
        ready = []
        while last - self._emitted >= self.LOOKAHEAD:
            ready.append(self._emit_one(last))
        return ready

    def flush(self) -> list:
        """Drain remaining frames (end-reflected) at end of stream / cut."""
        last = self._received - 1
        out = []
        while self._emitted <= last:
            out.append(self._emit_one(last))
        self._reset_state()
        return out
