"""Pre-upscale denoisers for the VSR harness.

Two options, both run at native resolution BEFORE super-resolution (the correct
order: SR synthesizes/amplifies high-frequency detail, so it bakes in noise it's
fed - clean first):

- SpatialDenoiser: per-frame CoreImage CINoiseReduction. No temporal state, cheap.
- McTemporalDenoiser: motion-compensated temporal denoise built on VideoToolbox
  optical flow (VTOpticalFlow, GPU, supported on M1 unlike the AVE-based
  VTTemporalNoiseFilter). Recursive/causal: it keeps the previous denoised
  frame, computes optical flow to it, warps it into alignment, and blends - more
  where the warp matches (static regions), less where it doesn't (occlusion /
  fast motion), so moving edges don't ghost.

Interface is MLX-array in / MLX-array out: (H,W,3) float32 RGB in [0, 1].
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mlx.core as mx

from . import pixel_buffers as _pb
from ._compat import Foundation, Quartz, autorelease_pool, require_pyobjc, vt
from .vsr import _suppress_native_stderr

# Max optical-flow calls in flight at once. Each flow is an ANE/IOKit dispatch
# (fixed ~17 ms, latency-bound) - a couple overlap well, but the kernel-side
# dispatch cost (System CPU) climbs with every concurrent flow. 2 measured as
# the sweet spot: it captures most of the overlap (~1.6x for two) while keeping
# System CPU low; beyond it the per-flow wall-time gain shrinks fast, and past
# ~5 it oversubscribes and spills to CPU. A large --mc-window still computes all
# its references, just no more than this many flows concurrently.
_MAX_CONCURRENT_FLOWS = 2

# Cache the sampling grid per resolution; it's constant across frames.
_GRID: dict[tuple[int, int], tuple[Any, Any]] = {}


def _grid(h: int, w: int) -> tuple[Any, Any]:
    g = _GRID.get((h, w))
    if g is None:
        ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
        g = (ys.astype(mx.float32), xs.astype(mx.float32))
        _GRID[(h, w)] = g
    return g


def warp(img: Any, flow: Any) -> Any:
    """Backward-warp an (H,W,C) f32 image by an (H,W,2) px flow field.

    out[p] = bilinear_sample(img, p + flow[p]). Used to pull a reference frame
    into alignment with the current one. Out-of-bounds samples clamp to edge.
    """
    h, w, c = img.shape
    ys, xs = _grid(h, w)
    sx = mx.clip(xs + flow[..., 0], 0, w - 1)
    sy = mx.clip(ys + flow[..., 1], 0, h - 1)
    x0 = mx.floor(sx).astype(mx.int32)
    y0 = mx.floor(sy).astype(mx.int32)
    x1 = mx.clip(x0 + 1, 0, w - 1)
    y1 = mx.clip(y0 + 1, 0, h - 1)
    wx = (sx - x0.astype(mx.float32))[..., None]
    wy = (sy - y0.astype(mx.float32))[..., None]
    flat = img.reshape(h * w, c)

    def g(yy: Any, xx: Any) -> Any:
        return flat[(yy * w + xx).reshape(-1)].reshape(h, w, c)

    top = g(y0, x0) * (1 - wx) + g(y0, x1) * wx
    bot = g(y1, x0) * (1 - wx) + g(y1, x1) * wx
    return top * (1 - wy) + bot * wy


class SpatialDenoiser:
    """Per-frame CoreImage CINoiseReduction. Spatial only; no temporal state."""

    def __init__(self, strength: float = 0.5):
        require_pyobjc()
        # CINoiseReduction's inputNoiseLevel is ~0.0-0.1 in practice; map strength
        # onto a gentle range so strength=0.5 is a moderate clean.
        self.noise_level = 0.01 + 0.04 * float(strength)
        self.sharpness = 0.4

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def denoise(self, rgb_f32: Any) -> Any:
        # fp16 in / fp16 out: feed CoreImage a half-float CIImage and render back
        # to half-float (kCIFormatRGBAh), so no 8-bit quantization round trip.
        h, w = int(rgb_f32.shape[0]), int(rgb_f32.shape[1])
        rgba = mx.concatenate(
            [rgb_f32.astype(mx.float16), mx.ones((h, w, 1), dtype=mx.float16)], axis=-1,
        )
        src = memoryview(mx.contiguous(rgba)).cast("B")
        data = Foundation.NSData.dataWithBytes_length_(src, len(src))
        ci = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
            data, w * 8, (w, h), Quartz.kCIFormatRGBAh, _pb.srgb_colorspace(),
        )
        filt = Quartz.CIFilter.filterWithName_("CINoiseReduction")
        filt.setValue_forKey_(ci, "inputImage")
        filt.setValue_forKey_(float(self.noise_level), "inputNoiseLevel")
        filt.setValue_forKey_(float(self.sharpness), "inputSharpness")
        out = filt.valueForKey_("outputImage")
        buf = bytearray(w * h * 8)
        _pb.ci_context().render_toBitmap_rowBytes_bounds_format_colorSpace_(
            out, buf, w * 8, ((0, 0), (w, h)), Quartz.kCIFormatRGBAh, _pb.srgb_colorspace(),
        )
        rgba_out = mx.array(memoryview(buf)).view(mx.float16).reshape(h, w, 4)
        return mx.contiguous(rgba_out[..., :3]).astype(mx.float32)


def _box_mean(x: Any, k: int) -> Any:
    """KxK box mean of an (H,W,C) array, same size out, via a depthwise grouped
    conv2d (no transposes: each channel convolved with its own box kernel)."""
    c = x.shape[2]
    ker = mx.full((c, k, k, 1), 1.0 / (k * k), dtype=x.dtype)
    return mx.conv2d(x[None], ker, stride=1, padding=k // 2, groups=c)[0]


class McTemporalDenoiser:
    """Motion-compensated temporal denoise via VTOpticalFlow, with optional
    anti-ghosting refinements that compose:

    - window=0 (default): recursive/IIR - blends the current frame with the
      previous *output*, warped into alignment. Strongest noise reduction but
      ghosts have a long (recursive) lifetime.
    - window=N>=1: causal FIR - averages the current frame with the last N
      *input* frames, each warped into alignment. Bounded ghost lifetime
      (a bad warp ages out in <= N frames) at the cost of N flow computes/frame.
      Causal (past frames only); no lookahead.

    Optional gates, each multiplied into the per-reference blend weight:
    - clamp:      neighborhood color clamping (TAA variance-clip): clamp the
      warped reference into mean +/- gamma*std of the current frame's local
      window, so history that disagrees with the local appearance can't ghost.
    - occlusion:  forward-backward flow consistency - reject history where the
      forward and backward flow don't round-trip (occlusion / bad flow).
    - confidence: down-weight where the flow magnitude is large (fast motion).

    Interface: (H,W,3) float32 RGB in [0,1] in/out. Apply reset() at scene cuts.
    """

    def __init__(
        self, width: int, height: int, strength: float = 0.5,
        window: int = 0, clamp: bool = False, occlusion: bool = False,
        confidence: bool = False,
    ):
        require_pyobjc()
        self.w, self.h = int(width), int(height)
        self.strength = float(strength)   # max blend weight toward a reference
        self.window = max(0, int(window))
        self.clamp = bool(clamp)
        self.occlusion = bool(occlusion)
        self.confidence = bool(confidence)
        # Tunables (sensible fixed defaults; strength is the user knob).
        self.sigma = 0.06        # residual rejection scale (luma, [0,1])
        self.clamp_k = 5         # neighborhood window for color clamping
        self.clamp_gamma = 1.25  # box half-width in std units
        self.occ_tau = 1.5       # FB-consistency tolerance (pixels)
        self.conf_scale = 10.0   # flow magnitude (px) at which confidence ~1/e
        cls = vt.VTOpticalFlowConfiguration
        if not cls.isSupported():
            raise SystemExit("VTOpticalFlow is not supported on this device.")
        # One flow worker (session + buffers) per window reference, so the
        # references' flows run concurrently. VTOpticalFlow is fixed-overhead /
        # latency-bound (~17 ms at any resolution) and releases the GIL during
        # the call, so N parallel sessions overlap (~1.6x for 2) rather than
        # serialize - the only real lever for the window's cost.
        self._src_attrs: Any = None
        self._dst_attrs: Any = None
        self._workers = [self._make_worker(cls) for _ in range(max(1, self.window))]
        # Single shared "current" buffer: every flow reads the same current frame,
        # so we upload it once per frame instead of once per reference.
        self._curr_buf = _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._src_attrs)
        # Bounded thread pool so concurrent flows never oversubscribe (capped
        # well below the window for large windows). None in recursive mode.
        self._pool = (
            ThreadPoolExecutor(max_workers=min(self.window, _MAX_CONCURRENT_FLOWS))
            if self.window > 1 else None
        )
        self._prev: Any = None       # previous OUTPUT frame (recursive mode)
        self._hist: list[Any] = []   # last N INPUT frames, oldest first (FIR mode)
        self._idx = 0

    def _make_worker(self, cls: Any) -> dict:
        cfg = cls.alloc().initWithFrameWidth_frameHeight_qualityPrioritization_revision_(
            self.w, self.h,
            vt.VTOpticalFlowConfigurationQualityPrioritizationNormal,
            cls.defaultRevision(),
        )
        if cfg is None:
            raise RuntimeError(f"VTOpticalFlow config init returned nil for {self.w}x{self.h}")
        proc = vt.VTFrameProcessor.alloc().init()
        with _suppress_native_stderr():
            ok, err = proc.startSessionWithConfiguration_error_(cfg, None)
        if not ok:
            raise RuntimeError(f"VTOpticalFlow startSession failed: {err}")
        self._src_attrs = dict(cfg.sourcePixelBufferAttributes() or {})
        self._dst_attrs = dict(cfg.destinationPixelBufferAttributes() or {})
        return {
            "proc": proc,
            # Per-worker source (the reference) + flow outputs. The "next" frame
            # (current) is a single shared buffer (see _curr_buf) - the flow only
            # reads it, so concurrent reads are fine and we upload it once/frame.
            "ref": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._src_attrs),
            "fwd": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._dst_attrs),
            "bwd": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._dst_attrs),
        }

    def reset(self) -> None:
        """Drop temporal history (call at scene cuts)."""
        self._prev = None
        self._hist = []

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for wk in self._workers:
            if wk["proc"] is not None:
                wk["proc"].endSession()
                wk["proc"] = None

    def _upload(self, rgb_f32: Any, buf: Any) -> None:
        h, w = int(rgb_f32.shape[0]), int(rgb_f32.shape[1])
        rgba = mx.concatenate(
            [rgb_f32.astype(mx.float16), mx.ones((h, w, 1), mx.float16)], axis=-1,
        )
        _pb.write_fp16_rgba(rgba, buf)

    def _read_flow(self, pb: Any) -> Any:
        Quartz.CVPixelBufferLockBaseAddress(pb, 1)
        try:
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
            base = Quartz.CVPixelBufferGetBaseAddress(pb)
            raw = mx.array(memoryview(base.as_buffer(self.h * bpr)))
            flow = raw.view(mx.float16).reshape(self.h, bpr // 2)[:, : self.w * 2].reshape(
                self.h, self.w, 2,
            ).astype(mx.float32)
            mx.eval(flow)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
        return flow

    def _compute_flows(self, curr: Any, refs: list[Any]) -> list[tuple[Any, Any]]:
        """Optical flow of each reference -> current. The references run on
        separate sessions concurrently (the GIL is released during the VT call,
        so they overlap). Uploads/reads are MLX and stay on the main thread;
        only the processWithParameters calls are threaded. Returns
        [(forwardFlow, backwardFlow_or_None), ...] as (H,W,2) px MLX arrays.
        """
        self._upload(curr, self._curr_buf)              # once, shared by all flows
        jobs = []
        for j, ref in enumerate(refs):
            wk = self._workers[j]
            self._upload(ref, wk["ref"])
            sf = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                wk["ref"], _pb.frame_pts(0, 24.0),
            )
            nf = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                self._curr_buf, _pb.frame_pts(1, 24.0),
            )
            fo = vt.VTFrameProcessorOpticalFlow.alloc().initWithForwardFlow_backwardFlow_(
                wk["fwd"], wk["bwd"],
            )
            pa = vt.VTOpticalFlowParameters.alloc().initWithSourceFrame_nextFrame_submissionMode_destinationOpticalFlow_(
                sf, nf, vt.VTOpticalFlowParametersSubmissionModeRandom, fo,
            )
            jobs.append((wk["proc"], pa))

        errs: list[Any] = [None] * len(jobs)

        def run(j: int) -> None:
            with autorelease_pool():
                ok, err = jobs[j][0].processWithParameters_error_(jobs[j][1], None)
                if not ok:
                    errs[j] = err

        if self._pool is None or len(jobs) == 1:
            for j in range(len(jobs)):
                run(j)
        else:
            list(self._pool.map(run, range(len(jobs))))   # <= _MAX_CONCURRENT_FLOWS in flight
        for e in errs:
            if e is not None:
                raise RuntimeError(f"VTOpticalFlow process failed: {e}")

        out = []
        for j in range(len(refs)):
            wk = self._workers[j]
            out.append((
                self._read_flow(wk["fwd"]),
                self._read_flow(wk["bwd"]) if self.occlusion else None,
            ))
        return out

    def _weight(self, curr: Any, warped: Any, fwd: Any, bwd: Any) -> Any:
        """Per-pixel blend weight (H,W,1) toward `warped`, combining the enabled
        gates: residual match, FB-consistency occlusion, motion confidence."""
        resid = mx.mean(mx.abs(curr - warped), axis=-1, keepdims=True)
        w = self.strength * mx.exp(-((resid / self.sigma) ** 2))
        if self.occlusion:
            # Round-trip: curr pixel p -> ref at p+bwd[p], then fwd should return
            # it; |bwd + fwd(at p+bwd)| ~ 0 when consistent, large at occlusion.
            fwd_at = warp(fwd, bwd)
            fb = mx.sqrt(mx.sum((bwd + fwd_at) ** 2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((fb / self.occ_tau) ** 2))
        if self.confidence:
            mag = mx.sqrt(mx.sum(fwd ** 2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((mag / self.conf_scale) ** 2))
        return w

    def denoise(self, rgb_f32: Any) -> Any:
        refs = ([self._prev] if self._prev is not None else []) if self.window == 0 \
            else list(self._hist)
        if not refs:
            self._remember(rgb_f32, rgb_f32)
            return rgb_f32
        lo = hi = None
        if self.clamp:
            mean = _box_mean(rgb_f32, self.clamp_k)
            var = mx.maximum(_box_mean(rgb_f32 * rgb_f32, self.clamp_k) - mean * mean, 0.0)
            std = mx.sqrt(var)
            lo, hi = mean - self.clamp_gamma * std, mean + self.clamp_gamma * std
        flows = self._compute_flows(rgb_f32, refs)      # references run concurrently
        acc = rgb_f32                                   # current frame, weight 1
        wsum = mx.ones((self.h, self.w, 1))
        for ref, (fwd, bwd) in zip(refs, flows, strict=True):
            warped = warp(ref, -fwd)
            if self.clamp:
                warped = mx.clip(warped, lo, hi)
            w = self._weight(rgb_f32, warped, fwd, bwd)
            acc = acc + w * warped
            wsum = wsum + w
        out = mx.clip(acc / wsum, 0.0, 1.0)
        mx.eval(out)
        self._remember(rgb_f32, out)
        return out

    def _remember(self, curr: Any, out: Any) -> None:
        if self.window == 0:
            self._prev = out                            # recursive: keep output
        else:
            self._hist.append(curr)                     # FIR: keep input frames
            if len(self._hist) > self.window:
                self._hist.pop(0)
        self._idx += 1


def luma_chroma_blend(orig: Any, new: Any, a_luma: float, a_chroma: float) -> Any:
    """Recombine `orig` and `new` (both (H,W,3) RGB in [0,1]) with separate blend
    strengths for luma and chroma, BT.601 full-range: the output luma is
    lerp(orig, new, a_luma) and the chroma is lerp(orig, new, a_chroma). a=1 takes the
    new (denoised) value, a=0 keeps the original; a_luma=a_chroma=1 returns `new` exactly.
    Computed in float32 -- the YCbCr round-trip divides by the chroma scales, which fp16
    coarsens."""
    Kr, Kg, Kb, Cb, Cr = 0.299, 0.587, 0.114, 1.772, 1.402
    o = orig.astype(mx.float32)
    n = new.astype(mx.float32)

    def _yc(x):
        y = Kr * x[..., 0:1] + Kg * x[..., 1:2] + Kb * x[..., 2:3]
        return y, (x[..., 2:3] - y) / Cb, (x[..., 0:1] - y) / Cr     # y, cb, cr

    yo, cbo, cro = _yc(o)
    yn, cbn, crn = _yc(n)
    y = yo + a_luma * (yn - yo)
    cb = cbo + a_chroma * (cbn - cbo)
    cr = cro + a_chroma * (crn - cro)
    r = y + Cr * cr
    b = y + Cb * cb
    g = (y - Kr * r - Kb * b) / Kg
    return mx.clip(mx.concatenate([r, g, b], axis=-1), 0.0, 1.0)


class LumaChromaDenoiser:
    """Wrap any harness denoiser to apply separate luma/chroma blend strengths between
    its input and output -- e.g. denoise chroma hard (a_chroma=1) while keeping luma
    texture (a_luma<1), the split a single joint RGB sigma cannot do. The base still
    denoises RGB jointly; this only re-weights its effect per channel group on the way
    out.

    Threads the input frame through the base's token so delay-line denoisers (FastDVDnet)
    still pair each delayed output with its own input; per-frame denoisers (spatial / mc)
    blend in step. Presents the feed/flush interface either way."""

    def __init__(self, base: Any, luma_strength: float = 1.0, chroma_strength: float = 1.0):
        self._base = base
        self._al = float(luma_strength)
        self._ac = float(chroma_strength)

    def reset(self) -> None:
        self._base.reset()

    def close(self) -> None:
        if hasattr(self._base, "close"):
            self._base.close()

    def _blend(self, orig: Any, den: Any) -> Any:
        return luma_chroma_blend(orig, den, self._al, self._ac)

    def feed(self, rgb: Any, token: Any = None) -> list:
        if hasattr(self._base, "feed"):
            return [(self._blend(o, d), t)
                    for d, (o, t) in self._base.feed(rgb, token=(rgb, token))]
        return [(self._blend(rgb, self._base.denoise(rgb)), token)]

    def flush(self) -> list:
        if hasattr(self._base, "flush"):
            return [(self._blend(o, d), t) for d, (o, t) in self._base.flush()]
        return []
