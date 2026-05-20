#!/usr/bin/env python3
"""VAE-decode (or read MP4) and pump frames through VideoToolbox Super
Resolution. Writes pre/post PNG frames plus a side-by-side comparison MP4.

Usage
-----
    # Latent path: VAE-decode the NPZ sidecar, then VSR it.
    scripts/vsr_harness.py --latent run.npz --weights $LTX_DEFAULT_WEIGHTS_PATH \
        --scale 2 --quality fast --output-dir outputs/vsr/run1

    # Video path: skip VAE; VSR an existing clip.
    scripts/vsr_harness.py --video clip.mp4 \
        --scale 4 --quality balanced --output-dir outputs/vsr/run2

Quality / scale combinations (macOS 26 VideoToolbox)
----------------------------------------------------
    fast      VTLowLatencySuperResolutionScalerConfiguration.
              scale=2 only, input must fit between 96x96 and 960x960.
              Per-frame, no temporal context, no on-demand model.
    balanced  VTSuperResolutionScalerConfiguration, InputType=Video.
              scale=4 only. Downloadable model (auto-fetched on first use).
              Uses previous source + previous output for temporal coherence;
              recommended for moving content.
    high      VTSuperResolutionScalerConfiguration, InputType=Image.
              scale=4 only. Per-frame run of the HQ model; no temporal
              context. Slowest path.

The VAE decoder defaults track scripts/generate.py's happy path
(native-conv3d + zero spatial padding) via the encode_modes_harness
helpers, and chunks are cast to uint8 inside MLX so peak RAM stays
bounded on long clips.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

import AVFoundation as av
import CoreAudio
import CoreMedia
import Foundation
import libdispatch
import Quartz
import VideoToolbox as vt


sys.path.insert(0, str(Path(__file__).parent.parent))


NATIVE_FPS = 24.0


# ---------------------------------------------------------------------------
# CVPixelBuffer <-> numpy via CoreImage. VSR's source/destination pixel formats
# (NV12 for low-latency, half-float RGBA for high-quality) are picked by the
# configuration, not by us, so CIImage + CIContext is the cleanest bridge.
# ---------------------------------------------------------------------------

_ci_context: Any = None
_srgb: Any = None


def _ci_context_singleton() -> Any:
    global _ci_context
    if _ci_context is None:
        _ci_context = Quartz.CIContext.contextWithOptions_(None)
    return _ci_context


def _srgb_colorspace() -> Any:
    global _srgb
    if _srgb is None:
        _srgb = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    return _srgb


def _resolve_pixel_format(attrs: dict) -> int:
    fmt = attrs.get("PixelFormatType")
    if not isinstance(fmt, int) and hasattr(fmt, "__getitem__"):
        fmt = int(fmt[0])
    return int(fmt)


def _make_pixel_buffer_from_attrs(w: int, h: int, attrs: dict) -> Any:
    fmt = _resolve_pixel_format(attrs)
    err, pb = Quartz.CVPixelBufferCreate(None, w, h, fmt, attrs, None)
    if err != 0:
        raise RuntimeError(
            f"CVPixelBufferCreate({w}x{h}, fmt={fmt:#x}) failed: status={err}"
        )
    return pb


def _write_fp16_rgba_to_buffer(rgba_fp16: np.ndarray, pb: Any) -> None:
    h, w, _ = rgba_fp16.shape
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        mv = base.as_buffer(h * bpr)
        if bpr == w * 8:
            mv[:] = rgba_fp16.tobytes()
        else:
            dst = np.frombuffer(mv, dtype=np.uint8).reshape(h, bpr)
            dst[:, : w * 8] = rgba_fp16.view(np.uint8).reshape(h, w * 8)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)


def _rgb_to_pixel_buffer(frame: np.ndarray, pb: Any) -> None:
    """Upload `frame` into `pb`. Dispatches by frame dtype and pb format.

    NV12 source (LowLatency VSR): frame must be (H,W,3) uint8 RGB. Uses
    CoreImage so sRGB -> YUV BT.709 color conversion is correct.

    RGBAHalf source (HighQuality VSR): two accepted input shapes:
      - (H,W,4) fp16 RGBA — direct memcpy. Best path (no quantization).
      - (H,W,3) uint8 RGB — promoted to fp16 inline (legacy / --video path).
    """
    pix_fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)

    if pix_fmt == 1380411457:  # 'RGhA' kCVPixelFormatType_64RGBAHalf
        if frame.dtype == np.float16:
            _write_fp16_rgba_to_buffer(frame, pb)
            return
        h, w = frame.shape[:2]
        rgba_fp16 = np.empty((h, w, 4), dtype=np.float16)
        rgba_fp16[..., 0:3] = frame.astype(np.float16) * np.float16(1.0 / 255.0)
        rgba_fp16[..., 3] = np.float16(1.0)
        _write_fp16_rgba_to_buffer(rgba_fp16, pb)
        return

    # NV12 path — uint8 RGB only.
    h, w, _ = frame.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0:3] = frame
    rgba[..., 3] = 255
    data = Foundation.NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
    ci_image = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data, w * 4, (w, h), Quartz.kCIFormatRGBA8, _srgb_colorspace()
    )
    _ci_context_singleton().render_toCVPixelBuffer_(ci_image, pb)


def _pixel_buffer_to_rgb(pb: Any) -> np.ndarray:
    w = Quartz.CVPixelBufferGetWidth(pb)
    h = Quartz.CVPixelBufferGetHeight(pb)
    ci_image = Quartz.CIImage.alloc().initWithCVPixelBuffer_(pb)
    buf = bytearray(w * h * 4)
    _ci_context_singleton().render_toBitmap_rowBytes_bounds_format_colorSpace_(
        ci_image, buf, w * 4, ((0, 0), (w, h)),
        Quartz.kCIFormatRGBA8, _srgb_colorspace(),
    )
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return rgba[..., :3].copy()


# ---------------------------------------------------------------------------
# VSR session
# ---------------------------------------------------------------------------

def _wait_for_model_download(config: Any) -> None:
    status = config.configurationModelStatus()
    if status == vt.VTSuperResolutionScalerConfigurationModelStatusReady:
        return
    print(f"VSR model not ready (status={status}); requesting download...")
    done = threading.Event()
    err_box: list[Any] = [None]

    def completion(error):
        err_box[0] = error
        done.set()

    config.downloadConfigurationModelWithCompletionHandler_(completion)
    last_reported = -1
    while not done.is_set():
        pct = int(config.configurationModelPercentageAvailable() * 100)
        if pct // 5 != last_reported // 5:
            print(f"  model download: {pct}%")
            last_reported = pct
        done.wait(timeout=0.5)
    if err_box[0] is not None:
        raise RuntimeError(f"VSR model download failed: {err_box[0]}")
    print("  model download: done")


def _validate_combination(w: int, h: int, scale: int, quality: str) -> None:
    if quality == "fast":
        cls = vt.VTLowLatencySuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("LowLatency VSR not supported on this device.")
        ok = list(cls.supportedScaleFactorsForFrameWidth_frameHeight_(w, h))
        if not ok:
            mn = cls.minimumDimensions()
            mx_dims = cls.maximumDimensions()
            raise SystemExit(
                f"--quality fast does not support {w}x{h} input. "
                f"Allowed: {mn.width}x{mn.height} to {mx_dims.width}x{mx_dims.height}."
            )
        if float(scale) not in [float(s) for s in ok]:
            raise SystemExit(
                f"--quality fast at {w}x{h} supports scale={ok}, got --scale {scale}."
            )
    else:
        cls = vt.VTSuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("High-quality VSR not supported on this device.")
        ok = [int(s) for s in cls.supportedScaleFactors()]
        if scale not in ok:
            raise SystemExit(
                f"--quality {quality} supports scale={ok}, got --scale {scale}. "
                f"Use --quality fast for scale=2."
            )


class VsrSession:
    """Per-frame VSR with the previous-frame chain held across chunks."""

    def __init__(self, in_w: int, in_h: int, scale: int, quality: str):
        _validate_combination(in_w, in_h, scale, quality)
        self.in_w, self.in_h = in_w, in_h
        self.out_w, self.out_h = in_w * scale, in_h * scale
        self.quality = quality

        if quality == "fast":
            self.config = vt.VTLowLatencySuperResolutionScalerConfiguration.alloc(
            ).initWithFrameWidth_frameHeight_scaleFactor_(in_w, in_h, float(scale))
            if self.config is None:
                raise RuntimeError("LowLatency VSR config init returned nil")
        else:
            input_type = (
                vt.VTSuperResolutionScalerConfigurationInputTypeVideo
                if quality == "balanced"
                else vt.VTSuperResolutionScalerConfigurationInputTypeImage
            )
            cls = vt.VTSuperResolutionScalerConfiguration
            self.config = cls.alloc().initWithFrameWidth_frameHeight_scaleFactor_inputType_usePrecomputedFlow_qualityPrioritization_revision_(
                in_w, in_h, scale, input_type, False,
                vt.VTSuperResolutionScalerConfigurationQualityPrioritizationNormal,
                cls.defaultRevision(),
            )
            if self.config is None:
                raise RuntimeError("High-quality VSR config init returned nil")
            _wait_for_model_download(self.config)

        self.processor = vt.VTFrameProcessor.alloc().init()
        ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(
                f"VTFrameProcessor.startSessionWithConfiguration_error_ failed: {err}"
            )

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        print(
            f"VSR session ready ({quality}, {in_w}x{in_h} -> {self.out_w}x{self.out_h}, "
            f"src fmt {_resolve_pixel_format(self.src_attrs):#x}, "
            f"dst fmt {_resolve_pixel_format(self.dst_attrs):#x})"
        )

        self._prev_src_frame = None
        self._prev_dst_frame = None

        # Local pool for src buffers — VSR's input format is fixed by the
        # config (NV12 or RGBAHalf). Two are in flight at any time (current +
        # prev_src_frame for temporal coherence) so the pool size auto-grows
        # to ~2-3 IOSurface-backed buffers.
        err, src_pool = Quartz.CVPixelBufferPoolCreate(None, None, self.src_attrs, None)
        if err != 0 or src_pool is None:
            print(f"  warning: CVPixelBufferPoolCreate(src) failed ({err}); "
                  f"falling back to per-frame allocation")
            self._src_pool = None
        else:
            self._src_pool = src_pool

        # Pool to draw dst CVPixelBuffers from. Default = None (fresh alloc per
        # frame). Wire to AVAssetWriter's adaptor pool to get true zero-copy
        # from VSR output → encoder.
        self._dst_pool: Any = None

    def use_dst_pool(self, pool: Any) -> None:
        self._dst_pool = pool

    def _make_src_buffer(self) -> Any:
        if self._src_pool is not None:
            err, pb = Quartz.CVPixelBufferPoolCreatePixelBuffer(None, self._src_pool, None)
            if err == 0 and pb is not None:
                return pb
        return _make_pixel_buffer_from_attrs(self.in_w, self.in_h, self.src_attrs)

    def _make_dst_buffer(self) -> Any:
        if self._dst_pool is not None:
            err, pb = Quartz.CVPixelBufferPoolCreatePixelBuffer(None, self._dst_pool, None)
            if err == 0 and pb is not None:
                return pb
        return _make_pixel_buffer_from_attrs(self.out_w, self.out_h, self.dst_attrs)

    def close(self) -> None:
        if self.processor is not None:
            self.processor.endSession()
            self.processor = None

    def upscale_to_buffer(self, frame: np.ndarray, frame_index: int) -> Any:
        src_pb = self._make_src_buffer()
        _rgb_to_pixel_buffer(frame, src_pb)
        dst_pb = self._make_dst_buffer()
        pts = CoreMedia.CMTimeMake(frame_index, int(NATIVE_FPS))
        src_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(src_pb, pts)
        dst_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(dst_pb, pts)

        if self.quality == "fast":
            params = vt.VTLowLatencySuperResolutionScalerParameters.alloc(
            ).initWithSourceFrame_destinationFrame_(src_frame, dst_frame)
        else:
            use_temporal = self.quality == "balanced"
            params = vt.VTSuperResolutionScalerParameters.alloc(
            ).initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
                src_frame,
                self._prev_src_frame if use_temporal else None,
                self._prev_dst_frame if use_temporal else None,
                None,
                vt.VTSuperResolutionScalerParametersSubmissionModeSequential,
                dst_frame,
            )

        ok, err = self.processor.processWithParameters_error_(params, None)
        if not ok:
            raise RuntimeError(
                f"VSR processWithParameters failed at frame {frame_index}: {err}"
            )
        self._prev_src_frame = src_frame
        self._prev_dst_frame = dst_frame
        return dst_pb

    def upscale(self, rgb_u8: np.ndarray, frame_index: int) -> np.ndarray:
        """Compat wrapper: returns RGB ndarray (involves a CoreImage readback)."""
        return _pixel_buffer_to_rgb(self.upscale_to_buffer(rgb_u8, frame_index))


# ---------------------------------------------------------------------------
# Input sources — chunk iterators
# ---------------------------------------------------------------------------

def chunk_to_rgba_fp16(chunk: Any, mx_mod: Any) -> np.ndarray:
    """(B,3,T,H,W) bf16 in [-1,1] -> (T,H,W,4) fp16 in [0,1] with alpha=1.

    Direct path for the HighQuality VSR source format (`RGhA`). Skips the
    intermediate uint8 quantization that chunk_to_uint8 does and the
    per-frame uint8->fp16 promotion we'd otherwise do on the CPU.
    """
    B, C, T, H, W = chunk.shape
    rescaled = mx_mod.clip((chunk + 1.0) * 0.5, 0.0, 1.0).astype(mx_mod.float16)
    alpha = mx_mod.ones((B, 1, T, H, W), dtype=mx_mod.float16)
    rgba = mx_mod.concatenate([rescaled, alpha], axis=1)  # (B, 4, T, H, W)
    transposed = mx_mod.transpose(rgba, (0, 2, 3, 4, 1))  # (B, T, H, W, 4)
    mx_mod.eval(transposed)
    arr = np.asarray(transposed)
    return arr[0] if arr.ndim == 5 else arr


def make_video_decoder_default(weights_path: str, compute_dtype: Any, *, backend: str, padding: str):
    """generate.py's happy-path defaults via encode_modes_harness."""
    from scripts.encode_modes_harness import make_video_decoder
    return make_video_decoder(
        weights_path, compute_dtype,
        backend=backend, spatial_padding_mode=padding,
    )


def latent_dims(latent: Any) -> tuple[int, int, int]:
    _, _, latent_frames, latent_height, latent_width = latent.shape
    n_frames = 1 + (latent_frames - 1) * 8
    height = latent_height * 32
    width = latent_width * 32
    return n_frames, height, width


def iter_latent_chunks(
    latent: Any,
    decoder: Any,
    *,
    backend: str,
    mx_mod: Any,
    output_format: str = "uint8_rgb",
) -> Iterator[np.ndarray]:
    """Yield decoded chunks. output_format selects the conversion:
       "uint8_rgb"  -> (T,H,W,3) uint8  (for LowLatency VSR / NV12 source)
       "fp16_rgba"  -> (T,H,W,4) fp16   (for HighQuality VSR / RGBAHalf source)
    """
    from LTX_2_MLX.model.video_vae.tiling import TilingConfig, decode_tiled
    from LTX_2_MLX.model.video_vae.simple_decoder import decode_latent
    from scripts.encode_modes_harness import chunk_to_uint8

    if output_format == "fp16_rgba":
        convert = chunk_to_rgba_fp16
    else:
        convert = chunk_to_uint8

    n_frames, height, width = latent_dims(latent)
    cfg = TilingConfig.auto(
        height=height, width=width, num_frames=n_frames,
        decoder_backend=backend,
    )
    if cfg is None:
        print("VAE tiling: off (auto picked single-shot decode for this backend/shape)")
        # Converter does its own mx.eval; no need to force a separate sync
        # on the lazy `video` array.
        video = decode_latent(latent, decoder)
        yield convert(video, mx_mod)
        del video
        gc.collect()
        try:
            mx_mod.clear_cache()
        except Exception:
            pass
        return

    sp = cfg.spatial_config
    tp = cfg.temporal_config
    spatial_desc = (
        f"spatial tile={sp.tile_size_in_pixels} overlap={sp.tile_overlap_in_pixels}"
        if sp else "no spatial tiling"
    )
    temporal_desc = (
        f"temporal tile={tp.tile_size_in_frames} overlap={tp.tile_overlap_in_frames}"
        if tp else "no temporal tiling"
    )
    print(f"VAE tiling: {spatial_desc}, {temporal_desc}")

    for chunk in decode_tiled(latent, decoder, cfg, show_progress=True):
        # The converter does its own mx.eval at the end. An extra eval here
        # would just add a sync point and prevent MLX from batching the
        # clip+cast+transpose with the VAE's tail kernels.
        out = convert(chunk, mx_mod)
        yield out
        del chunk, out
        gc.collect()
        try:
            mx_mod.clear_cache()
        except Exception:
            pass


def probe_video(mp4_path: Path) -> tuple[int, int, float, int]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "json", str(mp4_path),
        ],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else float(num)
    n = int(info.get("nb_frames", 0)) or 0
    return w, h, fps, n


def iter_video_chunks(mp4_path: Path, w: int, h: int, chunk_size: int = 32) -> Iterator[np.ndarray]:
    """Stream rgb24 frames from ffmpeg stdout in fixed-size chunks."""
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", str(mp4_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
    )
    frame_bytes = h * w * 3
    chunk_bytes = chunk_size * frame_bytes
    try:
        while True:
            buf = proc.stdout.read(chunk_bytes)
            if not buf:
                break
            n = len(buf) // frame_bytes
            if n == 0:
                break
            arr = np.frombuffer(buf[: n * frame_bytes], dtype=np.uint8).reshape(n, h, w, 3)
            yield arr.copy()
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------------------
# AVAssetWriter — appends CVPixelBuffers straight from the VT pipeline, no
# subprocess and no readback. Matches the video_encoder.py "default" tier:
# HEVC main10, BT.709, q=65.
# ---------------------------------------------------------------------------

PIX_BGRA = int.from_bytes(b"BGRA", "big")


HEVC_PROFILE_MAIN10 = "HEVC_Main10_AutoLevel"
HEVC_PROFILE_MAIN422_10 = "HEVC_Main42210_AutoLevel"

# CoreAudio FormatIDs (avoid pulling in the CoreAudio module just for constants)
AUDIO_FORMAT_ALAC = 1634492771   # 'alac'
AUDIO_FORMAT_AAC = 1633772320    # 'aac '
AUDIO_FORMAT_LPCM = 1819304813   # 'lpcm'


def _hevc_video_settings(width: int, height: int, quality: float, profile: str) -> dict:
    return {
        av.AVVideoCodecKey: av.AVVideoCodecTypeHEVC,
        av.AVVideoWidthKey: width,
        av.AVVideoHeightKey: height,
        av.AVVideoColorPropertiesKey: {
            av.AVVideoColorPrimariesKey: av.AVVideoColorPrimaries_ITU_R_709_2,
            av.AVVideoTransferFunctionKey: av.AVVideoTransferFunction_ITU_R_709_2,
            av.AVVideoYCbCrMatrixKey: av.AVVideoYCbCrMatrix_ITU_R_709_2,
        },
        av.AVVideoCompressionPropertiesKey: {
            av.AVVideoProfileLevelKey: profile,
            av.AVVideoQualityKey: quality,
        },
    }


class AudioTrack:
    """In-memory audio decoded from final_audio_latent. Builds CMSampleBuffers
    on demand without ever writing a WAV to disk.

    Constructed from a (channels, samples) float32 ndarray. Push the whole
    track into an AVAssetWriterInput once `feed_into` is called.
    """

    def __init__(self, waveform: np.ndarray, sample_rate: int):
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        if waveform.ndim != 2:
            raise ValueError(f"AudioTrack expects (channels, samples); got {waveform.shape}")
        self.sample_rate = int(sample_rate)
        self.channels = int(waveform.shape[0])
        self.n_samples = int(waveform.shape[1])
        interleaved = np.ascontiguousarray(waveform.T)
        self._bytes = interleaved.tobytes()
        bytes_per_frame = 4 * self.channels

        asbd = CoreAudio.AudioStreamBasicDescription(
            float(self.sample_rate),
            AUDIO_FORMAT_LPCM,
            CoreAudio.kAudioFormatFlagIsFloat | CoreAudio.kAudioFormatFlagIsPacked,
            bytes_per_frame,
            1,
            bytes_per_frame,
            self.channels,
            32,
            0,
        )
        err, fmt = CoreMedia.CMAudioFormatDescriptionCreate(
            None, asbd, 0, None, 0, None, None, None,
        )
        if err != 0 or fmt is None:
            raise RuntimeError(f"CMAudioFormatDescriptionCreate failed: status={err}")
        self.format_desc = fmt

    def save_wav(self, path: Path) -> None:
        from LTX_2_MLX.video_encoder import write_wav_float32
        interleaved = np.frombuffer(self._bytes, dtype=np.float32).reshape(
            self.n_samples, self.channels,
        )
        write_wav_float32(interleaved.T, path, self.sample_rate)

    def make_sample_buffer(self, start_frame: int, end_frame: int) -> Any:
        """Build one CMSampleBuffer covering audio samples [start_frame, end_frame)."""
        n = end_frame - start_frame
        if n <= 0:
            return None
        bytes_per_frame = 4 * self.channels
        chunk_bytes = self._bytes[start_frame * bytes_per_frame: end_frame * bytes_per_frame]
        data_len = len(chunk_bytes)

        err, block = CoreMedia.CMBlockBufferCreateWithMemoryBlock(
            None, None, data_len, None, None, 0, data_len, 1, None,
        )
        if err != 0 or block is None:
            raise RuntimeError(f"CMBlockBufferCreateWithMemoryBlock failed: {err}")
        err = CoreMedia.CMBlockBufferReplaceDataBytes(chunk_bytes, block, 0, data_len)
        if err != 0:
            raise RuntimeError(f"CMBlockBufferReplaceDataBytes failed: {err}")

        pts = CoreMedia.CMTimeMake(start_frame, self.sample_rate)
        err, sample_buf = CoreMedia.CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            None, block, self.format_desc, n, pts, None, None,
        )
        if err != 0 or sample_buf is None:
            raise RuntimeError(
                f"CMAudioSampleBufferCreateReadyWithPacketDescriptions failed: {err}"
            )
        return sample_buf


def _audio_writer_settings(codec: str, sample_rate: int, channels: int) -> dict:
    if codec == "alac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_ALAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitDepthHintKey: 24,
        }
    if codec == "aac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_AAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitRateKey: 256000,
        }
    raise ValueError(f"Unknown audio codec {codec!r}")


def _audio_reader_settings(sample_rate: int, channels: int) -> dict:
    return {
        av.AVFormatIDKey: AUDIO_FORMAT_LPCM,
        av.AVSampleRateKey: float(sample_rate),
        av.AVNumberOfChannelsKey: channels,
        av.AVLinearPCMBitDepthKey: 32,
        av.AVLinearPCMIsFloatKey: True,
        av.AVLinearPCMIsNonInterleavedKey: False,
        av.AVLinearPCMIsBigEndianKey: False,
    }


class AVWriter:
    """One AVAssetWriter + one video AVAssetWriterInput + a pixel buffer adaptor.

    If `audio_wav_path` is provided, also adds an audio AVAssetWriterInput
    encoded with `audio_codec`. The audio samples are pulled from the WAV via
    an AVAssetReader during `finish()` — no ffmpeg involved.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        *,
        source_pixel_format: int,
        profile: str = HEVC_PROFILE_MAIN10,
        quality: float = 0.65,
        label: str = "video",
        audio_track: AudioTrack | None = None,
        audio_codec: str = "alac",
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        url = Foundation.NSURL.fileURLWithPath_(str(output_path))
        writer, err = av.AVAssetWriter.alloc().initWithURL_fileType_error_(
            url, av.AVFileTypeMPEG4, None,
        )
        if writer is None:
            raise RuntimeError(f"AVAssetWriter init failed for {output_path}: {err}")

        video_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
            av.AVMediaTypeVideo, _hevc_video_settings(width, height, quality, profile),
        )
        video_input.setExpectsMediaDataInRealTime_(False)

        src_attrs = {
            Quartz.kCVPixelBufferPixelFormatTypeKey: source_pixel_format,
            Quartz.kCVPixelBufferWidthKey: width,
            Quartz.kCVPixelBufferHeightKey: height,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        }
        adaptor = av.AVAssetWriterInputPixelBufferAdaptor.assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
            video_input, src_attrs,
        )

        if not writer.canAddInput_(video_input):
            raise RuntimeError(f"AVAssetWriter cannot add video input for {output_path}")
        writer.addInput_(video_input)

        audio_input = None
        if audio_track is not None:
            audio_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
                av.AVMediaTypeAudio,
                _audio_writer_settings(audio_codec, audio_track.sample_rate, audio_track.channels),
            )
            audio_input.setExpectsMediaDataInRealTime_(False)
            if not writer.canAddInput_(audio_input):
                raise RuntimeError(
                    f"AVAssetWriter cannot add audio input ({audio_codec}) for {output_path}"
                )
            writer.addInput_(audio_input)

        if not writer.startWriting():
            raise RuntimeError(f"AVAssetWriter.startWriting failed: {writer.error()}")
        writer.startSessionAtSourceTime_(CoreMedia.CMTimeMake(0, int(round(fps))))

        self.writer = writer
        self.video_input = video_input
        self.audio_input = audio_input
        self.adaptor = adaptor
        self.fps = fps
        self.label = label
        self.path = output_path
        self.frame_count = 0
        self.audio_track = audio_track
        self._audio_codec = audio_codec
        audio_desc = f", audio={audio_codec}" if audio_input is not None else ""
        print(
            f"[{label}] AVAssetWriter -> {output_path} "
            f"(HEVC {profile} BT.709 q={quality}{audio_desc})"
        )

        # Audio uses the canonical AVAssetWriter pull pattern: the writer calls
        # back on a dispatch queue when it wants more samples, and we drain
        # while `isReadyForMoreMediaData` stays True. This avoids the polling
        # deadlock where audio + video appends starve each other.
        self._audio_done = threading.Event()
        self._audio_progress = [0]  # boxed so the closure can mutate
        if audio_track is not None:
            queue = libdispatch.dispatch_queue_create(
                f"vsr_harness.audio.{label}".encode(), None,
            )
            self._audio_queue = queue

            n_samples = audio_track.n_samples
            chunk_frames = max(4096, audio_track.sample_rate // 4)  # ~250 ms

            def pump():
                try:
                    while self.audio_input.isReadyForMoreMediaData():
                        pos = self._audio_progress[0]
                        if pos >= n_samples:
                            self.audio_input.markAsFinished()
                            self._audio_done.set()
                            return
                        end = min(pos + chunk_frames, n_samples)
                        sb = audio_track.make_sample_buffer(pos, end)
                        if sb is None or not self.audio_input.appendSampleBuffer_(sb):
                            self._audio_done.set()
                            raise RuntimeError(
                                f"[{label}] audio appendSampleBuffer failed at "
                                f"{pos}: {self.writer.error()}"
                            )
                        self._audio_progress[0] = end
                except Exception:
                    self._audio_done.set()
                    raise

            self.audio_input.requestMediaDataWhenReadyOnQueue_usingBlock_(queue, pump)
        else:
            self._audio_done.set()
            self._audio_queue = None

    def _wait_for_ready(self, input_obj: Any, what: str) -> None:
        waited = 0.0
        while not input_obj.isReadyForMoreMediaData():
            status = self.writer.status()
            if status in (3, 4):  # Failed, Cancelled
                raise RuntimeError(
                    f"[{self.label}] writer entered status={status} while waiting on {what}: "
                    f"{self.writer.error()}"
                )
            time.sleep(0.001)
            waited += 0.001
            if waited > 30.0:
                raise RuntimeError(
                    f"[{self.label}] {what} input never became ready (waited 30s, status={status})"
                )

    def append(self, pb: Any) -> None:
        self._wait_for_ready(self.video_input, "video")
        pts = CoreMedia.CMTimeMake(self.frame_count, int(round(self.fps)))
        if not self.adaptor.appendPixelBuffer_withPresentationTime_(pb, pts):
            raise RuntimeError(
                f"[{self.label}] appendPixelBuffer failed at frame {self.frame_count}: "
                f"status={self.writer.status()} error={self.writer.error()}"
            )
        self.frame_count += 1

    def finish(self) -> None:
        self.video_input.markAsFinished()
        # Wait for the audio pump dispatch block to drain all samples.
        if self.audio_input is not None:
            if not self._audio_done.wait(timeout=120.0):
                raise RuntimeError(
                    f"[{self.label}] audio pump didn't finish (progress="
                    f"{self._audio_progress[0]}/{self.audio_track.n_samples})"
                )
        self.writer.endSessionAtSourceTime_(
            CoreMedia.CMTimeMake(self.frame_count, int(round(self.fps)))
        )
        done = threading.Event()
        self.writer.finishWritingWithCompletionHandler_(lambda: done.set())
        done.wait()
        if self.writer.status() != 2:  # AVAssetWriterStatusCompleted = 2
            raise RuntimeError(
                f"[{self.label}] AVAssetWriter finished with status "
                f"{self.writer.status()}: {self.writer.error()}"
            )


def _make_bgra_pool_buffer(adaptor: Any, w: int, h: int) -> Any:
    """Try the adaptor's pool first; fall back to a fresh CVPixelBufferCreate."""
    pool = adaptor.pixelBufferPool()
    if pool is not None:
        err, pb = Quartz.CVPixelBufferPoolCreatePixelBuffer(None, pool, None)
        if err == 0 and pb is not None:
            return pb
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: PIX_BGRA,
        Quartz.kCVPixelBufferWidthKey: w,
        Quartz.kCVPixelBufferHeightKey: h,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    err, pb = Quartz.CVPixelBufferCreate(None, w, h, PIX_BGRA, attrs, None)
    if err != 0:
        raise RuntimeError(f"CVPixelBufferCreate({w}x{h}, BGRA) failed: {err}")
    return pb


def _render_comparison_buffer(pre_rgb: np.ndarray, post_pb: Any, scale: int, dest_pb: Any) -> None:
    """Side-by-side: NEAREST-upscaled pre on the left, VSR post on the right.

    Pre is upscaled in numpy with np.repeat (true NEAREST). Both halves are then
    composited via CoreImage and rendered into dest_pb in one GPU pass.
    """
    in_h, in_w, _ = pre_rgb.shape
    out_w, out_h = in_w * scale, in_h * scale

    pre_up = np.repeat(np.repeat(pre_rgb, scale, axis=0), scale, axis=1)
    rgba = np.empty((out_h, out_w, 4), dtype=np.uint8)
    rgba[..., 0:3] = pre_up
    rgba[..., 3] = 255
    data = Foundation.NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
    pre_ci = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data, out_w * 4, (out_w, out_h), Quartz.kCIFormatRGBA8, _srgb_colorspace(),
    )

    post_ci = Quartz.CIImage.alloc().initWithCVPixelBuffer_(post_pb)
    post_ci_translated = post_ci.imageByApplyingTransform_(
        Quartz.CGAffineTransformMakeTranslation(out_w, 0)
    )
    composite = post_ci_translated.imageByCompositingOverImage_(pre_ci)

    _ci_context_singleton().render_toCVPixelBuffer_bounds_colorSpace_(
        composite, dest_pb, ((0, 0), (2 * out_w, out_h)), _srgb_colorspace(),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _pick_hevc_profile(quality: str, encode_chroma: str) -> str:
    if encode_chroma == "420":
        return HEVC_PROFILE_MAIN10
    if encode_chroma == "422":
        return HEVC_PROFILE_MAIN422_10
    # auto: 4:2:2 when VSR's source is full-chroma (HQ RGBAHalf), else 4:2:0.
    # Adds ~10% to file size for ~2x chroma resolution. 4:4:4 isn't exposed
    # by VideoToolbox, so this is the best fidelity available here.
    return HEVC_PROFILE_MAIN422_10 if quality in ("balanced", "high") else HEVC_PROFILE_MAIN10


def _decode_audio_track(
    audio_latent: Any,
    weights: str,
    compute_dtype: Any,
) -> AudioTrack:
    """Decode an audio latent through the audio VAE + vocoder into an in-memory
    AudioTrack. No disk WAV unless the caller explicitly asks for a sidecar.
    """
    from scripts.decode_latent_debug import make_audio_decoder_and_vocoder, decode_audio_latent

    import mlx.core as mx

    print("Decoding audio latent (audio VAE + vocoder)...")
    audio_decoder, vocoder, sample_rate = make_audio_decoder_and_vocoder(weights, compute_dtype)
    waveform = decode_audio_latent(audio_latent, audio_decoder, vocoder, mx)
    arr = np.asarray(waveform)
    if arr.ndim == 3:
        arr = arr[0]
    track = AudioTrack(arr, sample_rate=int(sample_rate))
    print(f"  audio: {track.channels}ch, {track.sample_rate} Hz, {track.n_samples} samples")
    del waveform, audio_decoder, vocoder
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return track


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pre_dir = out_root / "pre"
    post_dir = out_root / "post"
    if args.save_pre_frames:
        pre_dir.mkdir(parents=True, exist_ok=True)
    if args.save_post_frames:
        post_dir.mkdir(parents=True, exist_ok=True)

    audio_track: AudioTrack | None = None

    if args.latent:
        from scripts.decode_latent_debug import load_latents, parse_dtype

        import mlx.core as mx

        print(f"[setup] VAE-decoding latent: {args.latent}")
        t = time.perf_counter()
        latent, audio_latent = load_latents(args.latent, mx, "auto")
        compute_dtype = parse_dtype(mx, args.vae_dtype)
        print(f"[setup] load_latents done in {time.perf_counter() - t:.2f}s "
              f"(video_latent={tuple(latent.shape)}, "
              f"audio_latent={'yes' if audio_latent is not None else 'no'})")

        # Audio decode runs serially. Tried threading it against the first
        # VAE chunk and it made the total 50% slower — both MLX workloads
        # serialize on the single Metal scheduler and contend for the GPU.
        if audio_latent is not None and args.audio:
            t = time.perf_counter()
            audio_track = _decode_audio_track(audio_latent, args.weights, compute_dtype)
            print(f"[setup] audio decode in {time.perf_counter() - t:.2f}s")
            if args.save_audio_sidecar:
                sidecar = out_root / "audio.wav"
                audio_track.save_wav(sidecar)
                print(f"[setup] audio sidecar: {sidecar}")

        t = time.perf_counter()
        decoder = make_video_decoder_default(
            args.weights, compute_dtype,
            backend=args.vae_decoder_backend,
            padding=args.vae_spatial_padding,
        )
        print(f"[setup] video VAE loaded in {time.perf_counter() - t:.2f}s")
        total_frames, in_h, in_w = latent_dims(latent)
        fps = args.fps
        chunk_format = "fp16_rgba" if args.quality in ("balanced", "high") else "uint8_rgb"
        chunks = iter_latent_chunks(
            latent, decoder,
            backend=args.vae_decoder_backend, mx_mod=mx,
            output_format=chunk_format,
        )
    else:
        print(f"Reading video: {args.video}")
        in_w, in_h, fps, total_frames = probe_video(Path(args.video))
        chunks = iter_video_chunks(Path(args.video), in_w, in_h, chunk_size=args.video_chunk_size)

    out_w, out_h = in_w * args.scale, in_h * args.scale
    profile = _pick_hevc_profile(args.quality, args.encode_chroma)
    print(
        f"Source: {in_w}x{in_h}, "
        f"total frames: {total_frames or 'unknown'}, "
        f"fps: {fps:.3f}"
    )
    print(f"Target: {out_w}x{out_h} at scale {args.scale}, quality={args.quality}")
    print(f"Encoder: HEVC profile={profile} q={args.quality_setting} "
          f"audio={args.audio_codec if audio_track else 'none'}")

    session = VsrSession(in_w, in_h, scale=args.scale, quality=args.quality)

    audio_kwargs: dict[str, Any] = {}
    if audio_track is not None:
        audio_kwargs = {"audio_track": audio_track, "audio_codec": args.audio_codec}

    post_writer: AVWriter | None = None
    if not args.no_post_mp4:
        post_writer = AVWriter(
            out_root / "post.mp4",
            width=out_w, height=out_h, fps=fps,
            source_pixel_format=_resolve_pixel_format(session.dst_attrs),
            profile=profile,
            quality=args.quality_setting,
            label="post",
            **audio_kwargs,
        )
        # Zero-copy from VSR output to encoder: VSR writes its dst buffer
        # straight into a slot belonging to the writer's adaptor pool. The
        # encoder consumes from the same slot; nothing is allocated per frame.
        pool = post_writer.adaptor.pixelBufferPool()
        if pool is not None:
            session.use_dst_pool(pool)

    comparison_writer: AVWriter | None = None
    if args.comparison:
        comparison_writer = AVWriter(
            out_root / "comparison.mp4",
            width=2 * out_w, height=out_h, fps=fps,
            source_pixel_format=PIX_BGRA,
            profile=profile,
            quality=args.quality_setting,
            label="comparison",
            **audio_kwargs,
        )

    processed = 0
    t_total = time.perf_counter()
    try:
        for chunk in chunks:
            t_h, t_w = chunk.shape[1], chunk.shape[2]
            if (t_w, t_h) != (in_w, in_h):
                raise RuntimeError(
                    f"chunk dims {t_w}x{t_h} don't match VSR config {in_w}x{in_h}"
                )
            for i in range(chunk.shape[0]):
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                src_frame = chunk[i]
                vsr_pb = session.upscale_to_buffer(src_frame, processed)

                if post_writer is not None:
                    post_writer.append(vsr_pb)

                if comparison_writer is not None:
                    comp_pb = _make_bgra_pool_buffer(
                        comparison_writer.adaptor, 2 * out_w, out_h,
                    )
                    _render_comparison_buffer(src_frame, vsr_pb, args.scale, comp_pb)
                    comparison_writer.append(comp_pb)

                if args.save_pre_frames:
                    Image.fromarray(src_frame).save(pre_dir / f"frame_{processed:05d}.png")
                if args.save_post_frames:
                    Image.fromarray(_pixel_buffer_to_rgb(vsr_pb)).save(
                        post_dir / f"frame_{processed:05d}.png"
                    )

                processed += 1
                if processed % 16 == 0:
                    elapsed = time.perf_counter() - t_total
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  pipeline: {processed} frames ({rate:.2f} fps)")

            if args.max_frames is not None and processed >= args.max_frames:
                break
            del chunk
            gc.collect()
    finally:
        session.close()
        for writer in (post_writer, comparison_writer):
            if writer is not None:
                writer.finish()

    elapsed = time.perf_counter() - t_total
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"Processed {processed} frames in {elapsed:.2f}s ({rate:.2f} fps)")
    if post_writer is not None:
        print(f"Post: {post_writer.path}")
    if comparison_writer is not None:
        print(f"Comparison: {comparison_writer.path}")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--latent", help="--save-latents NPZ sidecar (VAE-decoded first).")
    src.add_argument("--video", help="Already-decoded video file (mp4/mov/...).")

    parser.add_argument("--weights", help="LTX-2 .safetensors path (required with --latent).")
    parser.add_argument(
        "--vae-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--vae-decoder-backend",
        choices=["native-conv3d", "simple"],
        default="native-conv3d",
        help="VAE decoder backend. native-conv3d matches generate.py's default.",
    )
    parser.add_argument(
        "--vae-spatial-padding",
        choices=["zero", "reflect"],
        default="zero",
        help="VAE spatial padding. zero matches generate.py's default.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=NATIVE_FPS)
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        default="balanced",
        help=(
            "balanced (recommended for video) = HighQuality Video mode, scale=4, "
            "uses previous frame for temporal coherence. "
            "fast = LowLatency, scale=2 only, per-frame. "
            "high = HighQuality Image mode, scale=4, per-frame — NO temporal "
            "coherence, intended for stills not video."
        ),
    )
    parser.add_argument(
        "--quality-setting",
        type=float,
        default=0.65,
        help="AVVideoQualityKey (0..1) for the HEVC encoder. 0.65 matches the default tier.",
    )
    parser.add_argument(
        "--encode-chroma",
        choices=["auto", "420", "422"],
        default="auto",
        help=(
            "HEVC profile chroma subsampling. "
            "auto = 4:2:2 (Main42210) when --quality is balanced/high "
            "(VSR outputs full-chroma RGBAHalf), else 4:2:0 (Main10). "
            "420 forces Main10 to match generate.py's default tier. "
            "4:4:4 isn't exposed by VideoToolbox."
        ),
    )
    parser.add_argument(
        "--audio-codec",
        choices=["alac", "aac"],
        default="alac",
        help="Audio codec for muxed audio (alac=lossless, aac=256kbps).",
    )
    parser.add_argument("--audio", action="store_true",
                        help="Mux audio (decoded from final_audio_latent) into both MP4s.")
    parser.add_argument("--save-audio-sidecar", action="store_true",
                        help="Also write the decoded audio as audio.wav next to the MP4s. "
                             "Off by default — audio is muxed in memory.")
    parser.add_argument(
        "--video-chunk-size",
        type=int,
        default=32,
        help="Frames per chunk for --video input (streaming).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (debugging).",
    )
    parser.add_argument("--save-pre-frames", action="store_true",
                        help="Also write pre/*.png (off by default — comparison.mp4 has the same data).")
    parser.add_argument("--save-post-frames", action="store_true",
                        help="Also write post/*.png (off by default — post.mp4 has the same data).")
    parser.add_argument("--no-post-mp4", action="store_true",
                        help="Skip the upscaled-only post.mp4.")
    parser.add_argument("--comparison", action="store_true",
                        help="Also write a side-by-side comparison.mp4 "
                             "(NEAREST-upscaled pre vs VSR post). Off by default.")
    args = parser.parse_args()

    if args.latent and not args.weights:
        parser.error("--latent requires --weights")

    run(args)


if __name__ == "__main__":
    main()
