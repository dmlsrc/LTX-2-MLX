"""Video encoding for LTX-2 outputs.

Five tiers cover the practical use cases. Pick by *destination*, not by
encoder flags — the flags are an implementation detail.

  web        Browser <video> tags, social uploads, platforms that don't
             re-transcode. Universal compat from old browsers to current.
             H.264 SW 8-bit 4:2:0 CRF 18 + AAC 320k.

  default    Everyday output, viewed in Apple / Safari / modern Chrome.
             Best size-to-quality on Apple Silicon at the cost of browser
             reach. Hardware-encoded so essentially free runtime.
             HEVC HW 10-bit 4:2:0 q=65 + ALAC.

  hq         Local viewing where 4:4:4 chroma is wanted (sharp color edges,
             neon, skin against bright backgrounds). Won't play in any
             browser — IINA / QuickTime / FCP only.
             HEVC SW 10-bit 4:4:4 CRF 14 + ALAC.

  export     Hand off to an editor or colorist. Industry-standard ProRes 422
             HQ in MOV with 24-bit PCM audio. NLE-friendly, no alpha.
             ProRes 422 HQ + PCM 24-bit (.mov), 10-bit 4:2:2.

  reference  Canonical highest-fidelity copy to re-derive other tiers from
             later. Has alpha for compositing/VFX work. Large files.
             ProRes 4444 + PCM 24-bit + alpha (.mov), 10-bit 4:4:4.

Call `encode_video(frames, output_path, tier=...)` from generate.py / pipelines.
Importers may read `TIERS` and extend with additional `EncodePreset` entries;
the encode_modes_harness.py benchmarking script does exactly this, so keep
tier names and the `EncodePreset` shape stable.

Encoder-choice details that aren't obvious from the flags alone:
  - The libswscale `scale` filter is used instead of `zscale` because zscale
    needs ffmpeg built against zimg, which isn't shipped on every machine.
    Behavior for our 8/10-bit BT.709 conversions is equivalent.
  - For RGB-domain streams (the harness has one; none of the five tiers do),
    color tagging must go inside `-x265-params` — the global `-colorspace`
    flag is forwarded to libx265 under a parameter name the encoder rejects.
  - For ProRes 4444 the alpha plane is filled implicitly with opaque; we
    feed yuva444p10le even though the source has no real alpha.
"""

from __future__ import annotations

import struct
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


NATIVE_FPS = 24.0

# BT.709 SDR tagging. The encoder writes these into the bitstream VUI / atoms
# so players don't have to guess. Without them QuickTime defaults to BT.601 on
# small frames, which produces a subtle green-shift on skies / skin.
COLOR_TAGS_BT709 = [
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "bt709",
    "-color_range", "tv",
]

# For RGB-domain streams (gbrp10le etc.), set matrix_coefficients=0 (identity)
# so the decoder doesn't apply a YUV->RGB conversion. Required when encoding
# RGB-direct (e.g. libx265 lossless gbrp10le). Currently used only by the
# harness's benchmark presets; documented here for future HDR / extended tiers.
COLOR_TAGS_RGB = [
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "gbr",
    "-color_range", "pc",
]


def scale_filter(pix_fmt: str) -> list[str]:
    """RGB->{YUV or RGB} color conversion filter chain.

    Uses libswscale's `scale` filter (always present) rather than `zscale`
    (which needs an ffmpeg built against zimg). For YUV outputs applies a
    BT.709 matrix with full->limited range conversion; for RGB outputs it's
    a pure bit-depth/layout promotion. accurate_rnd + full_chroma_int +
    full_chroma_inp give the cleanest 8->10-bit gradient handling that
    swscale supports.
    """
    base_flags = "flags=accurate_rnd+full_chroma_int+full_chroma_inp"
    if pix_fmt.startswith(("gbrp", "rgb", "bgr")):
        return ["-vf", f"scale={base_flags},format={pix_fmt}"]
    return [
        "-vf",
        (
            f"scale={base_flags}"
            ":in_color_matrix=bt709:in_range=pc"
            ":out_color_matrix=bt709:out_range=tv,"
            f"format={pix_fmt}"
        ),
    ]


def h264_baseline_video() -> list[str]:
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]


@dataclass
class EncodePreset:
    name: str
    container: str = "mp4"
    video: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    frame_bit_depth: int = 8        # 8 -> rgb24 source; 16 -> rgb48le source
    notes: str = ""


TIERS: dict[str, EncodePreset] = {
    "web": EncodePreset(
        name="web",
        video=[*h264_baseline_video(), *COLOR_TAGS_BT709],
        audio=["-c:a", "aac", "-b:a", "320k"],
        notes="Universal browser/player compat. H.264 SW 8-bit 4:2:0 CRF 18 + AAC 320k.",
    ),
    "default": EncodePreset(
        name="default",
        video=[
            *scale_filter("p010le"),
            "-c:v", "hevc_videotoolbox", "-profile:v", "main10",
            "-q:v", "65",
            # -realtime 0 lets VideoToolbox spend more cycles per frame on
            # rate-distortion. Free quality gain; only hurts live streaming.
            "-realtime", "0",
            "-tag:v", "hvc1",
            *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "alac"],
        notes="Everyday output for Apple / Safari / modern Chrome. HEVC HW 10-bit 4:2:0 q=65 + ALAC.",
    ),
    "hq": EncodePreset(
        name="hq",
        video=[
            *scale_filter("yuv444p10le"),
            "-c:v", "libx265", "-profile:v", "main444-10",
            "-crf", "14", "-preset", "slow",
            "-x265-params", "aq-mode=3",
            "-tag:v", "hvc1",
            *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "alac"],
        notes="Local viewing, full chroma. HEVC SW 10-bit 4:4:4 CRF 14 + ALAC (no browser support).",
    ),
    "export": EncodePreset(
        name="export",
        container="mov",
        video=[
            *scale_filter("yuv422p10le"),
            "-c:v", "prores_ks", "-profile:v", "hq",
            "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
            *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "pcm_s24le"],
        frame_bit_depth=16,
        notes="Editor / colorist hand-off. ProRes 422 HQ + PCM 24-bit (.mov), 10-bit 4:2:2.",
    ),
    "reference": EncodePreset(
        name="reference",
        container="mov",
        video=[
            *scale_filter("yuva444p10le"),
            "-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
            *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "pcm_s24le"],
        frame_bit_depth=16,
        notes="Canonical highest-fidelity copy; alpha for VFX/compositing. ProRes 4444 + PCM 24-bit (.mov), 10-bit 4:4:4.",
    ),
}


def build_ffmpeg_cmd(
    preset: EncodePreset,
    *,
    raw_pix_fmt: str,
    width: int,
    height: int,
    fps: float,
    audio_path: Path | None,
    output_path: Path,
) -> list[str]:
    """Build the ffmpeg command for a preset, reading raw frames from stdin."""
    cmd: list[str] = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", raw_pix_fmt,
        "-s", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
    ]
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path)])
    cmd.extend(preset.video)
    if audio_path is not None:
        cmd.extend(preset.audio)
    cmd.extend(preset.extra)
    if audio_path is not None:
        cmd.append("-shortest")
    if preset.container == "mp4":
        cmd.extend(["-movflags", "+faststart"])
    cmd.extend(["-loglevel", "error"])
    cmd.append(str(output_path))
    return cmd


def write_wav_int16(audio_waveform: Any, path: Path, sample_rate: int) -> None:
    """Write a stereo int16 WAV. Accepts mx.array / np.ndarray with shape (B,C,T) or (C,T)."""
    audio_np = np.asarray(audio_waveform)
    if audio_np.ndim == 3:
        audio_np = audio_np[0]
    pcm = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    interleaved = pcm.T.flatten().tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved)


def write_wav_float32(audio_waveform: Any, path: Path, sample_rate: int) -> None:
    """WAVE_FORMAT_IEEE_FLOAT (float32 PCM). No int16 quantization."""
    audio_np = np.asarray(audio_waveform).astype(np.float32)
    if audio_np.ndim == 3:
        audio_np = audio_np[0]
    channels, samples = audio_np.shape
    pcm = audio_np.T.flatten().tobytes()
    bytes_per_sample = 4
    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_size = samples * block_align
    fmt = struct.pack(
        "<HHIIHH",
        3, channels, sample_rate, byte_rate, block_align, bytes_per_sample * 8,
    )
    body = (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", data_size) + pcm
    )
    path.write_bytes(body)


def _normalize_frames(
    frames: Sequence[np.ndarray] | np.ndarray,
    bit_depth: int,
) -> np.ndarray:
    """Coerce frames to a contiguous (T, H, W, 3) ndarray at the required dtype.

    Accepts a list of (H, W, 3) frames or a 4D ndarray. For 16-bit tiers the
    8->16 promotion uses *257 (so 0xFF -> 0xFFFF) which is the bit-exact
    "stretch to full range" conversion, not a left-shift.
    """
    if isinstance(frames, (list, tuple)):
        arr = np.stack([np.asarray(f) for f in frames], axis=0)
    else:
        arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3); got shape {arr.shape}")
    if bit_depth == 16:
        if arr.dtype == np.uint16:
            return np.ascontiguousarray(arr)
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr.astype(np.uint16) * 257)
        raise ValueError(f"Cannot use {arr.dtype} frames for a 16-bit tier")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return np.ascontiguousarray((arr >> 8).astype(np.uint8))
    raise ValueError(f"Cannot use {arr.dtype} frames for an 8-bit tier")


def encode_video(
    frames: Sequence[np.ndarray] | np.ndarray,
    output_path: str | Path,
    *,
    tier: str = "default",
    fps: float = NATIVE_FPS,
    audio_waveform: Any = None,
    audio_sample_rate: int | None = None,
    audio_bit_depth: str = "int16",
    save_audio_sidecar: bool = False,
    audio_onset_trim_mode: str = "auto",
    audio_onset_trim_ms: float | None = None,
    verbose: bool = True,
) -> Path:
    """Encode frames (+ optional audio) into the chosen output tier.

    Returns the actual output path. If the tier requires a different container
    extension (ProRes -> .mov), the supplied path is rewritten accordingly.

    `frames` can be a list of (H,W,3) uint8 frames or a (T,H,W,3) ndarray;
    8/16-bit conversion is handled internally based on the tier's frame_bit_depth.

    `audio_waveform` is anything castable to numpy with shape (B,C,T) or (C,T).
    Pass None to skip audio and produce a video-only file.

    When `save_audio_sidecar=True`, the WAV intermediate is preserved as
    `<output>.wav` next to the encoded video (useful for A/B comparison
    against the codec-compressed audio). Otherwise it's written to a hidden
    temp path and deleted once ffmpeg consumes it.

    `audio_onset_trim_mode` / `audio_onset_trim_ms` route through to
    `LTX_2_MLX.audio.onset.mitigate_onset()` before WAV writing.  Default
    is "auto" (detect-then-zero-fill); pass "off" to disable.  The trim
    is applied to BOTH the ffmpeg input and (if kept) the sidecar so the
    on-disk artifact and the muxed track agree.
    """
    if tier not in TIERS:
        raise ValueError(
            f"Unknown encode tier {tier!r}. Options: {sorted(TIERS)}"
        )
    preset = TIERS[tier]

    frames_arr = _normalize_frames(frames, preset.frame_bit_depth)
    n_frames, height, width, _ = frames_arr.shape

    output_path = Path(output_path)
    if output_path.suffix.lstrip(".").lower() != preset.container:
        output_path = output_path.with_suffix(f".{preset.container}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  encode tier: {tier} — {preset.notes}")
        print(f"  -> {output_path}")

    audio_path: Path | None = None
    sidecar_kept = False
    if audio_waveform is not None:
        if audio_sample_rate is None:
            raise ValueError(
                "audio_sample_rate is required when audio_waveform is provided"
            )
        # Sequence-start onset mitigation, applied once here so both the
        # ffmpeg input WAV and the optional sidecar carry the cleaned
        # waveform.  See LTX_2_MLX.audio.onset for the detector spec.
        from .audio import DEFAULT_TRIM_MS, mitigate_onset

        onset_trim_ms = (
            audio_onset_trim_ms if audio_onset_trim_ms is not None else DEFAULT_TRIM_MS
        )
        onset_result = mitigate_onset(
            audio_waveform, audio_sample_rate,
            mode=audio_onset_trim_mode, trim_ms=onset_trim_ms,
        )
        audio_waveform = onset_result.samples
        if verbose and onset_result.applied:
            print(f"  audio onset: {onset_result.detail}")
        if save_audio_sidecar:
            # Public sidecar: lives next to the output, survives encode.
            audio_path = output_path.with_suffix(".wav")
            sidecar_kept = True
        else:
            # Hidden temp: removed once ffmpeg consumes it.
            audio_path = output_path.with_name(f".{output_path.stem}.audio.wav")
        if audio_bit_depth == "float32":
            write_wav_float32(audio_waveform, audio_path, audio_sample_rate)
        else:
            write_wav_int16(audio_waveform, audio_path, audio_sample_rate)
        if verbose and sidecar_kept:
            print(f"  audio sidecar: {audio_path}  ({audio_bit_depth}, {audio_sample_rate} Hz)")

    raw_pix_fmt = "rgb48le" if preset.frame_bit_depth == 16 else "rgb24"
    cmd = build_ffmpeg_cmd(
        preset,
        raw_pix_fmt=raw_pix_fmt,
        width=width, height=height, fps=fps,
        audio_path=audio_path, output_path=output_path,
    )

    started = time.perf_counter()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            proc.stdin.write(frames_arr.tobytes())
        finally:
            proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg failed (rc={rc}) for encode tier {tier!r}"
            )
    finally:
        if audio_path is not None and not sidecar_kept and audio_path.exists():
            audio_path.unlink()

    if verbose:
        elapsed = time.perf_counter() - started
        size = output_path.stat().st_size
        print(f"  done: {_human_size(size)} in {elapsed:.1f}s")

    return output_path
