#!/usr/bin/env python3
"""Localize and characterize start-of-clip audio artifacts.

Built originally to diagnose the "loud click at t=0" sequence-start
spike that LTX-2.3 AV produces on *some* clips (dialog-heavy prompts
in particular - see `docs/AUDIO_ISSUES.md` -> "Sequence-Start Audio
Spike" for the dialog-vs-ambient comparison).  The tool is generic:
any LTX-2 audio output sidecar pair (decoded WAV + latent sidecar) can
be passed in, and it reports whether the head of the clip has
anomalously high energy relative to the rest, in both the time
domain and in the latent.

The two domains can diverge - elevated lat_t=0 RMS appears to be
universal (attention-sink behavior) while the audible WAV spike is
content-dependent.  The VERDICT line therefore reflects only the
audible (WAV) case; latent stats are printed for context and for
follow-up work characterizing *why* some clips click and some
don't.

Inputs
------
Point the script at any LTX-2-MLX run by giving it a path whose stem
is shared by the run's sidecars; the script derives the `.wav` and latent sidecar
from that stem.

    scripts/analyze_audio_onset.py \\
        --run $DIFFUSERS_OUTPUT_DIR/<stem>.mp4

Equivalently, pass the WAV or the latent sidecar directly:

    scripts/analyze_audio_onset.py --run $DIFFUSERS_OUTPUT_DIR/<stem>.wav

If the latent sidecar isn't present (e.g. the run wasn't launched with
`--save-latents` / `--save-all-sidecars`), only the WAV analysis runs.

What it prints
--------------
1. WAV summary - channels, sample rate, duration, global RMS/peak.
2. Coarse head profile - N-millisecond windows over the first M
   seconds (defaults: 50 ms windows, 2.0 s window).  Each window
   shows RMS, ratio vs. global RMS, and an ASCII bar.
3. Fine head profile - finer windows over a smaller leading slice
   (defaults: 5 ms windows, 120 ms window).  Useful for localizing
   transient attacks to within a single AV frame's worth of audio.
4. Per-frame latent profile - for each audio-latent key found in the
   sidecar (`final_audio_latent`, and on distilled two-stage runs also
   `stage_1_audio_latent` / `stage_2_audio_latent`), prints per-frame
   RMS / max-abs / energy-vs-median ratio for the first K latent
   frames (default 30, which spans ~1.2 s at LTX-2.3's 25 fps_lat).
5. Verdict - a one-line "spike at start" classification using a
   simple ratio threshold (default 2.0x global RMS in the first
   window).  Returns non-zero exit code when a spike is detected, so
   the script is also usable in CI / sweep contexts.

Why this script exists
----------------------
The sequence-start spike is reliably reproducible across prompts and
seeds - it's a model-side artifact, not a one-clip fluke.  Having a
dedicated diag tool means:

  - We can re-measure after any audio-side change (VAE decoder tweaks,
    vocoder/BWE precision changes, latent-trim mitigations, ...) to
    confirm whether the spike moved or stayed put.
  - We can characterize the spike on new prompt/seed combos without
    re-deriving the analysis from scratch each time.
  - Mitigation work (latent trim, fade-in, etc.) gets a single
    before/after metric to point at instead of "the click sounds
    quieter now, I think."

Implementation notes
--------------------
- WAV is read with `scipy.io.wavfile` (Python's stdlib `wave` doesn't
  parse IEEE_FLOAT format=3, which is what our audio writer emits).
- Audio latent shape is `(B=1, C=8, T, F=16)`.  The frame rate is
  `T / duration_seconds`; LTX-2.3 lands at 25 fps_lat for 30-second
  clips (752/30).  Older models or different durations may differ -
  the script computes per-run from the latent shape and WAV duration.
- Per-frame latent RMS uses the (C, F) dims as the "amplitude"
  channels and treats T as time, mirroring how the vocoder sees them.
- The "spike" heuristic is intentionally simple (single-window ratio
  vs. global) so it can't accidentally hide a real artifact behind a
  fancier statistic.  Tune `--spike-threshold` if you want stricter
  / laxer detection.

See `docs/AUDIO_ISSUES.md` -> "Sequence-Start Audio Spike (OPEN ...)" for
the full diagnosis on the canonical reproduction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from scipy.io import wavfile

# Thresholds for "is this a spike" come from the same module the encoders
# use for the mitigation, so the diagnostic and the production gate move
# together when either is tuned.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LTX_2_MLX.audio.onset import (
    DEFAULT_DETECT_THRESHOLD_RATIO,
    DEFAULT_DETECT_WINDOW_MS,
    detect_onset_spike,
)
from LTX_2_MLX.sidecars import find_sidecar, load_sidecar

# Default analysis parameters.  Picked to localize sub-100 ms transients
# while staying readable as ASCII tables in a terminal.  The coarse window
# size + spike threshold come from the shared onset module so the
# diagnostic line in this report uses the same numbers as the encoders'
# detect-then-trim gate.
DEFAULT_COARSE_WINDOW_MS = DEFAULT_DETECT_WINDOW_MS
DEFAULT_COARSE_HEAD_S = 2.0
DEFAULT_FINE_WINDOW_MS = 5.0
DEFAULT_FINE_HEAD_MS = 120.0
DEFAULT_LATENT_FRAMES = 30
DEFAULT_SPIKE_THRESHOLD = DEFAULT_DETECT_THRESHOLD_RATIO  # first-window RMS / global RMS


# ---------------------------------------------------------------------------
# Sidecar resolution
# ---------------------------------------------------------------------------

def resolve_sidecars(run_path: Path) -> tuple[Path, Path | None]:
    """Given any sidecar path (.mp4 / .wav / .npz / .safetensors), derive the WAV + latent sidecar.

    Returns (wav_path, latent_sidecar_or_None).  The WAV is required;
    the latent sidecar is optional.  Raises SystemExit with a clear message if the WAV is
    missing.
    """
    stem_base = run_path.with_suffix("")
    wav = stem_base.with_suffix(".wav")
    if not wav.exists():
        raise SystemExit(
            f"No WAV sidecar at {wav}.  Re-run generate.py with "
            f"--save-audio-sidecar (or --save-all-sidecars) so the "
            f"decoded audio is preserved next to the mp4."
        )
    latent_sidecar = find_sidecar(str(stem_base))
    return wav, (Path(latent_sidecar) if latent_sidecar else None)


# ---------------------------------------------------------------------------
# WAV reading + windowed RMS
# ---------------------------------------------------------------------------

def read_wav(path: Path) -> tuple[mx.array, int]:
    """Load any WAV format supported by scipy (PCM int16/int24/int32, float32).

    scipy hands back a NumPy array; convert it to MLX at this boundary and keep
    everything downstream native. Returns (samples (N, channels) float32 in
    [-1, 1] as an MLX array, sample_rate).
    """
    sr, arr = wavfile.read(str(path))
    if arr.ndim == 1:
        arr = arr[:, None]
    samples = mx.array(arr).astype(mx.float32)
    dtype = str(arr.dtype)
    if dtype == "int16":
        samples = samples / 32768.0
    elif dtype == "int32":
        samples = samples / 2147483648.0
    elif dtype == "uint8":
        samples = (samples - 128.0) / 128.0
    return samples, sr


def windowed_rms(
    samples: mx.array, sr: int, window_ms: float, head_ms: float,
) -> tuple[list[float], list[float], list[float]]:
    """Mono RMS over fixed-size windows covering the first `head_ms`.

    Mono is the per-window mean across channels (so a stereo spike
    that hits both channels equally shows as a spike, but a one-channel
    transient is averaged down).  This matches the human-perception
    case where the click is heard as "loud at the start," not "loud
    in the left ear only."

    Returns (window_start_times_ms, rms_per_window, peak_abs_per_window).
    """
    mono = mx.mean(samples, axis=1)
    win = max(1, int(round(window_ms / 1000 * sr)))
    n = max(1, int(round(head_ms / 1000 * sr)))
    head = mono[:n]
    n_full = head.size // win
    if n_full == 0:
        return [0.0], [0.0], [0.0]
    trimmed = head[: n_full * win].reshape(n_full, win)
    rms = mx.sqrt(mx.mean(trimmed ** 2, axis=1) + 1e-12)
    peak = mx.max(mx.abs(trimmed), axis=1)
    t_start_ms = [i * window_ms for i in range(n_full)]
    return t_start_ms, rms.tolist(), peak.tolist()


def ascii_bar(value: float, ref: float, max_chars: int = 60) -> str:
    """Scale `value` against `ref` and return `#` characters proportional."""
    if ref <= 0:
        return ""
    n = int(min(max_chars, max(0, value / ref * 20)))
    return "#" * n


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def report_wav_summary(samples: mx.array, sr: int, path: Path) -> None:
    n, ch = samples.shape
    duration = n / sr
    rms = float(mx.sqrt(mx.mean(samples ** 2)).item())
    peak = float(mx.max(mx.abs(samples)).item())
    print(f"WAV: {path}")
    print(
        f"  channels={ch}  sample_rate={sr}  samples={n}  "
        f"duration={duration:.3f}s"
    )
    print(f"  global: rms={rms:.4f}  peak={peak:.4f}")


def report_head_profile(
    samples: mx.array, sr: int, window_ms: float, head_ms: float,
    label: str,
) -> tuple[float, float]:
    """Print a windowed RMS table.  Returns (first_window_rms, global_rms)."""
    global_rms = float(mx.sqrt(mx.mean(mx.mean(samples, axis=1) ** 2)).item())
    t, rms, peak = windowed_rms(samples, sr, window_ms, head_ms)
    print(
        f"\n{label} - {window_ms:g} ms windows over first "
        f"{head_ms:g} ms (global RMS={global_rms:.4f}):"
    )
    for ti, ri, pi in zip(t, rms, peak, strict=True):
        ratio = ri / max(global_rms, 1e-9)
        bar = ascii_bar(ri, global_rms)
        end_ms = ti + window_ms
        print(
            f"  t={ti:6.1f}-{end_ms:6.1f} ms  rms={ri:.4f}  peak={pi:.4f}  "
            f"({ratio:5.2f}x global)  {bar}"
        )
    return float(rms[0]) if len(rms) else 0.0, global_rms


def _quantile(values: mx.array, q: float) -> float:
    """Linear-interpolation quantile (matches numpy's default), via mx.sort."""
    n = values.size
    if n == 0:
        return 0.0
    ordered = mx.sort(values.reshape(-1))
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float((ordered[lo] * (1.0 - frac) + ordered[hi] * frac).item())


def report_audio_latents(
    latent_sidecar: Path, latent_frames: int, duration_s: float,
) -> None:
    """Print per-frame audio-latent statistics for any audio latent keys
    found in the sidecar.  Handles both single-stage runs and distilled
    two-stage runs (which expose stage_1 / stage_2 / final keys).
    """
    print(f"\nLatents: {latent_sidecar}")
    arrays, _metadata = load_sidecar(str(latent_sidecar))
    # load_sidecar returns native MLX arrays; cast to float32 for the stats math.
    d = {k: v.astype(mx.float32) for k, v in arrays.items()}
    audio_keys = [k for k in d if "audio_latent" in k]
    if not audio_keys:
        print("  (no audio-latent keys found in the sidecar)")
        return
    # Order: final / stage_2 / stage_1 / anything else, so the comparison
    # progresses from "what the vocoder actually saw" backward through
    # the refinement chain.
    priority = {"final_audio_latent": 0, "stage_2_audio_latent": 1, "stage_1_audio_latent": 2}
    audio_keys.sort(key=lambda k: (priority.get(k, 99), k))

    for key in audio_keys:
        lat = d[key]
        if lat.ndim != 4:
            print(f"\n  {key}: unexpected ndim={lat.ndim} (skipping)")
            continue
        B, C, T, F = lat.shape
        # 4D `(c, f)` flatten -> "amplitude" channels, T is time.  Per-t
        # RMS divides by C*F so the scale matches a standard L2 norm
        # divided by sqrt(features), making cross-frame comparison easy.
        sample = lat[0]
        per_t_norm = mx.sqrt(mx.sum(sample ** 2, axis=(0, 2)) / (C * F))
        per_t_max = mx.max(mx.abs(sample), axis=(0, 2))
        global_rms = float(mx.sqrt(mx.mean(sample ** 2)).item())
        global_peak = float(mx.max(mx.abs(sample)).item())
        median = _quantile(per_t_norm, 0.5)
        p99 = _quantile(per_t_norm, 0.99)
        per_t_norm = per_t_norm.tolist()
        per_t_max = per_t_max.tolist()
        fps_lat = T / duration_s if duration_s > 0 else float("nan")
        ms_per_frame = 1000.0 / fps_lat if fps_lat > 0 else float("nan")
        print(f"\n  {key}: shape={lat.shape}  fps_lat={fps_lat:.2f}")
        print(
            f"    global: rms={global_rms:.4f}  peak={global_peak:.3f}  "
            f"per-t rms median={median:.3f}  p99={p99:.3f}"
        )
        n_show = min(latent_frames, T)
        print(f"    first {n_show} latent frames (~{n_show * ms_per_frame:.0f} ms):")
        for i in range(n_show):
            ratio = per_t_norm[i] / max(median, 1e-9)
            bar = ascii_bar(per_t_norm[i], median)
            print(
                f"      lat_t={i:3d} (~{i * ms_per_frame:6.0f} ms)  "
                f"rms={per_t_norm[i]:.3f}  max={per_t_max[i]:6.2f}  "
                f"({ratio:4.2f}x){bar}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run", required=True,
        help=(
            "Path to any sidecar from the run (.mp4, .wav, or the latent .npz/.safetensors).  "
            "The sibling .wav and latent sidecar are auto-resolved from the stem."
        ),
    )
    p.add_argument(
        "--coarse-window-ms", type=float, default=DEFAULT_COARSE_WINDOW_MS,
        help=f"Window size for the coarse head profile (default {DEFAULT_COARSE_WINDOW_MS:g})",
    )
    p.add_argument(
        "--coarse-head-s", type=float, default=DEFAULT_COARSE_HEAD_S,
        help=f"Coarse-profile total duration in seconds (default {DEFAULT_COARSE_HEAD_S:g})",
    )
    p.add_argument(
        "--fine-window-ms", type=float, default=DEFAULT_FINE_WINDOW_MS,
        help=f"Window size for the fine head profile (default {DEFAULT_FINE_WINDOW_MS:g})",
    )
    p.add_argument(
        "--fine-head-ms", type=float, default=DEFAULT_FINE_HEAD_MS,
        help=f"Fine-profile total duration in ms (default {DEFAULT_FINE_HEAD_MS:g})",
    )
    p.add_argument(
        "--latent-frames", type=int, default=DEFAULT_LATENT_FRAMES,
        help=(
            f"Number of leading latent frames to dump per audio-latent key "
            f"(default {DEFAULT_LATENT_FRAMES})."
        ),
    )
    p.add_argument(
        "--spike-threshold", type=float, default=DEFAULT_SPIKE_THRESHOLD,
        help=(
            f"Coarse-window-0 RMS ratio above which a start-of-clip spike is "
            f"flagged (default {DEFAULT_SPIKE_THRESHOLD:g}x global)."
        ),
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero when a spike is detected (useful in sweeps / CI).",
    )
    args = p.parse_args()

    run_path = Path(args.run)
    wav_path, latent_sidecar = resolve_sidecars(run_path)

    samples, sr = read_wav(wav_path)
    report_wav_summary(samples, sr, wav_path)

    first_win_rms, global_rms = report_head_profile(
        samples, sr,
        window_ms=args.coarse_window_ms,
        head_ms=args.coarse_head_s * 1000.0,
        label="Coarse head profile",
    )
    report_head_profile(
        samples, sr,
        window_ms=args.fine_window_ms,
        head_ms=args.fine_head_ms,
        label="Fine head profile",
    )

    duration_s = samples.shape[0] / sr
    if latent_sidecar is not None:
        report_audio_latents(latent_sidecar, args.latent_frames, duration_s)
    else:
        print(
            "\n(no latent sidecar next to the WAV - re-run with "
            "--save-latents / --save-all-sidecars to also see latent stats)"
        )

    # Verdict ---------------------------------------------------------------
    # Diagnostic verdict (single-window): catches anything elevated at t=0,
    # including legitimate loud onsets.
    # Mitigation verdict (two-window): matches the encoders' detect-then-
    # trim gate - fires only when the loud onset is followed by silence
    # (the click signature).  These can disagree, which is informative.
    ratio = first_win_rms / max(global_rms, 1e-9)
    mitigation_would_fire = detect_onset_spike(samples.T, sr)
    print()
    diagnostic_hit = ratio >= args.spike_threshold
    if diagnostic_hit:
        print(
            f"VERDICT: SPIKE detected - first {args.coarse_window_ms:g} ms "
            f"RMS is {ratio:.2f}x global "
            f"(threshold {args.spike_threshold:g}x).  See docs/AUDIO_ISSUES.md -> "
            f"\"Sequence-Start Audio Spike\" for mitigation options."
        )
    else:
        print(
            f"VERDICT: clean - first {args.coarse_window_ms:g} ms RMS is "
            f"{ratio:.2f}x global (below threshold {args.spike_threshold:g}x)."
        )
    print(
        f"MITIGATION: --audio-onset-trim auto would "
        f"{'FIRE' if mitigation_would_fire else 'NOT fire'} on this clip "
        f"(two-window check: loud first window AND quiet 100-250 ms trail-off)."
    )
    if diagnostic_hit and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
