"""Sequence-start onset-spike detection and mitigation.

Some LTX-2.3 AV generations produce a ~65 ms loud transient at t=0
followed by clean silence until the first spoken word.  See
`docs/AUDIO_ISSUES.md` -> "Sequence-Start Audio Spike" for the full
characterization; the short version is:

  * Dialog-heavy prompts: the model encodes a broadband loud onset
    that BWE synthesizes into an audible click.
  * Ambient-onset prompts: the elevated lat_t=0 is still there
    (universal attention-sink behavior) but decodes to quiet.

So the mitigation is **detect-then-trim**, not unconditional surgery,
and the trim **mutes/zero-fills** the leading region rather than
dropping samples - dropping would shift audio relative to video and
break lip sync.

Detection
---------
The signature is *loud burst followed by silence*, not just *loud
burst*.  A real speech onset at t=0 would also be loud at t=0 but
would NOT be followed by silence - it would sustain.  So we use a
two-window check:

  first 50 ms RMS > 2.0x global RMS
  AND
  mean RMS over 100-250 ms < 0.1x global RMS

A legitimate loud-speech onset trips condition 1 but not 2 (sustained
content keeps the 100-250 ms RMS near global RMS), so the trim does
not fire and lip sync at t=0 is preserved.

Trim
----
The trim zeros out the leading 120 ms by default - long enough to
clear the diagnosed click's ~95 ms decay tail with margin, short
enough to sit comfortably inside the intentional silence the model
places before the first spoken word (95-250 ms on the diagnosed clip).
Sample count is preserved.

Public API
----------
- `detect_onset_spike(samples, sample_rate, ...)` -> bool
- `trim_onset(samples, sample_rate, *, trim_ms)` -> trimmed mx.array
- `mitigate_onset(samples, sample_rate, *, mode, trim_ms)` -> result
- `parse_trim_mode(s)` -> (mode, trim_ms) for the CLI flag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

# ---------------------------------------------------------------------------
# Threshold constants - single source of truth for both this module and
# scripts/analyze_audio_onset.py (which imports the same defaults so the
# diagnostic and the mitigation agree on what counts as a spike).
# ---------------------------------------------------------------------------

DEFAULT_DETECT_WINDOW_MS = 50.0
"""First-window size for the detector (and the coarse-profile analyzer)."""

DEFAULT_DETECT_THRESHOLD_RATIO = 2.0
"""Multiplier on global RMS that the first window must exceed."""

DEFAULT_SILENCE_START_MS = 100.0
"""Lower bound of the 'must be quiet' trail-off window."""

DEFAULT_SILENCE_END_MS = 250.0
"""Upper bound of the 'must be quiet' trail-off window."""

DEFAULT_SILENCE_RATIO = 0.1
"""Multiplier on global RMS that the trail-off window must NOT exceed."""

DEFAULT_TRIM_MS = 120.0
"""Default leading-region duration to zero out when a spike is detected."""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def _to_audio_ct_mlx(samples: Any, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Coerce (B,C,T) / (C,T) / (T,) to an MLX (C,T) audio array."""
    arr = mx.array(samples, dtype=dtype)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            f"audio_waveform must be (B,C,T), (C,T), or (T,); got shape {arr.shape}"
        )
    return arr


def _rms(values: mx.array) -> mx.array:
    """RMS for an MLX float64 vector."""
    return mx.sqrt(mx.mean(mx.square(values)))


def detect_onset_spike(
    samples: Any,
    sample_rate: int,
    *,
    window_ms: float = DEFAULT_DETECT_WINDOW_MS,
    threshold_ratio: float = DEFAULT_DETECT_THRESHOLD_RATIO,
    silence_start_ms: float = DEFAULT_SILENCE_START_MS,
    silence_end_ms: float = DEFAULT_SILENCE_END_MS,
    silence_ratio: float = DEFAULT_SILENCE_RATIO,
) -> bool:
    """Return True iff the click signature is present.

    The signature is *loud first window* AND *near-silence in the
    [silence_start_ms, silence_end_ms] trail-off window*.  Operates on
    a mono mix of the input channels, which matches the perception
    case where the click is heard "at the start," not "in the left ear
    only."

    Degrades gracefully on inputs too short or too quiet to evaluate:
    if the clip is shorter than the silence-window endpoint, or global
    RMS is essentially zero, returns False.
    """
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        arr = _to_audio_ct_mlx(samples, dtype=mx.float64)
        mono = mx.mean(arr, axis=0)
        n_total = mono.shape[0]

        win = max(1, int(round(window_ms / 1000.0 * sample_rate)))
        silence_start = int(round(silence_start_ms / 1000.0 * sample_rate))
        silence_end = int(round(silence_end_ms / 1000.0 * sample_rate))

        # Need the full silence window present to evaluate the second condition.
        if n_total < silence_end or win <= 0:
            return False

        global_rms = _rms(mono)
        mx.eval(global_rms)
        global_rms_value = float(global_rms)
        if global_rms_value <= 1e-9:
            return False

        first_rms = _rms(mono[:win])
        tail_rms = _rms(mono[silence_start:silence_end])
        mx.eval(first_rms, tail_rms)
        if float(first_rms) <= threshold_ratio * global_rms_value:
            return False

        return float(tail_rms) < silence_ratio * global_rms_value
    finally:
        mx.set_default_device(old_device)


# ---------------------------------------------------------------------------
# Trim - zero-fill leading region, preserve sample count
# ---------------------------------------------------------------------------

def trim_onset(
    samples: Any,
    sample_rate: int,
    *,
    trim_ms: float = DEFAULT_TRIM_MS,
) -> mx.array:
    """Zero out the leading `trim_ms` milliseconds of every channel.

    Returns a float32 MLX array with the same shape as the input
    (modulo the (B,C,T) -> (C,T) drop of the batch axis, which the
    encoders already do too).  Sample count is preserved so video and
    audio stay in sync.

    `trim_ms <= 0` is a pass-through.  `trim_ms` is clamped to the
    clip duration: anything longer than the clip silences the entire
    track, which is the right behavior if a caller deliberately asks
    for it.
    """
    arr = _to_audio_ct_mlx(samples, dtype=mx.float32)
    if trim_ms <= 0:
        return arr
    n_zero = int(round(trim_ms / 1000.0 * sample_rate))
    n_zero = min(n_zero, arr.shape[1])
    if n_zero <= 0:
        return arr
    if n_zero == arr.shape[1]:
        return mx.zeros_like(arr)
    leading = mx.zeros((arr.shape[0], n_zero), dtype=arr.dtype)
    return mx.concatenate([leading, arr[:, n_zero:]], axis=1)


# ---------------------------------------------------------------------------
# High-level mitigation entry - what the encoders call
# ---------------------------------------------------------------------------

@dataclass
class OnsetTrimResult:
    """Outcome of an onset-mitigation call.

    `samples` is a float32 MLX array with channel layout normalized to
    (C, T). When `applied=True` the leading `trim_ms` is zero-filled.

    `detected` is the diagnostic verdict from the two-window check -
    it can be True even when `applied=False` (e.g. mode="off"), and
    False when `applied=True` (e.g. mode="force" / explicit N_ms).
    """

    samples: mx.array
    applied: bool
    detected: bool
    trim_ms: float
    mode: str
    detail: str


def mitigate_onset(
    samples: Any,
    sample_rate: int,
    *,
    mode: str = "auto",
    trim_ms: float = DEFAULT_TRIM_MS,
) -> OnsetTrimResult:
    """Apply the sequence-start onset mitigation according to `mode`.

    Modes:
      auto   detect-then-trim.  Trim is applied only when the two-window
             detector fires.  Quiet-onset clips pass through unchanged.
      off    no detection, no trim.  Returned `samples` is a (C,T)
             MLX array normalized from the input, untouched.
      force  always trim by `trim_ms`, regardless of detection.  Used
             by the CLI when the user specifies an explicit N_ms.

    `trim_ms` is the duration of the leading zero-fill applied when a
    trim occurs.  Detector parameters are not exposed here - callers
    that want non-default thresholds should call `detect_onset_spike`
    + `trim_onset` directly.
    """
    arr = _to_audio_ct_mlx(samples, dtype=mx.float32)
    mode = mode.lower()
    if mode == "off":
        return OnsetTrimResult(
            samples=arr,
            applied=False,
            detected=False,
            trim_ms=0.0,
            mode="off",
            detail="onset trim disabled",
        )

    if mode == "auto":
        is_spike = detect_onset_spike(arr, sample_rate)
        if not is_spike:
            return OnsetTrimResult(
                samples=arr,
                applied=False,
                detected=False,
                trim_ms=0.0,
                mode="auto",
                detail="no onset spike detected",
            )
        trimmed = trim_onset(arr, sample_rate, trim_ms=trim_ms)
        return OnsetTrimResult(
            samples=trimmed,
            applied=True,
            detected=True,
            trim_ms=float(trim_ms),
            mode="auto",
            detail=f"onset spike detected; zero-filled leading {trim_ms:g} ms",
        )

    if mode == "force":
        trimmed = trim_onset(arr, sample_rate, trim_ms=trim_ms)
        # Still report the diagnostic verdict so the run-log captures
        # whether the forced trim was actually needed.
        is_spike = detect_onset_spike(arr, sample_rate)
        return OnsetTrimResult(
            samples=trimmed,
            applied=True,
            detected=is_spike,
            trim_ms=float(trim_ms),
            mode="force",
            detail=f"forced zero-fill of leading {trim_ms:g} ms",
        )

    raise ValueError(
        f"Unknown onset mitigation mode {mode!r}.  Expected one of: "
        f"auto, off, force."
    )


# ---------------------------------------------------------------------------
# Latent-domain onset mitigation (causal-VAE sequence-start boundary spike)
# ---------------------------------------------------------------------------
# The audio VAE is causal in time, so its first latent frame is encoded against
# zero-padding (a cold-start boundary).  In the 8-channel audio latent this lands
# as a concentrated single-channel spike at frame 0 that the vocoder renders as a
# broadband t=0 click.  Total frame-0 energy is normal -- it is the per-channel
# concentration that clicks -- so the waveform RMS detector above misses it.
# Flattening the leading frames to the typical (time-mean) frame removes it at
# the source.  See docs/AUDIO_ISSUES.md -> "what predicts a click".

DEFAULT_LATENT_FLATTEN_FRAMES = 3
DEFAULT_LATENT_SPIKE_RATIO = 2.0
DEFAULT_LATENT_CONCENTRATION = 1.8


def _audio_latent_ct(latent: mx.array) -> mx.array:
    """(B, C, T, F) or (C, T, F) -> (C, T, F)."""
    return latent[0] if latent.ndim == 4 else latent


def detect_onset_latent_spike(
    latent: mx.array,
    *,
    ratio: float = DEFAULT_LATENT_SPIKE_RATIO,
    concentration: float = DEFAULT_LATENT_CONCENTRATION,
) -> bool:
    """True iff the first audio-latent frame has a concentrated single-channel spike.

    Per-channel-per-frame RMS; a channel's frame-0 energy must exceed `ratio` x
    that channel's mean over all frames (anomalous start), AND the frame-0
    per-channel profile must be concentrated -- peak channel > `concentration` x
    the mean channel -- so a broad all-channel elevation (which decodes quietly)
    does not trip it.
    """
    a = _audio_latent_ct(latent).astype(mx.float32)
    if a.shape[1] < 4:
        return False
    perch = mx.sqrt(mx.mean(a * a, axis=2))            # (C, T)
    base = mx.mean(perch, axis=1) + 1e-9               # (C,)
    f0 = perch[:, 0] / base                            # (C,)
    peak = float(mx.max(f0))
    mean = float(mx.mean(f0))
    return peak > ratio and peak > concentration * mean


def flatten_onset_latent(
    latent: mx.array, *, n_frames: int = DEFAULT_LATENT_FLATTEN_FRAMES
) -> mx.array:
    """Replace the leading `n_frames` audio-latent frames with the time-mean frame.

    Returns a NEW array; the input (e.g. a saved sidecar) is never mutated, and
    the frame count is preserved.
    """
    if n_frames <= 0:
        return latent
    has_batch = latent.ndim == 4
    x = latent if has_batch else latent[None]
    n = min(n_frames, x.shape[2])
    rep = mx.mean(x, axis=2, keepdims=True)            # (B, C, 1, F): the typical frame
    rep = mx.broadcast_to(rep, (x.shape[0], x.shape[1], n, x.shape[3])).astype(x.dtype)
    out = mx.concatenate([rep, x[:, :, n:, :]], axis=2)
    return out if has_batch else out[0]


def mitigate_onset_latent(
    latent: mx.array,
    *,
    mode: str = "auto",
    n_frames: int = DEFAULT_LATENT_FLATTEN_FRAMES,
    verbose: bool = False,
) -> mx.array:
    """Apply the latent-domain onset flatten per `mode` (auto / off / force).

    auto   flatten only when `detect_onset_latent_spike` fires.
    off    pass through unchanged.
    force  always flatten the leading `n_frames`.

    Decode-time mitigation: the returned latent is what the VAE decodes, NOT what
    is saved as a sidecar -- callers must keep the original for the sidecar.
    """
    m = (mode or "auto").lower()
    if m == "off":
        return latent
    if m == "force" or (m == "auto" and detect_onset_latent_spike(latent)):
        if verbose:
            why = "forced" if m == "force" else "sequence-start spike detected"
            print(f"  audio onset (latent): flattened leading {n_frames} frames ({why})")
        return flatten_onset_latent(latent, n_frames=n_frames)
    return latent


# ---------------------------------------------------------------------------
# CLI parsing - accepts "auto" / "off" / "<float ms>"
# ---------------------------------------------------------------------------

def parse_trim_mode(spec: str) -> tuple[str, float]:
    """Map the CLI value to (mode, trim_ms).

    Accepted spellings:
      "auto"        -> ("auto", DEFAULT_TRIM_MS)
      "off" / "0"   -> ("off",  0.0)
      "<float>"     -> ("force", float(spec))    # ms, e.g. "150"

    A bare integer / float is interpreted as "force this duration."
    Negative values raise ValueError.
    """
    s = spec.strip().lower()
    if s == "auto":
        return ("auto", DEFAULT_TRIM_MS)
    if s in ("off", "none"):
        return ("off", 0.0)
    try:
        value = float(s)
    except ValueError:
        raise ValueError(
            f"Invalid --audio-onset-trim value {spec!r}.  Expected "
            f"'auto', 'off', or a duration in milliseconds (e.g. '120')."
        ) from None
    if value < 0:
        raise ValueError(
            f"--audio-onset-trim must be non-negative; got {value}"
        )
    if value == 0:
        return ("off", 0.0)
    return ("force", value)
