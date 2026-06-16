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
dropping samples — dropping would shift audio relative to video and
break lip sync.

Detection
---------
The signature is *loud burst followed by silence*, not just *loud
burst*.  A real speech onset at t=0 would also be loud at t=0 but
would NOT be followed by silence — it would sustain.  So we use a
two-window check:

  first 50 ms RMS > 2.0x global RMS
  AND
  mean RMS over 100-250 ms < 0.1x global RMS

A legitimate loud-speech onset trips condition 1 but not 2 (sustained
content keeps the 100-250 ms RMS near global RMS), so the trim does
not fire and lip sync at t=0 is preserved.

Trim
----
The trim zeros out the leading 120 ms by default — long enough to
clear the diagnosed click's ~95 ms decay tail with margin, short
enough to sit comfortably inside the intentional silence the model
places before the first spoken word (95-250 ms on the diagnosed clip).
Sample count is preserved.

Public API
----------
- `detect_onset_spike(samples, sample_rate, ...)` -> bool
- `trim_onset(samples, sample_rate, *, trim_ms)` -> trimmed ndarray
- `mitigate_onset(samples, sample_rate, *, mode, trim_ms)` -> result
- `parse_trim_mode(s)` -> (mode, trim_ms) for the CLI flag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Threshold constants — single source of truth for both this module and
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

def _to_mono_2d(samples: Any) -> np.ndarray:
    """Coerce (B,C,T) / (C,T) / (T,) -> (C,T) float32 ndarray, no copy when possible."""
    arr = np.asarray(samples)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            f"audio_waveform must be (B,C,T), (C,T), or (T,); got shape {arr.shape}"
        )
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


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
    arr = _to_mono_2d(samples)
    mono = arr.mean(axis=0)
    n_total = mono.shape[0]

    win = max(1, int(round(window_ms / 1000.0 * sample_rate)))
    silence_start = int(round(silence_start_ms / 1000.0 * sample_rate))
    silence_end = int(round(silence_end_ms / 1000.0 * sample_rate))

    # Need the full silence window present to evaluate the second condition.
    if n_total < silence_end or win <= 0:
        return False

    global_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    if global_rms <= 1e-9:
        return False

    first = mono[:win].astype(np.float64)
    first_rms = float(np.sqrt(np.mean(first ** 2)))
    if first_rms <= threshold_ratio * global_rms:
        return False

    tail = mono[silence_start:silence_end].astype(np.float64)
    tail_rms = float(np.sqrt(np.mean(tail ** 2)))
    return tail_rms < silence_ratio * global_rms


# ---------------------------------------------------------------------------
# Trim — zero-fill leading region, preserve sample count
# ---------------------------------------------------------------------------

def trim_onset(
    samples: Any,
    sample_rate: int,
    *,
    trim_ms: float = DEFAULT_TRIM_MS,
) -> np.ndarray:
    """Zero out the leading `trim_ms` milliseconds of every channel.

    Returns a *new* float32 ndarray with the same shape as the input
    (modulo the (B,C,T) -> (C,T) drop of the batch axis, which the
    encoders already do too).  Sample count is preserved so video and
    audio stay in sync.

    `trim_ms <= 0` is a pass-through.  `trim_ms` is clamped to the
    clip duration: anything longer than the clip silences the entire
    track, which is the right behavior if a caller deliberately asks
    for it.
    """
    arr = _to_mono_2d(samples).copy()
    if trim_ms <= 0:
        return arr
    n_zero = int(round(trim_ms / 1000.0 * sample_rate))
    n_zero = min(n_zero, arr.shape[1])
    if n_zero > 0:
        arr[:, :n_zero] = 0.0
    return arr


# ---------------------------------------------------------------------------
# High-level mitigation entry — what the encoders call
# ---------------------------------------------------------------------------

@dataclass
class OnsetTrimResult:
    """Outcome of an onset-mitigation call.

    `samples` is always a fresh ndarray (callers can mutate without
    aliasing the input).  When `applied=False` it is a copy of the
    input with the channel layout normalized to (C, T); when
    `applied=True` the leading `trim_ms` is zero-filled.

    `detected` is the diagnostic verdict from the two-window check —
    it can be True even when `applied=False` (e.g. mode="off"), and
    False when `applied=True` (e.g. mode="force" / explicit N_ms).
    """

    samples: np.ndarray
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
             ndarray copy of the input, untouched.
      force  always trim by `trim_ms`, regardless of detection.  Used
             by the CLI when the user specifies an explicit N_ms.

    `trim_ms` is the duration of the leading zero-fill applied when a
    trim occurs.  Detector parameters are not exposed here — callers
    that want non-default thresholds should call `detect_onset_spike`
    + `trim_onset` directly.
    """
    arr = _to_mono_2d(samples)
    mode = mode.lower()
    if mode == "off":
        return OnsetTrimResult(
            samples=arr.copy(),
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
                samples=arr.copy(),
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
# CLI parsing — accepts "auto" / "off" / "<float ms>"
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
        )
    if value < 0:
        raise ValueError(
            f"--audio-onset-trim must be non-negative; got {value}"
        )
    if value == 0:
        return ("off", 0.0)
    return ("force", value)
