#!/usr/bin/env python3
"""Tests for the sequence-start audio onset detector + trim mitigation.

Covers `LTX_2_MLX.audio.onset` - the module both encoders (ffmpeg and
VideoToolbox) and the diagnostic script `analyze_audio_onset.py` share.

Synthetic waveform fixtures model the documented signal classes from
`docs/AUDIO_ISSUES.md` -> "Sequence-Start Audio Spike":

  * click_signature   loud 60 ms burst, then silence, then steady speech
                      -- detector MUST fire
  * ambient_onset     quiet noise throughout, no burst at t=0
                      -- detector MUST NOT fire
  * loud_speech       loud sustained content from t=0 onward (no silent
                      gap) -- detector MUST NOT fire (this is the
                      AV-sync-safety case: a real loud onset shouldn't
                      get trimmed)
  * silent            all zeros -- detector MUST NOT fire (no div-by-0)
  * too_short         100 ms clip, can't even evaluate the silence
                      window -- detector MUST NOT fire

Then the trim function is checked: sample count preserved, leading region
zeroed, trailing region untouched.

Then the mitigation entry point is checked: auto / off / force modes
behave as documented, and the OnsetTrimResult carries the expected
diagnostic flags.

Lastly the CLI parser is sanity-checked.

Exit code: 0 on pass, non-zero on any failed assertion.

Usage:
    scripts/test_audio_onset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running directly from the repo root without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LTX_2_MLX.audio.onset import (
    DEFAULT_DETECT_THRESHOLD_RATIO,
    DEFAULT_DETECT_WINDOW_MS,
    DEFAULT_SILENCE_END_MS,
    DEFAULT_SILENCE_RATIO,
    DEFAULT_SILENCE_START_MS,
    DEFAULT_TRIM_MS,
    detect_onset_spike,
    mitigate_onset,
    parse_trim_mode,
    trim_onset,
)

_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"{name}{(' - ' + detail) if detail else ''}"
        print(f"  FAIL  {msg}")
        _FAILURES.append(msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 48000
RNG = np.random.default_rng(seed=0)


def make_click_signature(duration_s: float = 5.0, burst_ms: float = 60.0,
                         burst_amp: float = 0.5,
                         silence_until_ms: float = 250.0,
                         speech_amp: float = 0.15) -> np.ndarray:
    """Loud 60 ms burst, ~190 ms silence, sustained quiet speech.

    Matches the canonical click signature documented in docs/AUDIO_ISSUES.md
    (dialog-heavy clip class): first 50 ms RMS >> global, 100-250 ms
    RMS ~ 0, then normal speech RMS from 250 ms onward.
    """
    n = int(duration_s * SR)
    out = np.zeros((2, n), dtype=np.float32)
    burst_n = int(burst_ms / 1000.0 * SR)
    silence_n = int(silence_until_ms / 1000.0 * SR)
    # Loud burst at start.  Use bandlimited noise so the global RMS
    # picks it up but it's not a pure DC offset.
    out[:, :burst_n] = (RNG.normal(0, 1, size=(2, burst_n)) * burst_amp).astype(np.float32)
    # Steady "speech" content after the silent gap.
    speech_n = n - silence_n
    if speech_n > 0:
        out[:, silence_n:] = (
            RNG.normal(0, 1, size=(2, speech_n)) * speech_amp
        ).astype(np.float32)
    return out


def make_ambient_onset(duration_s: float = 5.0, amp: float = 0.02) -> np.ndarray:
    """Quiet noise throughout - no transient at t=0."""
    n = int(duration_s * SR)
    return (RNG.normal(0, 1, size=(2, n)) * amp).astype(np.float32)


def make_loud_speech_from_zero(duration_s: float = 5.0, amp: float = 0.3) -> np.ndarray:
    """Sustained loud content from t=0 with no silent gap.

    This is the AV-sync-safety case: real speech that begins at t=0
    must NOT trigger the trim, because trimming would zero out the
    first 120 ms of speech and the first half-word would be missing.
    """
    n = int(duration_s * SR)
    return (RNG.normal(0, 1, size=(2, n)) * amp).astype(np.float32)


def make_silent(duration_s: float = 5.0) -> np.ndarray:
    return np.zeros((2, int(duration_s * SR)), dtype=np.float32)


def make_too_short(duration_ms: float = 100.0, amp: float = 0.5) -> np.ndarray:
    """Clip shorter than the silence-window endpoint (250 ms)."""
    n = int(duration_ms / 1000.0 * SR)
    return (RNG.normal(0, 1, size=(2, n)) * amp).astype(np.float32)


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

def test_detector_classifies_signal_classes() -> None:
    print("\n[1] detector classifies each signal class correctly")

    click = make_click_signature()
    check("click signature detected", detect_onset_spike(click, SR))

    ambient = make_ambient_onset()
    check("ambient onset NOT detected", not detect_onset_spike(ambient, SR))

    loud_speech = make_loud_speech_from_zero()
    check(
        "loud speech from t=0 NOT detected (AV sync safety)",
        not detect_onset_spike(loud_speech, SR),
    )

    silent = make_silent()
    check("all-silent clip NOT detected", not detect_onset_spike(silent, SR))

    short = make_too_short()
    check(
        "too-short clip NOT detected (degrades gracefully)",
        not detect_onset_spike(short, SR),
    )


def test_detector_accepts_shape_variants() -> None:
    print("\n[2] detector accepts (B,C,T), (C,T), (T,) input shapes")
    click_ct = make_click_signature()
    click_bct = click_ct[None, ...]
    click_t = click_ct.mean(axis=0)
    check("(C,T) -> detected", detect_onset_spike(click_ct, SR))
    check("(B,C,T) -> detected", detect_onset_spike(click_bct, SR))
    check("(T,) mono -> detected", detect_onset_spike(click_t, SR))


def test_detector_threshold_responds_to_kwargs() -> None:
    """Bumping the threshold ratio above the click's first-window/global
    ratio must disable detection; lowering it below the loud-speech
    ratio must (incorrectly) detect - proving the threshold is live.
    """
    print("\n[3] detector responds to threshold kwargs")
    click = make_click_signature()
    # Click's first-window RMS / global RMS is in the 2.5-3.5x range.
    check(
        "click NOT detected at threshold=20x (way above)",
        not detect_onset_spike(click, SR, threshold_ratio=20.0),
    )
    loud = make_loud_speech_from_zero()
    # Loud-speech first-window RMS ~ global RMS; with a 0.1x threshold
    # the first condition trips but the silence condition still rejects.
    check(
        "loud speech still NOT detected at threshold=0.1x (silence cond rejects)",
        not detect_onset_spike(loud, SR, threshold_ratio=0.1),
    )


# ---------------------------------------------------------------------------
# Trim tests
# ---------------------------------------------------------------------------

def test_trim_preserves_sample_count() -> None:
    """The trim MUST mute, never drop, samples - AV sync depends on it."""
    print("\n[4] trim preserves sample count (mute, never drop)")
    click = make_click_signature()
    n_in = click.shape[1]
    trimmed = trim_onset(click, SR, trim_ms=120.0)
    check(
        f"sample count preserved (in={n_in}, out={trimmed.shape[1]})",
        trimmed.shape[1] == n_in,
    )
    n_zero = int(120.0 / 1000.0 * SR)
    check(
        f"leading {n_zero} samples are exactly zero",
        np.all(trimmed[:, :n_zero] == 0.0),
    )
    check(
        "samples past the trim region are untouched",
        np.allclose(trimmed[:, n_zero:], click[:, n_zero:]),
    )


def test_trim_zero_passthrough() -> None:
    print("\n[5] trim_ms<=0 returns a faithful copy")
    click = make_click_signature()
    out_zero = trim_onset(click, SR, trim_ms=0.0)
    out_neg = trim_onset(click, SR, trim_ms=-5.0)
    check("trim_ms=0 -> identical content", np.array_equal(out_zero, click))
    check("trim_ms<0 -> identical content", np.array_equal(out_neg, click))
    check("returned ndarray is a copy, not the input", out_zero is not click)


def test_trim_clamps_to_clip_length() -> None:
    print("\n[6] trim clamps to clip length when oversized")
    click = make_click_signature(duration_s=0.5)
    trimmed = trim_onset(click, SR, trim_ms=10_000.0)
    check("over-trim does not raise", trimmed.shape == click.shape)
    check("entire clip is silenced", np.all(trimmed == 0.0))


# ---------------------------------------------------------------------------
# Mitigation entry-point tests
# ---------------------------------------------------------------------------

def test_mitigate_modes() -> None:
    print("\n[7] mitigate_onset modes: auto / off / force")

    click = make_click_signature()
    ambient = make_ambient_onset()

    # auto fires on a click clip.
    r_auto_click = mitigate_onset(click, SR, mode="auto", trim_ms=120.0)
    n_zero = int(120.0 / 1000.0 * SR)
    check("auto + click: applied=True", r_auto_click.applied)
    check("auto + click: detected=True", r_auto_click.detected)
    check(
        "auto + click: leading samples zeroed",
        np.all(r_auto_click.samples[:, :n_zero] == 0.0),
    )
    check(
        "auto + click: sample count preserved",
        r_auto_click.samples.shape[1] == click.shape[1],
    )

    # auto passes ambient through untouched.
    r_auto_amb = mitigate_onset(ambient, SR, mode="auto", trim_ms=120.0)
    check("auto + ambient: applied=False", not r_auto_amb.applied)
    check("auto + ambient: detected=False", not r_auto_amb.detected)
    check(
        "auto + ambient: content unchanged",
        np.allclose(r_auto_amb.samples, ambient),
    )

    # off never trims.
    r_off = mitigate_onset(click, SR, mode="off")
    check("off + click: applied=False", not r_off.applied)
    check("off + click: detected=False", not r_off.detected)
    check(
        "off + click: content unchanged",
        np.allclose(r_off.samples, click),
    )

    # force trims regardless of detection - even on a clean ambient clip.
    r_force = mitigate_onset(ambient, SR, mode="force", trim_ms=80.0)
    n_zero_force = int(80.0 / 1000.0 * SR)
    check("force + ambient: applied=True", r_force.applied)
    check(
        "force + ambient: leading 80 ms zeroed",
        np.all(r_force.samples[:, :n_zero_force] == 0.0),
    )
    check(
        "force + ambient: detected=False (diag still ran)",
        not r_force.detected,
    )

    # Unknown mode is a hard error.
    try:
        mitigate_onset(click, SR, mode="bogus")
        check("unknown mode raises ValueError", False, "no exception")
    except ValueError:
        check("unknown mode raises ValueError", True)


def test_mitigate_returns_fresh_array() -> None:
    """OnsetTrimResult.samples must be a *new* ndarray so the caller
    can mutate / convert dtype without aliasing the upstream pipeline
    output.
    """
    print("\n[8] mitigate_onset returns a fresh ndarray (no aliasing)")
    click = make_click_signature()
    r = mitigate_onset(click, SR, mode="off")
    r.samples[:, 0] = 999.0
    check(
        "mutating result.samples does not corrupt input",
        not np.any(click[:, 0] == 999.0),
    )


# ---------------------------------------------------------------------------
# CLI parser tests
# ---------------------------------------------------------------------------

def test_parse_trim_mode() -> None:
    print("\n[9] parse_trim_mode handles auto / off / numeric / errors")
    check("auto -> (auto, default)", parse_trim_mode("auto") == ("auto", DEFAULT_TRIM_MS))
    check("AUTO (caps) -> (auto, default)", parse_trim_mode("AUTO") == ("auto", DEFAULT_TRIM_MS))
    check("off -> (off, 0)", parse_trim_mode("off") == ("off", 0.0))
    check("none -> (off, 0)", parse_trim_mode("none") == ("off", 0.0))
    check("'0' -> (off, 0)", parse_trim_mode("0") == ("off", 0.0))
    check("'150' -> (force, 150)", parse_trim_mode("150") == ("force", 150.0))
    check("'80.5' -> (force, 80.5)", parse_trim_mode("80.5") == ("force", 80.5))

    for bogus in ("yes", "abc", "1.2.3"):
        try:
            parse_trim_mode(bogus)
            check(f"bogus value {bogus!r} raises", False, "no exception")
        except ValueError:
            check(f"bogus value {bogus!r} raises", True)

    try:
        parse_trim_mode("-5")
        check("negative value raises", False, "no exception")
    except ValueError:
        check("negative value raises", True)


# ---------------------------------------------------------------------------
# Constants surface
# ---------------------------------------------------------------------------

def test_constants_are_exposed() -> None:
    """The analyzer script and the mitigation share the same threshold
    constants - verify they exist on the public surface.
    """
    print("\n[10] documented constants are importable")
    check("DEFAULT_DETECT_WINDOW_MS == 50", DEFAULT_DETECT_WINDOW_MS == 50.0)
    check("DEFAULT_DETECT_THRESHOLD_RATIO == 2", DEFAULT_DETECT_THRESHOLD_RATIO == 2.0)
    check("DEFAULT_SILENCE_START_MS == 100", DEFAULT_SILENCE_START_MS == 100.0)
    check("DEFAULT_SILENCE_END_MS == 250", DEFAULT_SILENCE_END_MS == 250.0)
    check("DEFAULT_SILENCE_RATIO == 0.1", DEFAULT_SILENCE_RATIO == 0.1)
    check("DEFAULT_TRIM_MS == 120", DEFAULT_TRIM_MS == 120.0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("Running audio-onset tests...")
    test_detector_classifies_signal_classes()
    test_detector_accepts_shape_variants()
    test_detector_threshold_responds_to_kwargs()
    test_trim_preserves_sample_count()
    test_trim_zero_passthrough()
    test_trim_clamps_to_clip_length()
    test_mitigate_modes()
    test_mitigate_returns_fresh_array()
    test_parse_trim_mode()
    test_constants_are_exposed()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} test(s) FAILED:")
        for msg in _FAILURES:
            print(f"  - {msg}")
        return 1
    print("All audio-onset tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
