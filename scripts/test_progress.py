#!/usr/bin/env python3
"""Quick verification of LTX_2_MLX.videotoolbox.progress.PhaseBar.

Runs a battery of assertions on PhaseBar's timing logic (deferred start,
first-tick measurement, idempotent start, math-consistent pace/ETA,
fixed bar width). Total runtime ~2 seconds.

With --demo, also renders a stacked-bar visualization that mimics the
VAE+VSR interleave from scripts/vsr_harness.py so the live appearance
can be eyeballed (bar widths line up, postfix shows real pace at 1/N,
ETA decreases as ticks accumulate). Adds ~2 seconds.

Exit code: 0 on pass, non-zero on any failed assertion.

Usage:
    scripts/test_progress.py            # assertions only
    scripts/test_progress.py --demo     # + visual demo
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import Optional


# Allow running directly from the repo root without an install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.videotoolbox.progress import (  # noqa: E402
    PhaseBar, StackedPhaseBars, _fmt_duration,
)


# ---------------------------------------------------------------------------
# Tiny test framework: a failing assert prints a "FAIL" line and bumps the
# exit code, but the rest of the suite continues so a single regression
# doesn't hide later ones.
# ---------------------------------------------------------------------------

_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"{name}{(' — ' + detail) if detail else ''}"
        print(f"  FAIL  {msg}")
        _FAILURES.append(msg)


def near(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def test_first_tick_records_duration() -> None:
    """A bar at 1/N must show a measured pace, not the n=0 'measuring'
    placeholder.  Verifies that the first `update()` advances `_t_last`
    so elapsed becomes non-zero — pace is derived from
    `n / (now - origin)`."""
    print("\n[1] first tick advances the clock past origin")
    buf = io.StringIO()
    with redirect_stderr(buf):
        b = PhaseBar(total=5, desc="t", unit="it", mininterval=0.0)
        time.sleep(0.05)
        b.update(1)
        elapsed = (b._t_last or 0) - b._t_origin
        check(
            "_t_last set on first update",
            b._t_last is not None,
        )
        check(
            "elapsed since origin ≈ time slept before first update",
            near(elapsed, 0.05, 0.04),
            f"got {elapsed * 1000:.1f} ms (expected ~50 ms)",
        )
        # Pace is shown as "X.Xs/<unit>" or "<f.2><unit>/s" once n>=1;
        # the n=0 placeholder is "measuring".
        check(
            "line shows a measured pace (not the 'measuring' placeholder)",
            "measuring" not in b._build_line(),
            f"line={b._build_line()!r}",
        )
        b.close()


def _parse_duration(s: str) -> Optional[float]:
    """Parse the dashed clock format (`--:--:SS`, `--:MM:SS`, `HH:MM:SS`)
    back to seconds. Substitutes `--` → `00` then parses HH:MM:SS."""
    s = s.strip().replace("--", "00")
    m = re.match(r"^(\d+):(\d{2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return None


def test_fmt_duration_covers_all_scales() -> None:
    """`_fmt_duration` returns `HH:MM:SS` with leading-zero groups
    replaced by `--:` so the active fields stand out. Always 8 chars
    up to 99:59:59 — no use-site padding needed.
    """
    print("\n[8] _fmt_duration: dashed HH:MM:SS, 8 chars constant")
    cases = [
        # (seconds, expected_string)
        (0.0,        "--:--:00"),
        (0.05,       "--:--:00"),  # sub-second rounds to whole secs
        (0.6,        "--:--:01"),  # round-half-up
        (12.3,       "--:--:12"),
        (59.4,       "--:--:59"),
        (59.5,       "--:01:00"),  # carry past minute reveals MM field
        (60.0,       "--:01:00"),
        (60.5,       "--:01:01"),  # round-half-up (not banker's-rounding)
        (83.0,       "--:01:23"),
        (3599.0,     "--:59:59"),
        (3599.5,     "01:00:00"),  # carry past hour reveals HH field
        (3600.0,     "01:00:00"),
        (3660.0,     "01:01:00"),
        (86400.0,    "24:00:00"),  # 24 hours — primary target
        (90061.0,    "25:01:01"),
        (359999.0,   "99:59:59"),  # just under 100h
        (360000.0,   "100:00:00"), # 100h+ — column overflows by 1
    ]
    for seconds, expected in cases:
        got = _fmt_duration(seconds)
        check(
            f"_fmt_duration({seconds:>9}) == {expected!r}",
            got == expected,
            f"got {got!r}",
        )
    # All non-overflow durations are intrinsically 8 chars wide.
    for seconds in (0.0, 59.9, 60.0, 3599.0, 3600.0, 86400.0, 359999.0):
        got = _fmt_duration(seconds)
        check(
            f"{seconds:>8}s → {got!r} is exactly 8 chars",
            len(got) == 8,
            f"got len={len(got)}",
        )


def test_eta_uses_post_update_count() -> None:
    """ETA at 1/N with constant pace t should be ~(N-1)·t, not N·t.
    Regression for the bug where the count used for ETA hadn't been
    incremented yet at the moment of render."""
    print("\n[4] ETA reflects post-update count")
    buf = io.StringIO()
    with redirect_stderr(buf):
        # 1.0 s/tick so ETA differences survive HH:MM:SS rounding:
        # 3 remaining × 1.0 → 00:00:03 (fixed) vs 4 × 1.0 → 00:00:04 (bug).
        b = PhaseBar(total=4, desc="t", unit="it", mininterval=0.0)
        time.sleep(1.0)
        b.update(1)
        line = b._build_line()
        # Line shape: "... | ETA <HH:MM:SS> | <pace>" — match the field
        # between "ETA " and " |".
        m = re.search(r"ETA\s+(\S+)\s+\|", line)
        check("line contains ETA field", m is not None, f"line={line!r}")
        if m:
            eta = _parse_duration(m.group(1))
            check("ETA field is parseable", eta is not None, f"got {m.group(1)!r}")
            if eta is not None:
                check(
                    "eta ≈ remaining × median (not (remaining+1) × median)",
                    near(eta, 3.0, 1.0),
                    f"got eta={eta}s (expected ~3s, would be ~4s on the bug)",
                )
        b.close()


def test_pace_elapsed_count_are_math_consistent() -> None:
    """The OCD invariant: at any moment, `pace × elapsed ≈ n_so_far` and
    `eta × pace ≈ total - n_so_far` (within rounding noise).

    Regression: an earlier median-of-window implementation displayed an
    "instantaneous" pace that didn't multiply out to the work actually
    done — fine for a stable phase but dishonest across cross-phase
    waits (e.g. VSR pace × RUN underestimated frames done by ~30%).
    """
    print("\n[5] pace × elapsed ≈ n; eta × pace ≈ remaining")
    buf = io.StringIO()
    with redirect_stderr(buf):
        b = PhaseBar(total=30, desc="t", unit="it", mininterval=0.0)
        for _ in range(8):
            time.sleep(0.005)
            b.update(1)
        # Inject one outlier (~100 ms, 20× the normal interval).
        time.sleep(0.1)
        b.update(1)
        for _ in range(8):
            time.sleep(0.005)
            b.update(1)
        elapsed = (b._t_last or 0) - b._t_origin
        sec_per_unit = elapsed / b._n
        # pace × elapsed must equal n exactly (no slop — same arithmetic).
        check(
            "pace × elapsed == n_so_far (exact)",
            abs(sec_per_unit * b._n - elapsed) < 1e-9,
            f"pace*elapsed={sec_per_unit * b._n}, elapsed={elapsed}",
        )
        # ETA × pace ≈ remaining (within rounding from int-second display).
        # The displayed ETA = remaining * sec_per_unit (same math), so
        # exact at the model layer; rounding happens in _fmt_duration.
        remaining = 30 - b._n
        eta_seconds = remaining * sec_per_unit
        check(
            "model eta ≈ remaining × sec_per_unit",
            near(eta_seconds, remaining * sec_per_unit, 1e-9),
            f"eta={eta_seconds}, remaining={remaining}, spu={sec_per_unit}",
        )
        b.close()


def test_line_columns_do_not_jitter() -> None:
    """As n grows from 1-digit to 3-digit values and elapsed crosses the
    minute threshold, every section's width must stay constant — that's
    the whole point of the fixed-width allocation. Measured by section
    length, splitting on ` | ` (the unambiguous separator between
    prefix / STEP1 / RUN / ETA / pace).

    The prefix section is `<indent><label> [<bar>] <count> <pct>` — its
    width depends on indent + label (both fixed for one bar), bar_width
    (fixed), count_str (fixed via n_width), and pct_str (fixed `5.1f%`).
    """
    print("\n[6a] line sections are fixed-width across all tick counts")
    buf = io.StringIO()
    prefix_widths: set[int] = set()
    pct_widths: set[int] = set()
    run_widths: set[int] = set()
    eta_widths: set[int] = set()
    pace_widths: set[int] = set()
    with redirect_stderr(buf):
        b = PhaseBar(total=200, desc="t", unit="it", mininterval=0.0)
        for i in range(1, 151):
            # Vary pace so RUN/ETA stress different format branches.
            time.sleep(0.001 + (0.01 if i in (5, 50, 100) else 0))
            b.update(1)
            sections = b._build_line().split(" | ")
            if len(sections) >= 5:
                prefix_widths.add(len(sections[0]))
                pct_widths.add(len(sections[1]))
                run_widths.add(len(sections[2]))
                eta_widths.add(len(sections[3]))
                pace_widths.add(len(sections[4]))
        b.close()
    check(
        "prefix (indent + label + [bar] + count) constant width",
        len(prefix_widths) == 1,
        f"saw widths {sorted(prefix_widths)}",
    )
    check(
        "pct section constant width",
        len(pct_widths) == 1,
        f"saw widths {sorted(pct_widths)}",
    )
    check(
        "RUN section constant width",
        len(run_widths) == 1,
        f"saw widths {sorted(run_widths)}",
    )
    check(
        "ETA section constant width",
        len(eta_widths) == 1,
        f"saw widths {sorted(eta_widths)}",
    )
    check(
        "pace section constant width",
        len(pace_widths) == 1,
        f"saw widths {sorted(pace_widths)}",
    )


def test_count_column_scales_with_total() -> None:
    """The count column width is `max(3, len(str(total))) * 2 + 1` —
    verify large totals (9999, 10_000+) don't truncate or overflow."""
    print("\n[6b] count column scales to large totals (9999, 10000+)")
    for total, max_n_token in [
        (3,     "3/3"),          # 3 chars; padded into a 7-char slot
        (80,    "80/80"),        # 5 chars; padded into a 7-char slot
        (999,   "999/999"),      # 7 chars; fills the 7-char slot exactly
        (9999,  "9999/9999"),    # 9 chars; n_width grows to 4 → 9-char slot
        (10000, "10000/10000"),  # 11 chars; n_width grows to 5 → 11-char slot
    ]:
        buf = io.StringIO()
        with redirect_stderr(buf):
            b = PhaseBar(total=total, desc="t", unit="it", mininterval=0.0)
            b._n = total  # jump straight to "done" for the worst case
            b._t_last = b._t_origin + 0.1  # fake an elapsed so window works
            b._t_last = b._t_origin + 0.1
            line = b._build_line()
            b.close()
        # Extract just the count column — between "] " (after the bar) and " |".
        m = re.search(r"\] +(\S+(?:/\S+)?) +\|", line)
        check(
            f"total={total}: count slot contains the largest token {max_n_token!r}",
            m is not None and max_n_token in m.group(0),
            f"line slice: {m.group(0) if m else '<no match>'}",
        )


def test_bar_width_pinned() -> None:
    """The rendered `[####----]` segment must be exactly `bar_width`
    chars regardless of the rest of the line — that's why we build the
    bar string ourselves rather than letting it stretch to fill."""
    print("\n[6] visual bar width matches bar_width parameter")
    buf = io.StringIO()
    with redirect_stderr(buf):
        b = PhaseBar(total=5, desc="t", unit="it", mininterval=0.0, bar_width=28)
        m = re.search(r"\[([#-]+)\]", b._build_line())
        check("rendered line has [####----] segment", m is not None)
        if m:
            check(
                "bar segment is exactly bar_width chars",
                len(m.group(1)) == 28,
                f"got {len(m.group(1))} chars (expected 28)",
            )
        b.close()


def test_stacked_bars_share_count_column() -> None:
    """Stacked bars with very different totals must share a count column
    width so the `|` divider after the count aligns across rows. e.g.
    total=4 and total=9999 in the same stack: both render counts in a
    9-char slot (the max of their natural widths)."""
    print("\n[7a] stacked bars share count column across disparate totals")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            small = bars.add(total=4, desc="small", unit="it", mininterval=0.0)
            huge = bars.add(total=9999, desc="huger", unit="it", mininterval=0.0)
            small._n = 4
            huge._n = 9999
            small._t_last = small._t_origin + 0.1
            huge._t_last = huge._t_origin + 0.1
            small_line = small._build_line()
            huge_line = huge._build_line()
    # The "| " after the count column must be at the same string index
    # in both lines (label is same length, bar is fixed-width, so the
    # only thing that could shift the divider is the count slot width).
    small_pipe = small_line.find(" | ")
    huge_pipe = huge_line.find(" | ")
    check(
        "first ` | ` divider lands at the same column index in both bars",
        small_pipe == huge_pipe and small_pipe > 0,
        f"small={small_pipe}, huge={huge_pipe}",
    )
    check(
        "count slot widened to fit 9999/9999",
        "9999/9999" in huge_line,
        f"huge_line={huge_line!r}",
    )


def test_stacked_bars_share_visual_width() -> None:
    """Two stacked bars whose line lengths differ (different total widths,
    different unit names, etc.) must still render `[####----]` segments
    of the same visual width — driven by the same `bar_width` parameter."""
    print("\n[7] stacked bars share visual width across line lengths")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            short = bars.add(total=3, desc="short", unit="it", mininterval=0.0)
            wide = bars.add(total=999, desc="wider", unit="frame", mininterval=0.0)
            time.sleep(0.05)
            short.update(1)
            for _ in range(50):
                time.sleep(0.001)
                wide.update(1)

            def bar_seg(line: str) -> int:
                m = re.search(r"\[([#-]+)\]", line)
                return len(m.group(1)) if m else -1

            ws = bar_seg(short._build_line())
            ww = bar_seg(wide._build_line())
    check(
        "both bars produced a [####----] segment",
        ws > 0 and ww > 0,
        f"short={ws}, wide={ww}",
    )
    check(
        "bar widths equal across stacked bars",
        ws == ww,
        f"short bar={ws} chars, wide bar={ww} chars",
    )


# ---------------------------------------------------------------------------
# Visual demo (opt-in)
# ---------------------------------------------------------------------------

def visual_demo() -> None:
    """Mimics the harness's VAE+VSR interleave at human-watchable speed
    (~8 s). Per-step sleeps are intentionally slow enough that the bars
    visibly progress and the postfix updates can be read in flight.

    What to look for as it runs:
      * Layout: <indent><label> [<bar>] <n/total> <pct>% | RUN <d> | ETA <d> | <pace>
      * VAE bar's first tick (1/4) shows real numbers for RUN, ETA, and
        a "X.Xs/chunk" pace — NOT the "measuring" placeholder.  RUN
        climbs, ETA shrinks.
      * VSR bar shows "measuring" only at 0/80; after the first frame
        completes it switches to "RUN 0.1s | ETA 0.4s | NN.NN frame/s".
      * Durations: dashed clock — `--:--:SS` under a minute, `--:MM:SS`
        once minutes are non-zero, `HH:MM:SS` once hours appear. Always
        8 chars, with `--:` marking the "empty" fields.
      * Both bars' `[####----]` segments are exactly bar_width chars
        regardless of the rest of the line.
      * Columns (count, pct, RUN, ETA, pace) never jitter as digits grow.
    """
    n_chunks = 4
    frames_per_chunk = 20
    vae_decode_s = 1.00
    vsr_frame_s = 0.05
    est_total = n_chunks * (vae_decode_s + frames_per_chunk * vsr_frame_s)

    print()
    print(f"--- visual demo: {n_chunks} VAE chunks, "
          f"{n_chunks * frames_per_chunk} VSR frames, ~{est_total:.1f}s total ---")
    print("(per-step delays slowed so the bars visibly tick — real runs "
          "are faster per step but have many more steps)")
    print()

    with StackedPhaseBars() as bars:
        vae_bar = bars.add(
            total=n_chunks, desc="VAE chunks", unit="chunk", mininterval=0.1,
        )
        vsr_bar = bars.add(
            total=n_chunks * frames_per_chunk, desc="VSR frames", unit="frame",
            mininterval=0.1,
        )
        for _ in range(n_chunks):
            time.sleep(vae_decode_s)
            vae_bar.update(1)
            for _ in range(frames_per_chunk):
                time.sleep(vsr_frame_s)
                vsr_bar.update(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true",
        help="Also run the visual demo after the assertion battery.",
    )
    args = parser.parse_args()

    print("== PhaseBar assertion battery ==")
    t0 = time.perf_counter()
    test_first_tick_records_duration()
    test_eta_uses_post_update_count()
    test_pace_elapsed_count_are_math_consistent()
    test_line_columns_do_not_jitter()
    test_count_column_scales_with_total()
    test_bar_width_pinned()
    test_stacked_bars_share_count_column()
    test_stacked_bars_share_visual_width()
    test_fmt_duration_covers_all_scales()
    dt = time.perf_counter() - t0

    print(f"\n== summary == {len(_FAILURES)} failure(s), {dt:.2f}s")
    if _FAILURES:
        for f in _FAILURES:
            print(f"  FAIL: {f}")

    if args.demo and not _FAILURES:
        visual_demo()

    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
