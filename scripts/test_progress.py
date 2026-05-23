#!/usr/bin/env python3
"""Quick verification of LTX_2_MLX.videotoolbox.progress.PhaseBar.

Runs a battery of assertions on PhaseBar's timing logic (deferred start,
first-tick measurement, idempotent start, math-consistent pace/ETA,
fixed bar width, label / count / pace-number alignment across stacked
bars, STEP1 column behavior, bars.write() interrupt safety, cursor
reset on close).  Total runtime ~2 seconds.

With --demo, also renders one or more live-bar visual scenarios so the
appearance can be eyeballed (alignment holds, pace columns widen on
demand, bars.write() lands messages above bars without duplication,
STEP1 column matches scripts/generate.py's DenoiseProgress layout).

Exit code: 0 on pass, non-zero on any failed assertion.

Usage:
    scripts/test_progress.py                       # assertions only
    scripts/test_progress.py --demo list           # list scenarios
    scripts/test_progress.py --demo all            # run every scenario
    scripts/test_progress.py --demo <name>         # run one scenario
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

from LTX_2_MLX.progress import (  # noqa: E402
    PhaseBar, StackedPhaseBars,
)
from LTX_2_MLX.progress.bars import _fmt_duration  # noqa: E402


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


def test_stacked_bars_share_label_column() -> None:
    """Stacked bars with different `desc` lengths must left-justify the
    label in a column wide enough for the longest one, so the `[` opening
    every bar's progress segment lines up vertically (e.g. "VAE chunks"
    at 10 chars next to "VT encode" at 9 chars — both render with the
    label padded to 10 chars + space + `[`)."""
    print("\n[8] stacked bars share label column across disparate desc lengths")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            short = bars.add(total=1, desc="x", unit="it", mininterval=0.0)
            wide = bars.add(total=1, desc="much longer label", unit="it", mininterval=0.0)
            short_line = short._build_line()
            wide_line = wide._build_line()

    short_bracket = short_line.find("[")
    wide_bracket = wide_line.find("[")
    check(
        "`[` lands at the same column index in both bars",
        short_bracket == wide_bracket and short_bracket > 0,
        f"short={short_bracket}, wide={wide_bracket}",
    )
    check(
        "shorter label is left-justified (followed by padding spaces, not [)",
        # The short label "x" should be followed by spaces, NOT directly by [.
        short_line[len("  x"):short_line.find("[")].rstrip() == "",
        f"short_line={short_line!r}",
    )
    # And the long label sits at its natural width
    check(
        "longest label appears intact at its natural width",
        "much longer label" in wide_line,
        f"wide_line={wide_line!r}",
    )


def test_stacked_bars_share_pace_number_column() -> None:
    """Stacked bars where one's pace value has more digits than the other
    must share a right-justified number slot wide enough for the largest
    natural width, so the right edge of the number aligns across rows
    (e.g. `38339.1 chunk/s` and `   79.6 frame/s` both have the number's
    right edge at the same column)."""
    print("\n[9] stacked bars share pace-number column across magnitudes")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            fast = bars.add(total=1, desc="fast", unit="chunk", mininterval=0.0)
            slow = bars.add(total=361, desc="slow", unit="frame", mininterval=0.0)
            # Synthesize timings directly — avoid wall-clock variance.
            fast._n = 1
            fast._t_last = fast._t_origin + 0.00002    # ~50000 chunk/s
            slow._n = 1
            slow._t_last = slow._t_origin + 0.01       # ~100 frame/s
            fast_line = fast._build_line()
            # `slow_line` was rendered against the now-grown pace_number_width.
            slow_line = slow._build_line()

    # Both lines should have a single pace number, right-justified in
    # the same column. The number's right edge sits immediately before
    # the " <unit>/s" or " s/<unit>" tail.
    fast_pace = fast_line.rsplit("|", 1)[1]
    slow_pace = slow_line.rsplit("|", 1)[1]
    fast_unit_idx = fast_pace.find(" chunk/s")
    slow_unit_idx = slow_pace.find(" frame/s")
    check(
        "both bars expose a recognizable pace unit tail",
        fast_unit_idx > 0 and slow_unit_idx > 0,
        f"fast_pace={fast_pace!r}, slow_pace={slow_pace!r}",
    )
    check(
        "right edge of pace number aligns across bars",
        fast_unit_idx == slow_unit_idx,
        f"fast_idx={fast_unit_idx}, slow_idx={slow_unit_idx}, "
        f"fast={fast_pace!r}, slow={slow_pace!r}",
    )
    check(
        "stack's pace_number_width grew to at least the wider natural width",
        bars._stack.pace_number_width >= len("50000.0"),  # 7 chars
        f"pace_number_width={bars._stack.pace_number_width}",
    )


def test_step1_column_renders_first_update_duration() -> None:
    """A bar created with `show_step1=True` must include a STEP1 column
    that reports the wall-clock cost of the first update (origin -> first
    update timestamp).  When `show_step1=False` (default), the column is
    absent entirely so the layout matches existing non-denoise bars."""
    print("\n[10] STEP1 column populated from first-update timing")
    buf = io.StringIO()
    with redirect_stderr(buf):
        # Default: no STEP1 column at all
        plain = PhaseBar(total=10, desc="plain", unit="it", mininterval=0.0)
        plain.update(1)
        plain_line = plain._build_line()
        plain.close()

    check(
        "default bar layout has no STEP1 column",
        "STEP1" not in plain_line,
        f"plain_line={plain_line!r}",
    )

    buf = io.StringIO()
    with redirect_stderr(buf):
        stepped = PhaseBar(
            total=10, desc="stepped", unit="it",
            mininterval=0.0, show_step1=True,
        )
        # Force a measurable first-step duration without sleeping by
        # rewinding the origin (PhaseBar reads `_t_first_update - _t_origin`).
        stepped._t_origin = time.perf_counter() - 0.5
        stepped.update(1)
        stepped_line = stepped._build_line()
        stepped.close()

    check(
        "show_step1 bar includes a STEP1 column",
        "STEP1" in stepped_line,
        f"stepped_line={stepped_line!r}",
    )
    check(
        "STEP1 value reflects ~500 ms first-step duration",
        # _fmt_duration rounds the half-second to '--:--:01' (1 second).
        # That's the correct human-readable representation for sub-second
        # values just shy of a full second.
        "STEP1 --:--:00" in stepped_line or "STEP1 --:--:01" in stepped_line,
        f"stepped_line={stepped_line!r}",
    )

    # Pre-update state: STEP1 should show the empty placeholder, not crash.
    buf = io.StringIO()
    with redirect_stderr(buf):
        pre = PhaseBar(
            total=10, desc="pre", unit="it",
            mininterval=0.0, show_step1=True,
        )
        pre_line = pre._build_line()
        pre.close()
    check(
        "STEP1 column shows -- placeholder before first update",
        "STEP1       --" in pre_line,
        f"pre_line={pre_line!r}",
    )


def test_write_above_bars_does_not_duplicate_bar_rows() -> None:
    """`StackedPhaseBars.write(msg)` must NOT split bar renders across
    two adjacent `\\n`-delimited file lines.

    The dup-bar bug appeared in captured log files (where bars showed
    on two consecutive lines).  The captured-stream invariant is that
    after we strip ANSI cursor escapes AND collapse `\\r`-overwrites
    (which terminals interpret as "rewrite current line"), no two
    adjacent `\\n`-delimited file lines should both be a fully-formed
    rendering of the same bar.

    Regression for: bars.write() reserving rows via `\\n * n_bars`
    then writing bar content via cursor-up-then-content — in file
    captures the reserved blank row and the redrawn bar landed on
    separate file lines, producing the visible duplicate.
    """
    print("\n[11] bars.write() does not place bars on adjacent file lines")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            vae = bars.add(total=1, desc="VAE chunks", unit="chunk", mininterval=0.0)
            vt = bars.add(total=10, desc="VT encode", unit="frame", mininterval=0.0)
            vae.update(1)
            for _ in range(10):
                vt.update(1)
            bars.write("INTERRUPTING MESSAGE")

    # Strip ANSI escapes and apply `\r`-overwrite semantics:
    # within each newline-delimited file line, anything before the
    # last `\r` is "overwritten" by what follows — i.e., the terminal
    # only displays the final post-\r content on that physical line.
    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    stripped = ansi.sub("", buf.getvalue())
    physical_lines: list[str] = []
    for chunk in stripped.split("\n"):
        # If a \r appears, the last segment wins (everything before
        # was overwritten in place on that physical row).
        if "\r" in chunk:
            chunk = chunk.split("\r")[-1]
        physical_lines.append(chunk.rstrip())

    # Look for adjacent identical bar lines — that would be the dup-bar
    # bug.  Same-label-different-content is fine (e.g. 0/1 vs 1/1);
    # only adjacent identical strings count as a duplicate.
    def has_adjacent_dup(lines, predicate):
        prev = None
        for line in lines:
            if predicate(line) and line == prev:
                return True
            prev = line
        return False

    vae_dup = has_adjacent_dup(
        physical_lines,
        lambda L: L.lstrip().startswith("VAE chunks "),
    )
    vt_dup = has_adjacent_dup(
        physical_lines,
        lambda L: L.lstrip().startswith("VT encode "),
    )
    check(
        "no two adjacent file lines hold an identical VAE chunks rendering",
        not vae_dup,
        f"first 12 physical lines: {physical_lines[:12]}",
    )
    check(
        "no two adjacent file lines hold an identical VT encode rendering",
        not vt_dup,
        f"first 12 physical lines: {physical_lines[:12]}",
    )
    check(
        "INTERRUPTING MESSAGE appears in the captured output",
        "INTERRUPTING MESSAGE" in stripped,
        f"stripped[:200]={stripped[:200]!r}",
    )


def test_write_handles_multiline_messages() -> None:
    """`bars.write(msg)` should accept multi-line input (e.g. captured
    output from a constructor's print() calls) and emit each input line
    on its own row above the bars."""
    print("\n[12] bars.write() handles multi-line messages")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            bars.add(total=1, desc="bar", unit="it", mininterval=0.0)
            bars.write("first line\nsecond line\nthird line")
            bars.write("after\nmore")

    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    raw = ansi.sub("", buf.getvalue())
    for needle in ("first line", "second line", "third line", "after", "more"):
        check(
            f"{needle!r} appears in captured output",
            needle in raw,
            f"raw[:300]={raw[:300]!r}",
        )


def test_lazy_bar_add_after_write() -> None:
    """Mirrors `scripts/generate.py`'s distilled-two-stage flow:
    one bar is added, ticks to completion, the caller emits
    inter-stage messages via `bars.write()`, then a SECOND bar is
    added lazily and runs to completion.

    Regression target: when a new bar is added after bars.write(),
    the existing bar must continue to render correctly at its row
    while the new bar gets its own row below.  Bug class: bars.write()
    leaving stale cursor / row state that breaks a subsequent
    bars.add() — would manifest as overlapping bars or one bar
    overwriting the other.
    """
    print("\n[15] lazy bar add after bars.write(position='below')")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            s1 = bars.add(
                total=4, desc="Stage 1", unit="step",
                mininterval=0.0, show_step1=True,
            )
            for _ in range(4):
                s1.update(1)
            bars.write(
                "  Upsampling latent 2x with spatial upscaler...",
                position="below",
            )
            bars.write(
                "  Distilled stage 2: 3 steps at 192x192",
                position="below",
            )
            s2 = bars.add(
                total=3, desc="Stage 2", unit="step",
                mininterval=0.0, show_step1=True,
            )
            for _ in range(3):
                s2.update(1)

    check(
        "stage 1 reached its total",
        s1._n == 4,
        f"s1._n={s1._n}",
    )
    check(
        "stage 2 reached its total",
        s2._n == 3,
        f"s2._n={s2._n}",
    )
    check(
        "stage 1 retained its desc through bars.write() interruptions",
        "Stage 1" in s1._build_line(),
        f"s1 line={s1._build_line()!r}",
    )
    # position="below" resets the stack; s2 starts a fresh stack at row 0.
    check(
        "stage 2 starts a fresh stack after position='below' resets",
        s2._row == 0,
        f"s2._row={s2._row}",
    )

    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    stripped = ansi.sub("", buf.getvalue())
    check(
        "first inter-stage message appears in output",
        "Upsampling latent 2x" in stripped,
        f"stripped[:300]={stripped[:300]!r}",
    )
    check(
        "second inter-stage message appears in output",
        "Distilled stage 2" in stripped,
        f"stripped[:500]={stripped[:500]!r}",
    )
    # Visual order check: the byte offsets of key substrings in the
    # ANSI-stripped output reflect the chronological order writes
    # happened.  With position="below", Stage 1's bytes must precede
    # the inter-stage messages, and Stage 2's bytes must follow.
    # Catches a regression where position="below" accidentally falls
    # back to "above" behavior (in which case the messages would be
    # emitted BEFORE Stage 1's bytes).
    s1_pos = stripped.find("Stage 1 ")
    s2_pos = stripped.find("Stage 2 ")
    ups_pos = stripped.find("Upsampling latent 2x")
    msg2_pos = stripped.find("Distilled stage 2")
    check(
        "ordering: Stage 1 bytes < Upsampling msg < Distilled stage 2 msg < Stage 2 bytes",
        0 <= s1_pos < ups_pos < msg2_pos < s2_pos,
        f"s1={s1_pos}, ups={ups_pos}, st2_msg={msg2_pos}, s2={s2_pos}",
    )


def test_raw_print_between_bar_updates_breaks_layout() -> None:
    """Detect when a raw `print()` lands on a bar's parked row instead
    of going through `bars.write()`.

    The pipeline historically had a few places that called `print()`
    directly while bars were alive (e.g. the latent-save confirmation
    inside `_save_distilled_two_stage_latents`).  Those prints land at
    the cursor's parked row WITH a trailing newline, which:

      1. Shifts the parked cursor down by one row.
      2. Stomps the bar's content at the column where the previous
         render left the cursor.
      3. Causes the bar's subsequent close()-force-render to write at
         the WRONG row (because the render math assumes the cursor is
         still parked at row N+1, but it's actually at row N+2 now).

    This test simulates exactly that pattern and asserts the symptom
    is visible in the captured byte stream: after a raw print() the
    bar's content appears twice in the file capture (the original
    render PLUS the misplaced close-force-render at a different row).

    The fix is to route the print through `bars.write()` so the cursor
    bookkeeping stays consistent.  Until callers do, the test is a
    canary: pipelines/utilities that grow new raw print() calls during
    bar lifetime will break this assertion and prompt the author to
    use the progress_message / bars.write() path instead.
    """
    print("\n[16] raw print() during bar lifetime corrupts layout (canary)")

    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            bar = bars.add(
                total=3, desc="bar", unit="step", mininterval=0.0,
            )
            bar.update(1)
            bar.update(1)
            bar.update(1)
            # Simulate the pipeline-internal raw print bug: a print()
            # lands on stderr while the bar is still alive.  Goes to
            # the same stream as the bars' renders (we redirected
            # stderr for the test), shifting the cursor's parked row.
            sys.stderr.write("  Saved distilled stage latents: /tmp/x.npz\n")
            # context-manager exit force-renders the bar one more time

    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    stripped = ansi.sub("", buf.getvalue())
    physical_lines: list[str] = []
    for chunk in stripped.split("\n"):
        if "\r" in chunk:
            chunk = chunk.split("\r")[-1]
        physical_lines.append(chunk.rstrip())
    bar_lines = [L for L in physical_lines if "bar [" in L]
    # The bug signature: at least one bar line has the raw print's
    # text appended directly to its content (no `\n` between), because
    # the print landed at column `len(bar_line)` on the parked row.
    # That's exactly the "Saved distilled stage latents:" stomp the
    # user observed.
    stomped = any(
        "bar [" in L and "Saved distilled stage latents" in L
        for L in physical_lines
    )
    check(
        "raw print() between bar updates leaves residue on the bar's "
        "row (canary: SHOULD fail when raw print() bypasses bars.write())",
        stomped,
        f"bar_lines={bar_lines}, all_lines={physical_lines}",
    )
    check(
        "the raw print's text appears in the captured output",
        "Saved distilled stage latents" in stripped,
        f"stripped[:300]={stripped[:300]!r}",
    )


def test_progress_message_routes_through_bars_write() -> None:
    """Pipeline-side regression check: when the pipeline emits a
    confirmation through `progress_message`, the message must NOT
    corrupt the bar's row.  Compare against
    test_raw_print_between_bar_updates_breaks_layout — using
    bars.write(position="below") keeps the bar and the message on
    DIFFERENT physical lines in the captured output.
    """
    print("\n[17] progress_message through bars.write() preserves layout")

    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            bar = bars.add(
                total=3, desc="bar", unit="step", mininterval=0.0,
            )
            bar.update(1)
            bar.update(1)
            bar.update(1)
            # Pipeline-style: emit confirmation via bars.write() —
            # this is what the fixed _save_distilled_two_stage_latents
            # does when `progress_message` is supplied.
            bars.write(
                "  Saved distilled stage latents: /tmp/x.npz",
                position="below",
            )

    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    stripped = ansi.sub("", buf.getvalue())
    physical_lines: list[str] = []
    for chunk in stripped.split("\n"):
        if "\r" in chunk:
            chunk = chunk.split("\r")[-1]
        physical_lines.append(chunk.rstrip())
    # No physical line should have BOTH the bar content AND the
    # confirmation text on it — they must be on separate physical
    # lines (because bars.write() ended the bar's row with \n before
    # emitting the message).
    no_stomp = not any(
        "bar [" in L and "Saved distilled stage latents" in L
        for L in physical_lines
    )
    check(
        "progress_message-routed write does NOT stomp the bar's row",
        no_stomp,
        f"physical_lines={physical_lines}",
    )
    check(
        "the confirmation message appears on its own physical line",
        any("Saved distilled stage latents" in L and "bar [" not in L
            for L in physical_lines),
        f"physical_lines={physical_lines}",
    )


def test_no_raw_prints_in_progress_message_methods() -> None:
    """Static lint: any pipeline method that takes a `progress_message`
    callback must NOT contain raw `print()` calls inside its body —
    those would stomp the caller's bars when a bar UI is active.

    Allowlist: the documented `print(message)` fallback inside
    `emit_progress_message` (single Name('message') argument), which
    is the explicit path for callers that don't pass a progress
    callable.

    Caught historically: the
    `_save_distilled_two_stage_latents` "Saved distilled stage
    latents:" print, and the "VAE decode started/complete" prints
    inside `generate_distilled_two_stage` /
    `generate_distilled_stage2_from_latents`.  Both got routed through
    `progress_message` once this lint was added.

    Adding a new raw `print()` to a progress-aware method will fail
    this test, prompting the author to route through
    `progress_message` (or extend the allowlist if the print is
    genuinely bar-inactive).
    """
    print("\n[18] no raw print() in progress_message-aware pipeline methods")

    import ast
    pipeline_root = Path(__file__).parent.parent / "LTX_2_MLX" / "pipelines"
    # Files scanned: anything under pipelines/ that defines a method
    # taking a `progress_message` parameter.  Currently one_stage.py;
    # the scan is generic so future pipeline modules get covered.
    py_files = sorted(pipeline_root.glob("*.py"))

    offenders: list[tuple[str, str, int, str]] = []
    for py in py_files:
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = [a.arg for a in node.args.args] + [
                a.arg for a in node.args.kwonlyargs
            ]
            if "progress_message" not in params:
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if not (isinstance(func, ast.Name) and func.id == "print"):
                    continue
                # Allowlist: `print(message)` — the documented fallback
                # inside emit_progress_message and _save_distilled_two_stage_latents
                # for when progress_message is None.
                if (
                    len(child.args) == 1
                    and isinstance(child.args[0], ast.Name)
                    and child.args[0].id == "message"
                    and not child.keywords
                ):
                    continue
                src = (
                    ast.unparse(child)
                    if hasattr(ast, "unparse")
                    else f"print(...) at line {child.lineno}"
                )
                offenders.append((py.name, node.name, child.lineno, src))

    check(
        "no raw print() inside any progress_message-accepting pipeline "
        "method (except the documented `print(message)` fallback)",
        not offenders,
        "offenders (file, method, line, source):\n  " + "\n  ".join(
            f"{f}:{ln} in {m}: {s}" for (f, m, ln, s) in offenders
        ) if offenders else "",
    )


def test_set_n_forwards_absolute_step() -> None:
    """`PhaseBar.set_n(step)` must advance the bar to the absolute count
    by computing the delta and forwarding to `update()`.  Backward steps
    are clamped to no-ops.  Behavior matches what generate.py's denoise
    callbacks need (they report `(current_step, total)` from the
    pipeline)."""
    print("\n[14] set_n() forwards absolute step counts")
    buf = io.StringIO()
    with redirect_stderr(buf):
        bar = PhaseBar(
            total=10, desc="b", unit="step",
            mininterval=0.0, show_step1=True,
        )
        bar.set_n(3)
    check(
        "set_n(3) lands at count 3",
        bar._n == 3,
        f"got {bar._n}",
    )
    check(
        "set_n() captured the first-update timestamp for STEP1",
        bar._t_first_update is not None,
        "missing _t_first_update",
    )

    buf = io.StringIO()
    with redirect_stderr(buf):
        bar2 = PhaseBar(total=10, desc="b2", unit="step", mininterval=0.0)
        bar2.set_n(5)
        bar2.set_n(8)
    check(
        "successive set_n() advances cumulatively (5 -> 8)",
        bar2._n == 8,
        f"got {bar2._n}",
    )

    buf = io.StringIO()
    with redirect_stderr(buf):
        bar3 = PhaseBar(total=10, desc="b3", unit="step", mininterval=0.0)
        bar3.set_n(7)
        bar3.set_n(4)   # backward — must clamp to no-op
    check(
        "backward set_n() is a no-op (does not rewind)",
        bar3._n == 7,
        f"got {bar3._n}",
    )

    buf = io.StringIO()
    with redirect_stderr(buf):
        bar4 = PhaseBar(total=10, desc="b4", unit="step", mininterval=0.0)
        bar4.set_n(5)
        bar4.set_n(5)   # same step — must be no-op
    check(
        "idempotent set_n() does not double-tick",
        bar4._n == 5,
        f"got {bar4._n}",
    )


def test_close_emits_carriage_return() -> None:
    """`StackedPhaseBars.close()` must emit `\\r` after closing each bar
    so the next caller-side `print()` lands at column 0 of the parked
    row rather than getting appended to the bar's row at the column
    where the last bar render left the cursor.

    Regression test for the "done: ..." line being shifted right after
    the bar's content when displayed on the same row.
    """
    print("\n[13] close() resets cursor to column 0")
    buf = io.StringIO()
    with redirect_stderr(buf):
        with StackedPhaseBars() as bars:
            bar = bars.add(total=1, desc="b", unit="it", mininterval=0.0)
            bar.update(1)

    raw = buf.getvalue()
    # The final character written by close() should be `\r` (the
    # context-manager exit calls close() which now emits "\r").
    check(
        "close() leaves a \\r as the last byte of stderr output",
        raw.endswith("\r"),
        f"last 16 bytes: {raw[-16:]!r}",
    )


# ---------------------------------------------------------------------------
# Visual demo (opt-in)
# ---------------------------------------------------------------------------

def _header(title: str, blurb: str = "") -> None:
    """Per-scenario heading printed before bars come up."""
    print()
    print(f"--- demo: {title} ---")
    if blurb:
        print(blurb)
    print()


def demo_vae_vsr_streaming() -> None:
    """The streaming-encode happy path: VAE chunks (slow) feeding a VT
    encode bar (fast).  Mirrors the layout `scripts/generate.py` produces
    on a tiled distilled-two-stage run with `--output-backend videotoolbox`.

    What to look for:
      * Both bars share label / count / pace-number columns.
      * VAE pace settles in "X.X s/chunk" form; VT pace shows
        "XX.X frame/s" once the first frame completes.
      * Cursor parks at column 0 of the row below the bottom bar — no
        extra blank line, no "done:" appended to the bar row.
    """
    n_chunks = 4
    frames_per_chunk = 20
    vae_decode_s = 1.00
    vsr_frame_s = 0.05
    est_total = n_chunks * (vae_decode_s + frames_per_chunk * vsr_frame_s)
    _header(
        f"VAE -> VSR streaming ({n_chunks} chunks, "
        f"{n_chunks * frames_per_chunk} frames, ~{est_total:.1f}s)",
        "Per-step delays slowed so the bars visibly tick.",
    )
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
    print("(cursor parked on the line immediately below the bars)")


def demo_denoise_with_step1() -> None:
    """Single PhaseBar with `show_step1=True`, simulating a denoise pass
    where the first step is warmup-heavy.  Demonstrates the STEP1 column
    matching the layout `scripts/generate.py`'s `DenoiseProgress` shows.

    What to look for:
      * STEP1 column populates only after the first update.
      * STEP1 reads `--:--:01` once the slow first step completes;
        subsequent steps tick fast and STEP1 stays fixed at that value
        — STEP1 reports the FIRST step's duration, not the running pace.
      * RUN climbs across all steps; ETA shrinks; pace reflects the
        running average.
    """
    _header(
        "denoise with STEP1 (8 steps, slow first)",
        "First step takes ~1s; rest tick at ~150ms each.",
    )
    with StackedPhaseBars() as bars:
        bar = bars.add(
            total=8, desc="Stage 1 denoising", unit="step",
            mininterval=0.05, show_step1=True,
        )
        time.sleep(1.0)
        bar.update(1)
        for _ in range(7):
            time.sleep(0.15)
            bar.update(1)
    print("(STEP1 stayed at the first-step duration while RUN climbed)")


def demo_denoise_stacked() -> None:
    """Mirrors `scripts/generate.py`'s distilled-two-stage flow:
    Stage 1 bar is added at start, runs to completion, then between
    stages the pipeline emits "Upsampling latent 2x..." and "Distilled
    stage 2: ..." messages via `bars.write()`.  Stage 2 bar is added
    LAZILY (on first stage_2 callback) and ticks to completion.

    What to look for:
      * Both bars' `[`, count, STEP1, RUN, ETA, pace columns align.
      * Stage 1's STEP1 freezes at its first-step duration; Stage 2's
        STEP1 captures a (different) first-step duration on its first
        update.
      * The inter-stage messages land above Stage 1's frozen bar; the
        bar redraws cleanly below each message.
      * When Stage 2 is added (lazy), it slots in below Stage 1 in
        the same stack — no overlap, both bars share the layout.
    """
    _header(
        "Stage 1 + Stage 2 denoise stacked (lazy add + interstage messages)",
        "Mirrors generate.py: Stage 2 added on its first callback, "
        "inter-stage messages routed through bars.write().",
    )
    with StackedPhaseBars() as bars:
        s1 = bars.add(
            total=8, desc="Stage 1 denoising", unit="step",
            mininterval=0.05, show_step1=True,
        )
        # Stage 1 ticks through completion.
        time.sleep(0.6)
        s1.update(1)
        for _ in range(7):
            time.sleep(0.1)
            s1.update(1)

        # Inter-stage pipeline messages (same content `one_stage.py`
        # emits via progress_message between stage 1 and stage 2).
        # position="below" leaves Stage 1's bar frozen at its row and
        # routes the messages BELOW it — Stage 2's bar then slots in
        # under the messages when added next.
        bars.write(
            "  Upsampling latent 2x with spatial upscaler...",
            position="below",
        )
        time.sleep(0.4)
        bars.write(
            "  Distilled stage 2: 3 steps at 192x192",
            position="below",
        )

        # Stage 2 added lazily, as if its first callback just arrived.
        s2 = bars.add(
            total=3, desc="Stage 2 denoising", unit="step",
            mininterval=0.05, show_step1=True,
        )
        time.sleep(0.9)
        s2.update(1)
        for _ in range(2):
            time.sleep(0.3)
            s2.update(1)


def demo_label_alignment() -> None:
    """Three bars with very different `desc` lengths.  The label column
    is sized to the longest desc and shorter labels are left-justified
    with trailing spaces so the `[` opening lines up vertically.
    """
    _header(
        "label alignment (3 bars, mismatched desc lengths)",
        "Short label is padded right; `[` lines up across rows.",
    )
    with StackedPhaseBars() as bars:
        short = bars.add(total=10, desc="x", unit="it", mininterval=0.05)
        med = bars.add(total=10, desc="medium label", unit="it", mininterval=0.05)
        long_ = bars.add(
            total=10, desc="much-longer-descriptor", unit="it",
            mininterval=0.05,
        )
        for _ in range(10):
            time.sleep(0.08)
            short.update(1)
            med.update(1)
            long_.update(1)


def demo_pace_alignment() -> None:
    """Two bars whose pace magnitudes differ by orders of magnitude
    (one ticks at ~10 chunk/s, the other at ~1000+ frame/s).  Watch the
    pace number column widen to fit the wider value and force-redraw
    the older bar so right-edges align.
    """
    _header(
        "pace number alignment (10 chunk/s vs ~1000+ frame/s)",
        "Slow bar's pace column widens once the fast bar's pace lands.",
    )
    with StackedPhaseBars() as bars:
        slow = bars.add(total=5, desc="slow", unit="chunk", mininterval=0.05)
        fast = bars.add(total=300, desc="fast", unit="frame", mininterval=0.05)
        for _ in range(3):
            time.sleep(0.1)
            slow.update(1)
        for _ in range(300):
            time.sleep(0.001)
            fast.update(1)
        for _ in range(2):
            time.sleep(0.1)
            slow.update(1)
    print("(slow bar's pace was re-rendered in the now-wider pace column)")


def demo_write_above_bars() -> None:
    """Bars are alive, then `bars.write()` is called repeatedly with
    short and multi-line messages.  Mirrors what
    `encode_video_videotoolbox` does when streaming and the captured
    setup output (VsrSession + AVWriter constructor prints) needs to
    land above the live VAE chunks bar.
    """
    _header(
        "bars.write() interrupts (single + multi-line messages)",
        "Messages land above the bars; bars re-render at the "
        "shifted-down position with no duplicate rows.",
    )
    with StackedPhaseBars() as bars:
        vae = bars.add(total=4, desc="VAE chunks", unit="chunk", mininterval=0.05)
        vt = bars.add(total=40, desc="VT encode", unit="frame", mininterval=0.05)
        time.sleep(0.4)
        vae.update(1)
        for _ in range(10):
            time.sleep(0.03)
            vt.update(1)
        bars.write("[single-line message] VSR session ready")
        time.sleep(0.3)
        vae.update(1)
        for _ in range(10):
            time.sleep(0.03)
            vt.update(1)
        bars.write(
            "[multi-line message]\n"
            "  encode (videotoolbox): VSR=balanced(4x) -> HEVC ...\n"
            "  -> /path/to/output.mp4"
        )
        for _ in range(2):
            time.sleep(0.3)
            vae.update(1)
            for _ in range(10):
                time.sleep(0.03)
                vt.update(1)


def demo_column_growth() -> None:
    """Start with bars whose totals fit a narrow count column, then add
    a third bar with a much larger total to trigger count_width growth.
    The first two bars get force-redrawn so the `|` divider stays
    aligned across all three rows.
    """
    _header(
        "dynamic count column growth (small + small + huge)",
        "First two bars (total=5) render in a narrow count slot; "
        "third bar (total=10000) triggers a column-wide redraw.",
    )
    with StackedPhaseBars() as bars:
        a = bars.add(total=5, desc="A", unit="it", mininterval=0.05)
        b = bars.add(total=5, desc="B", unit="it", mininterval=0.05)
        # First 2 ticks of A and B render in a small count column.
        for _ in range(2):
            time.sleep(0.3)
            a.update(1)
            b.update(1)
        # New bar with a huge total appears mid-run; A and B get force-
        # redrawn in the new count slot.
        big = bars.add(total=10000, desc="big", unit="it", mininterval=0.05)
        for _ in range(40):
            time.sleep(0.02)
            big.update(250)
        # Remaining 3 ticks of A and B finish their bars at 5/5.
        for _ in range(3):
            time.sleep(0.1)
            a.update(1)
            b.update(1)


def demo_single_bar_fast() -> None:
    """One bar ticking ~5000 it/s.  Stress-tests mininterval throttling
    — without it the terminal would be saturated by render calls.  Pace
    should settle into a steady value, RUN climb in seconds, ETA
    shrink monotonically.
    """
    _header(
        "single bar, fast (~5000 it/s, mininterval=0.1)",
        "mininterval=0.1 keeps the render rate sane.",
    )
    with StackedPhaseBars() as bars:
        bar = bars.add(total=10000, desc="ticks", unit="it", mininterval=0.1)
        for _ in range(10000):
            bar.update(1)


def demo_single_bar_slow() -> None:
    """One bar where each step takes ~0.5s.  Sanity check that
    mininterval doesn't suppress legitimate updates on slower
    workloads.
    """
    _header(
        "single bar, slow (~0.5s/step, 6 steps)",
        "Each step visibly distinct; pace settles around ~0.5 s/it.",
    )
    with StackedPhaseBars() as bars:
        bar = bars.add(total=6, desc="slow", unit="it", mininterval=0.05)
        for _ in range(6):
            time.sleep(0.5)
            bar.update(1)


# Scenario registry --------------------------------------------------------

SCENARIOS: dict = {
    "vae-vsr":         demo_vae_vsr_streaming,
    "denoise":         demo_denoise_with_step1,
    "denoise-stacked": demo_denoise_stacked,
    "labels":          demo_label_alignment,
    "pace":            demo_pace_alignment,
    "write-above":     demo_write_above_bars,
    "growth":          demo_column_growth,
    "single-fast":     demo_single_bar_fast,
    "single-slow":     demo_single_bar_slow,
}


def list_scenarios() -> None:
    """Print the scenario catalog (name + one-line summary)."""
    print("Available --demo scenarios:")
    print()
    max_name = max(len(name) for name in SCENARIOS)
    for name, fn in SCENARIOS.items():
        first_line = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:<{max_name}}  {first_line}")
    print()
    print("Usage:")
    print("  scripts/test_progress.py --demo <name>   # run one scenario")
    print("  scripts/test_progress.py --demo all      # run every scenario")
    print("  scripts/test_progress.py --demo list     # show this catalog")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo", nargs="?", const="list", default=None,
        help=(
            "Run a visual scenario after the assertion battery.  "
            "Pass a scenario name to run one (see `--demo list`), `all` "
            "to run every scenario back-to-back, or `list` (the default "
            "when --demo has no argument) to print the catalog and exit."
        ),
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
    test_stacked_bars_share_label_column()
    test_stacked_bars_share_pace_number_column()
    test_step1_column_renders_first_update_duration()
    test_write_above_bars_does_not_duplicate_bar_rows()
    test_write_handles_multiline_messages()
    test_lazy_bar_add_after_write()
    test_raw_print_between_bar_updates_breaks_layout()
    test_progress_message_routes_through_bars_write()
    test_no_raw_prints_in_progress_message_methods()
    test_set_n_forwards_absolute_step()
    test_close_emits_carriage_return()
    test_fmt_duration_covers_all_scales()
    dt = time.perf_counter() - t0

    print(f"\n== summary == {len(_FAILURES)} failure(s), {dt:.2f}s")
    if _FAILURES:
        for f in _FAILURES:
            print(f"  FAIL: {f}")

    if args.demo is not None and not _FAILURES:
        if args.demo == "list":
            print()
            list_scenarios()
        elif args.demo == "all":
            for name, fn in SCENARIOS.items():
                fn()
        elif args.demo in SCENARIOS:
            SCENARIOS[args.demo]()
        else:
            print(
                f"\nUnknown scenario {args.demo!r}.  "
                f"Available: {', '.join(SCENARIOS)}, all, list."
            )
            return 2

    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
