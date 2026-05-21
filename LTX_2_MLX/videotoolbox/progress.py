"""Phase-accurate stacked progress bars for the VSR pipeline.

Single-line bars rendered directly to stderr via carriage-return +
ANSI cursor movement — no tqdm dependency. The layout matches
scripts/generate.py's `DenoiseProgress` shape, minus the STEP1 column
(meaningful for seconds-long denoise steps, useless for 50 ms VSR
frames that round to "0m 00s"):

    <indent><label> [<bar>] <n:>W>/<total:<W> <pct:5.1f>% \
        | RUN <duration> | ETA <duration> | <pace>

Bar on the LEFT (right after the label), numbers on the RIGHT,
separated by ` | `. Durations use `X.Xs` under a minute, `XmYYs`
above — sub-second resolution where it matters, compact above.

PhaseBar features beyond generate.py's DenoiseProgress:

* **Wall-clock origin** — each bar's clock starts at its construction
  time. RUN displays wall time from origin to the bar's last tick,
  so a phase that finishes earlier than its sibling shows a smaller
  RUN — the cross-bar finish-time delta is visible at a glance.
  Pace includes any pre-first-tick warmup (the bar was alive but no
  work had reported yet); that warmup amortizes over the first ~30
  ticks and pace converges to its steady-state value.
* **Math-consistent rate** — pace = `n / elapsed_from_origin`, so the
  displayed pace, RUN time and count multiply out cleanly: `pace × RUN
  == n_so_far`. ETA uses the same per-unit time, so `ETA × pace ≈
  remaining`. Cross-phase outliers drag the average briefly then it
  recovers as more ticks roll in — no median sleight-of-hand.
* **Fixed-width columns** — count uses `max(3, digits(total))` per side,
  pct is `5.1f%`, STEP1/RUN/ETA use `format_progress_duration`'s
  `"Xm YYs"` form, pace is `"<f.1>s/<unit>"` or `"<f.2><unit>/s"`. None
  of the columns shift width as numbers grow.

Stacking via `StackedPhaseBars`: each new bar reserves its row by
emitting `\\n` at construction. Renders move the cursor up to the
bar's row via `\\033[<n>A`, rewrite the line in place, then move
back down via `\\033[<n>B`. The logical cursor sits one line below
the last bar in the stack throughout the run.
"""

from __future__ import annotations

import sys
import time
from typing import Optional


__all__ = ["PhaseBar", "StackedPhaseBars"]


# ---------------------------------------------------------------------------
# Formatting helpers — matched 1:1 to scripts/generate.py
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """`HH:MM:SS` clock with empty leading-zero groups shown as `--:`
    so the "active" fields stand out:

        seconds < 60       →  `--:--:SS`
        seconds < 3600     →  `--:MM:SS`
        seconds ≥ 3600     →  `HH:MM:SS`

    Always 8 chars up to 99:59:59; grows to 9 at 100h+ (rare). Round-
    half-up via `int(seconds + 0.5)` so 60.5 → `--:01:01`, not `--:01:00`.
    """
    total_s = int(max(0.0, seconds) + 0.5)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    if m > 0:
        return f"--:{m:02d}:{s:02d}"
    return f"--:--:{s:02d}"


def _fmt_pace(seconds_per_step: float, unit: str) -> str:
    """Slow → ` X.X s/<unit>`, fast → ` XX.X <unit>/s`.

    Both forms are exactly `9 + len(unit)` chars wide: a 6-char number
    via `:>6.1f`, a 1-char separator space, then either `s/<unit>` or
    `<unit>/s` — both 2 + len(unit) chars. Per-bar pace width is
    constant; across bars with different unit names the column width
    differs by the difference in unit-name length, but that's stable
    once the bars are constructed (unit is fixed per bar).
    """
    if seconds_per_step >= 1.0:
        return f"{seconds_per_step:>6.1f} s/{unit}"
    return f"{1.0 / seconds_per_step:>6.1f} {unit}/s"


# ---------------------------------------------------------------------------
# Stack-state shared between PhaseBars in the same stack.
# ---------------------------------------------------------------------------

class _StackState:
    """Shared state for bars in the same stack.

    * `total` — running count of bars; each PhaseBar uses it at render
      time to know how many rows up its own row sits.
    * `count_width` — the widest "n/total" slot across all registered
      bars. When a bar with a larger total registers, the slot grows
      and all previously-registered bars are force-redrawn so the count
      column (and therefore the `|` divider following it) stays aligned
      across the stack.
    * `bars` — references kept so we can iterate for cross-bar redraws
      when `count_width` grows.
    """

    __slots__ = ("total", "count_width", "bars")

    def __init__(self) -> None:
        self.total: int = 0
        self.count_width: int = 7  # n_width=3 floor → 2*3+1 = 7
        self.bars: list["PhaseBar"] = []

    def register(self, bar: "PhaseBar") -> int:
        """Add a bar, expand `count_width` if needed, and re-render any
        previously-registered bars so their count column matches the
        new width. Returns the bar's row index."""
        row = self.total
        self.total += 1
        natural = bar._natural_count_width()
        widened = natural > self.count_width
        if widened:
            self.count_width = natural
        self.bars.append(bar)
        if widened:
            # Re-render every bar registered BEFORE this one so their
            # count slot uses the new (wider) width. The new bar's own
            # initial render happens after register() returns.
            for b in self.bars[:-1]:
                b._render(force=True)
        return row


# ---------------------------------------------------------------------------
# PhaseBar
# ---------------------------------------------------------------------------

class PhaseBar:
    """One progress row rendered directly to stderr.

    Construction reserves the row by emitting `\\n`. Each render moves
    the cursor up to the bar's row (`\\033[<n>A`), prints `\\r<line>`,
    then moves back down (`\\033[<n>B`).
    """

    def __init__(
        self,
        *,
        total: Optional[int],
        desc: str,
        unit: str = "it",
        bar_width: int = 28,
        mininterval: float = 1.0,
        indent: str = "  ",
        _stack: Optional[_StackState] = None,
    ):
        self._total = total
        self._desc_label = desc
        self._unit = unit
        self._bar_width = bar_width
        self._mininterval = mininterval
        self._indent = indent

        # Timing state.
        self._t_origin: float = time.perf_counter()
        self._t_last: Optional[float] = None  # None until first update()
        self._n: int = 0

        # Render bookkeeping.
        self._last_render_t: float = 0.0
        self._last_line_len: int = 0
        self._closed: bool = False

        # Register with the stack (or create a private one for solo use).
        # Reserve our terminal row BEFORE registering so the cursor is
        # already at the bottom of the stack when register() iterates
        # older bars for a count-column-widening redraw.
        sys.stderr.write("\n")
        sys.stderr.flush()
        self._stack = _stack if _stack is not None else _StackState()
        self._row = self._stack.register(self)
        # Initial paint so the user sees the column structure right away
        # instead of a blank line until the first update() arrives.
        self._render(force=True)

    def _natural_count_width(self) -> int:
        """The count slot width *this* bar would need on its own —
        `2 * max(3, digits(total)) + 1`. The stack picks the max across
        all registered bars so the `|` after the count column aligns
        across the whole stack."""
        n_width = max(3, len(str(self._total))) if self._total else 4
        return 2 * n_width + 1

    # ---- public API -------------------------------------------------------

    def update(self, n: int = 1) -> None:
        """Advance by `n` units. Pace and ETA are derived from `self._n`
        divided by `self._t_last - self._t_origin` at render time."""
        self._t_last = time.perf_counter()
        self._n += n
        self._render(force=False)

    def close(self) -> None:
        """Force a final render so the displayed numbers reflect the last
        tick (not whatever was on screen at the previous mininterval
        boundary). Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._render(force=True)

    # ---- rendering --------------------------------------------------------

    def _build_line(self) -> str:
        """The full info line, ready to be printed after `\\r`."""
        n = self._n
        total = self._total
        # Count: right-align the whole "n/total" token in the stack's
        # shared count_width slot. Stack-shared means a bar with total=4
        # and a bar with total=9999 in the same stack both render their
        # counts in a 9-char slot (the max of their natural widths), so
        # the `|` divider after the count lines up across rows.
        count_width = self._stack.count_width
        if total is not None:
            count_str = f"{f'{n}/{total}':>{count_width}}"
            pct = (n / total * 100.0) if total > 0 else 0.0
        else:
            # Width-stable no-total fallback so layout doesn't shift if
            # total becomes known mid-run.
            count_str = f"{n!s:>{count_width}}"
            pct = 0.0
        pct_str = f"{pct:5.1f}%"

        # Bar: "{'#'*filled}{'-'*(width-filled)}" wrapped in [].
        if total and total > 0:
            filled = min(self._bar_width, int(round(pct / 100.0 * self._bar_width)))
        else:
            filled = 0
        bar = "#" * filled + "-" * (self._bar_width - filled)

        # Time / pace columns. Durations get padded to 6 chars right-justified
        # at the use site so the section width stays constant whether
        # the value is `--:--:07` or `24:00:00`.
        elapsed = (self._t_last - self._t_origin) if self._t_last is not None else 0.0
        if self._n > 0 and elapsed > 0:
            # Math-consistent triplet: pace × elapsed = work_done, and
            # eta × pace ≈ remaining. The median-of-window approach was
            # arithmetically dishonest — it showed an "instantaneous"
            # pace that didn't multiply out to the frames already done.
            # Using `n / elapsed` for both pace and ETA makes the three
            # numbers add up: at the end of a run, `pace × RUN == count`.
            sec_per_unit = elapsed / self._n
            run_str = f"{_fmt_duration(elapsed):>8}"
            if total:
                eta_str = f"{_fmt_duration(max(0, total - n) * sec_per_unit):>8}"
            else:
                eta_str = "      --"
            pace_str = _fmt_pace(sec_per_unit, self._unit)
        else:
            run_str = "      --"
            eta_str = "      --"
            pace_str = "warming up"

        return (
            f"{self._indent}{self._desc_label} [{bar}] "
            f"{count_str} | {pct_str} | RUN {run_str} | ETA {eta_str} | {pace_str}"
        )

    def _render(self, *, force: bool) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_render_t) < self._mininterval:
            return
        line = self._build_line()
        # Pad-erase trailing chars when the new line is shorter than the
        # previous one. Without this, leftovers from a longer previous
        # render stay visible on the row.
        pad = " " * max(0, self._last_line_len - len(line))
        # Cursor lives one line below the bottom of the stack throughout
        # the run, so to redraw our row we go up (stack.total - self._row)
        # lines, print, then come back down the same amount.
        below = self._stack.total - self._row
        up = f"\033[{below}A" if below > 0 else ""
        down = f"\033[{below}B" if below > 0 else ""
        sys.stderr.write(up + "\r" + line + pad + down)
        sys.stderr.flush()
        self._last_render_t = now
        self._last_line_len = len(line)


# ---------------------------------------------------------------------------
# StackedPhaseBars
# ---------------------------------------------------------------------------

class StackedPhaseBars:
    """Context manager for a small fixed stack of PhaseBars.

    Bars added in call order claim rows 0, 1, 2, ... (top to bottom).
    On exit (or explicit `close()`) every bar is finalized and a single
    trailing newline lands the cursor below the stack.

    Usage::

        with StackedPhaseBars() as bars:
            vae = bars.add(total=N, desc="VAE chunks", unit="chunk")
            vsr = bars.add(total=M, desc="VSR frames", unit="frame")
            for chunk in chunks:
                vae.update(1)
                vsr.start()        # idempotent post-first-tick
                for frame in chunk:
                    ...
                    vsr.update(1)
    """

    def __init__(self) -> None:
        self._stack = _StackState()
        self._bars: list[PhaseBar] = []

    def add(self, **kwargs) -> PhaseBar:
        bar = PhaseBar(_stack=self._stack, **kwargs)
        self._bars.append(bar)
        return bar

    def close(self) -> None:
        """Close all bars and drop the cursor below the stack. Idempotent."""
        if not self._bars:
            return
        for bar in self._bars:
            bar.close()
        self._bars.clear()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self) -> "StackedPhaseBars":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
