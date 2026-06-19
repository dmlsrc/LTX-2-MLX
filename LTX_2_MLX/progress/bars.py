"""Phase-accurate stacked progress bars for the VSR pipeline.

Single-line bars rendered directly to stderr via carriage-return +
ANSI cursor movement - no third-party progress-bar dependency. The layout matches
LTX_2_MLX/generate.py's `DenoiseProgress` shape, minus the STEP1 column
(meaningful for seconds-long denoise steps, useless for 50 ms VSR
frames that round to "0m 00s"):

    <indent><label> [<bar>] <n:>W>/<total:<W> <pct:5.1f>% \
        | RUN <duration> | ETA <duration> | <pace>

Bar on the LEFT (right after the label), numbers on the RIGHT,
separated by ` | `. Durations use `X.Xs` under a minute, `XmYYs`
above - sub-second resolution where it matters, compact above.

PhaseBar features beyond generate.py's DenoiseProgress:

* **Wall-clock origin** - each bar's clock starts at its construction
  time. RUN displays wall time from origin to the bar's last tick,
  so a phase that finishes earlier than its sibling shows a smaller
  RUN - the cross-bar finish-time delta is visible at a glance.
  Pace includes any pre-first-tick warmup (the bar was alive but no
  work had reported yet); that warmup amortizes over the first ~30
  ticks and pace converges to its steady-state value.
* **Math-consistent rate** - pace = `n / elapsed_from_origin`, so the
  displayed pace, RUN time and count multiply out cleanly: `pace x RUN
  == n_so_far`. ETA uses the same per-unit time, so `ETA x pace ~
  remaining`. Cross-phase outliers drag the average briefly then it
  recovers as more ticks roll in - no median sleight-of-hand.
* **Fixed-width columns** - count uses `max(3, digits(total))` per side,
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

__all__ = ["PhaseBar", "StackedPhaseBars"]


# ---------------------------------------------------------------------------
# Formatting helpers - matched 1:1 to LTX_2_MLX/generate.py
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """`HH:MM:SS` clock with empty leading-zero groups shown as `--:`
    so the "active" fields stand out:

        seconds < 60       ->  `--:--:SS`
        seconds < 3600     ->  `--:MM:SS`
        seconds >= 3600     ->  `HH:MM:SS`

    Always 8 chars up to 99:59:59; grows to 9 at 100h+ (rare). Round-
    half-up via `int(seconds + 0.5)` so 60.5 -> `--:01:01`, not `--:01:00`.
    """
    total_s = int(max(0.0, seconds) + 0.5)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    if m > 0:
        return f"--:{m:02d}:{s:02d}"
    return f"--:--:{s:02d}"


def _fmt_pace(seconds_per_step: float, unit: str, number_width: int = 6) -> str:
    """Slow -> ` X.X s/<unit>`, fast -> ` XX.X <unit>/s`.

    The number is right-justified in a `number_width`-char slot via
    `:>{number_width}.1f`.  The default 6 fits values up to 9999.9.
    When stacked with other bars, the caller passes the stack's
    shared pace number width so the right edges of the numbers line
    up across bars even when one bar's pace overflows the default
    slot (e.g. a 38339.1 chunk/s pace alongside a 79.7 frame/s pace).

    Per-bar total width is `3 + number_width + len(unit)` chars: the
    number, a 1-char separator, then either `s/<unit>` or `<unit>/s`
    - both 2 + len(unit) chars.
    """
    if seconds_per_step >= 1.0:
        return f"{seconds_per_step:>{number_width}.1f} s/{unit}"
    return f"{1.0 / seconds_per_step:>{number_width}.1f} {unit}/s"


def _natural_pace_number_width(seconds_per_step: float) -> int:
    """Width the pace number would consume without padding."""
    if seconds_per_step >= 1.0:
        return len(f"{seconds_per_step:.1f}")
    return len(f"{1.0 / seconds_per_step:.1f}")


# ---------------------------------------------------------------------------
# Stack-state shared between PhaseBars in the same stack.
# ---------------------------------------------------------------------------

class _StackState:
    """Shared state for bars in the same stack.

    * `total` - running count of bars; each PhaseBar uses it at render
      time to know how many rows up its own row sits.
    * `count_width` - the widest "n/total" slot across all registered
      bars. When a bar with a larger total registers, the slot grows
      and all previously-registered bars are force-redrawn so the count
      column (and therefore the `|` divider following it) stays aligned
      across the stack.
    * `bars` - references kept so we can iterate for cross-bar redraws
      when `count_width` grows.
    """

    __slots__ = ("bars", "count_width", "label_width", "pace_number_width", "total")

    def __init__(self) -> None:
        self.total: int = 0
        self.count_width: int = 7  # n_width=3 floor -> 2*3+1 = 7
        # Label slot width is grown to the longest registered bar's
        # `desc` so the `[` opening every bar's progress segment lines
        # up vertically across the stack ("VAE chunks [...]" and
        # "VT encode  [...]" instead of mis-aligned brackets).
        self.label_width: int = 0
        # Pace-number slot width - grown lazily during render when a
        # bar's natural pace formatting needs more digits than the
        # current slot.  Default 6 fits values up to 9999.9; bumps to
        # 7+ for high-throughput bars (38339.1 chunk/s, etc.).  When
        # the slot grows, every previously-rendered bar in the stack
        # is force-redrawn so all pace numbers right-align to the
        # same column across rows.
        self.pace_number_width: int = 6
        self.bars: list[PhaseBar] = []

    def register(self, bar: PhaseBar) -> int:
        """Add a bar, expand `count_width` / `label_width` if needed, and
        re-render any previously-registered bars so their columns match
        the new widths. Returns the bar's row index."""
        row = self.total
        self.total += 1
        natural_count = bar._natural_count_width()
        natural_label = len(bar._desc_label)
        widened = False
        if natural_count > self.count_width:
            self.count_width = natural_count
            widened = True
        if natural_label > self.label_width:
            self.label_width = natural_label
            widened = True
        self.bars.append(bar)
        if widened:
            # Re-render every bar registered BEFORE this one so their
            # count / label slots use the new (wider) widths. The new
            # bar's own initial render happens after register() returns.
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
        total: int | None,
        desc: str,
        unit: str = "it",
        bar_width: int = 28,
        mininterval: float = 1.0,
        indent: str = "  ",
        show_step1: bool = False,
        _stack: _StackState | None = None,
    ):
        self._total = total
        self._desc_label = desc
        self._unit = unit
        self._bar_width = bar_width
        self._mininterval = mininterval
        self._indent = indent
        self._show_step1 = show_step1

        # Timing state.
        self._t_origin: float = time.perf_counter()
        self._t_last: float | None = None  # None until first update()
        # Time of the first update(), captured once.  When `show_step1`
        # is enabled, the STEP1 column reports `_t_first_update - _t_origin`
        # - the wall-clock cost of the first unit of work.  Useful for
        # denoise-style bars where the first step is often warmup-heavy
        # (compile cache miss, first kernel dispatch, etc.) and worth
        # surfacing separately from the running pace.
        self._t_first_update: float | None = None
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
        """The count slot width *this* bar would need on its own -
        `2 * max(3, digits(total)) + 1`. The stack picks the max across
        all registered bars so the `|` after the count column aligns
        across the whole stack."""
        n_width = max(3, len(str(self._total))) if self._total else 4
        return 2 * n_width + 1

    # ---- public API -------------------------------------------------------

    def update(self, n: int = 1) -> None:
        """Advance by `n` units. Pace and ETA are derived from `self._n`
        divided by `self._t_last - self._t_origin` at render time."""
        now = time.perf_counter()
        self._t_last = now
        if self._t_first_update is None:
            # Captured once; STEP1 column reads (this - origin) when
            # show_step1=True is set on the bar.
            self._t_first_update = now
        self._n += n
        self._render(force=False)

    def set_n(self, n: int) -> None:
        """Set the current count to `n` (absolute value).

        Forwards to `update()` with the computed delta `n - self._n`,
        so timing/pace/STEP1 capture work identically.  Useful for
        callbacks that report progress as `(current_step, total_steps)`
        tuples rather than per-update deltas - e.g.,
        `LTX_2_MLX/generate.py`'s denoise stage callback.

        Backward steps (n < self._n) are clamped to a no-op rather
        than rewinding the count.  Backward progress would invalidate
        the running pace and STEP1 metrics; callers that legitimately
        need to reset a bar should construct a new one.
        """
        delta = n - self._n
        if delta > 0:
            self.update(delta)

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
            # Math-consistent triplet: pace x elapsed = work_done, and
            # eta x pace ~ remaining. The median-of-window approach was
            # arithmetically dishonest - it showed an "instantaneous"
            # pace that didn't multiply out to the frames already done.
            # Using `n / elapsed` for both pace and ETA makes the three
            # numbers add up: at the end of a run, `pace x RUN == count`.
            sec_per_unit = elapsed / self._n
            run_str = f"{_fmt_duration(elapsed):>8}"
            if total:
                eta_str = f"{_fmt_duration(max(0, total - n) * sec_per_unit):>8}"
            else:
                eta_str = "      --"
            # Grow the stack's shared pace-number slot if this bar's
            # natural pace formatting needs more chars than the
            # current slot.  Force-redraw the OTHER bars (not self -
            # we're about to render below) so their pace numbers
            # right-align in the new wider column.
            natural_pw = _natural_pace_number_width(sec_per_unit)
            if natural_pw > self._stack.pace_number_width:
                self._stack.pace_number_width = natural_pw
                for b in self._stack.bars:
                    if b is not self:
                        b._render(force=True)
            pace_str = _fmt_pace(
                sec_per_unit, self._unit,
                number_width=self._stack.pace_number_width,
            )
        else:
            run_str = "      --"
            eta_str = "      --"
            pace_str = "measuring"

        # Pad the label to the stack's max so the `[` opening lines up
        # across bars (e.g. "VAE chunks" (10) and "VT encode " (9) both
        # render in a 10-char slot when stacked together).  Standalone
        # bars use their own natural width.
        label_width = max(self._stack.label_width, len(self._desc_label))
        padded_label = f"{self._desc_label:<{label_width}}"

        # Optional STEP1 column: wall-clock cost of the first update
        # (origin -> first update timestamp).  Matches the DenoiseProgress
        # surface in LTX_2_MLX/generate.py so a PhaseBar can drop-in
        # replace it for denoise-style runs where the first step is
        # typically warmup-heavy.  Same 8-char `_fmt_duration` format
        # as RUN / ETA.
        if self._show_step1:
            if self._t_first_update is not None:
                step1_dur = max(0.0, self._t_first_update - self._t_origin)
                step1_str = f" | STEP1 {_fmt_duration(step1_dur):>8}"
            else:
                step1_str = " | STEP1       --"
        else:
            step1_str = ""

        return (
            f"{self._indent}{padded_label} [{bar}] "
            f"{count_str} | {pct_str}{step1_str} "
            f"| RUN {run_str} | ETA {eta_str} | {pace_str}"
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

    def write(self, message: str, *, position: str = "above") -> None:
        """Print `message` without corrupting the bars.

        `position` controls where the message lands relative to the bars
        in the captured / persistent terminal scrollback:

        - `"above"` (default, scroll-message-above-bars semantics):
            Erase the live bars, print the message at the position they
            used to occupy, then redraw the bars one (or more) lines
            further down.  Live bars stay anchored at the bottom; the
            message scrolls up with prior terminal content.  Use this
            for streaming-style logs that should sit alongside the bars
            (e.g. the encoder's captured setup output during streaming).

        - `"below"`:
            Treat the existing bars as frozen at their current rows
            (static screen content) and emit the message on the parked
            row beneath them.  The bar stack is RESET - subsequent
            `bars.add()` calls land BELOW the message, in a fresh stack
            that inherits column widths from the old one (so alignment
            spans the message gap).  Use this for permanent log entries
            that should be visually sandwiched between completed bars
            and new ones to come (e.g. inter-stage messages in the
            distilled two-stage denoise loop).

            CONSTRAINT: callers must not call `.update()` on the
            now-frozen bars after a `position="below"` write - their
            row indexes no longer match the stack's cursor accounting.
            Reaching their `total` and stopping is the expected usage.

        Multi-line messages are fine - each embedded newline advances
        the bars (above mode) or pushes the parked cursor (below mode)
        by one more line.

        No-op pass-through when no bars are active yet (writes to
        stderr as-is) so callers can use this unconditionally during
        setup.
        """
        if not self._bars:
            sys.stderr.write(message)
            if not message.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()
            return

        if position == "below":
            # Force a final render on every bar so the frozen state
            # reflects the latest counts, not whatever was on screen at
            # the previous mininterval-rate-limited render.  Without
            # this, fast-ticking bars often park at e.g. 239/241 because
            # the last two update() calls came in under the mininterval
            # threshold and never triggered a re-render of their own.
            for bar in self._bars:
                bar._render(force=True)
            # Emit at the parked cursor row.  Carriage return first
            # because the most-recent render left the cursor at column
            # `len(line)`, not column 0.
            sys.stderr.write("\r" + message)
            if not message.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()
            # Freeze the existing bars: they stay on screen as static
            # content, and a fresh stack starts here for future
            # `bars.add()`.  Preserve column widths so any new bars
            # render aligned to the same columns that the old bars
            # established.
            self._bars = []
            self._stack.bars = []
            self._stack.total = 0
            return

        if position != "above":
            raise ValueError(
                f"StackedPhaseBars.write: position must be 'above' or "
                f"'below', got {position!r}"
            )

        n_bars = self._stack.total
        # Up to the topmost bar's row, then erase from there to end of screen.
        # \033[<n>A = move cursor up n lines.  \033[J = erase from cursor to
        # end of screen.  Together: wipe all bar rows and leave cursor at the
        # top of the freshly-vacated region.
        sys.stderr.write(f"\033[{n_bars}A\r\033[J")
        # Write the message, ensuring it terminates with a newline so the
        # cursor lands on the line below.
        sys.stderr.write(message)
        if not message.endswith("\n"):
            sys.stderr.write("\n")
        # Re-emit the bars inline (each line followed by \n), so the cursor
        # naturally advances row-by-row and the final \n parks it one row
        # below the bottom bar - restoring the render-invariant column.
        #
        # Earlier this method reserved rows via `\n * n_bars` and then
        # re-rendered each bar with the cursor-up-then-content-then-cursor-
        # down dance.  In a live terminal that's equivalent, but file
        # captures that don't fully interpret every ANSI escape end up
        # showing the blank reserved row AND the bar's content as separate
        # lines (the bar appears twice).  Direct inline writes here avoid
        # the dance entirely.
        for i, bar in enumerate(self._bars):
            line = bar._build_line()
            pad = " " * max(0, bar._last_line_len - len(line))
            sys.stderr.write(line + pad)
            bar._last_line_len = len(line)
            bar._last_render_t = time.perf_counter()
            if i < n_bars - 1:
                sys.stderr.write("\n")
        # One final \n so the cursor parks at col 0 of the row immediately
        # below the bottom bar - same invariant render() relies on.
        sys.stderr.write("\n")
        sys.stderr.flush()

    def close(self) -> None:
        """Close all bars and leave the cursor at column 0 of the parked row.

        During the run the logical cursor sits one row below the bottom
        bar (the row reserved by the final `\\n` printed when the last
        bar was added), but at whatever column the most recent render
        left it - `_render` uses `\\033[<n>B` to return the cursor to
        the parked row after writing, which preserves the *column*
        position, leaving the cursor at column `len(line+pad)` rather
        than column 0.

        Without a carriage return here, the next caller-side `print()`
        would append to the parked row at that column, producing output
        like "<bar line>                <padding>done: 11.1 MiB ...".
        We emit `\\r` to move the cursor to column 0 of the parked row;
        the caller's next `print()` then writes on a clean line tightly
        below the bar.

        Idempotent.
        """
        if not self._bars:
            return
        for bar in self._bars:
            bar.close()
        self._bars.clear()
        sys.stderr.write("\r")
        sys.stderr.flush()

    def __enter__(self) -> StackedPhaseBars:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
