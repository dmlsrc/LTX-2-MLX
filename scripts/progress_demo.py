#!/usr/bin/env python3
"""Visual progress-bar demo CLI for LTX_2_MLX.progress.

Renders one or more live-bar visual scenarios so the appearance can be
eyeballed by hand (alignment holds, pace columns widen on demand,
bars.write() lands messages above bars without duplication, STEP1 column
matches scripts/generate.py's DenoiseProgress layout).

This is a manual-inspection tool, NOT a test.  The automated assertions
for PhaseBar / StackedPhaseBars live in tests/test_progress.py.

Usage:
    scripts/progress_demo.py --demo list           # list scenarios
    scripts/progress_demo.py --demo all            # run every scenario
    scripts/progress_demo.py --demo <name>         # run one scenario
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly from the repo root without an install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.progress import (
    StackedPhaseBars,
)

# ---------------------------------------------------------------------------
# Visual demo
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
      * Cursor parks at column 0 of the row below the bottom bar - no
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
        - STEP1 reports the FIRST step's duration, not the running pace.
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
        the same stack - no overlap, both bars share the layout.
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

        # Inter-stage pipeline messages (same content `av_pipeline.py`
        # emits via progress_message between stage 1 and stage 2).
        # position="below" leaves Stage 1's bar frozen at its row and
        # routes the messages BELOW it - Stage 2's bar then slots in
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
    - without it the terminal would be saturated by render calls.  Pace
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
    print("  scripts/progress_demo.py --demo <name>   # run one scenario")
    print("  scripts/progress_demo.py --demo all      # run every scenario")
    print("  scripts/progress_demo.py --demo list     # show this catalog")


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
            "Run a visual scenario.  Pass a scenario name to run one "
            "(see `--demo list`), `all` to run every scenario "
            "back-to-back, or `list` (the default when --demo has no "
            "argument) to print the catalog and exit."
        ),
    )
    args = parser.parse_args()

    if args.demo is None or args.demo == "list":
        list_scenarios()
    elif args.demo == "all":
        for _, fn in SCENARIOS.items():
            fn()
    elif args.demo in SCENARIOS:
        SCENARIOS[args.demo]()
    else:
        print(
            f"\nUnknown scenario {args.demo!r}.  "
            f"Available: {', '.join(SCENARIOS)}, all, list."
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
