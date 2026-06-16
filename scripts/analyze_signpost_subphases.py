#!/usr/bin/env python3
"""Analyze a signpost sidecar log produced by LTX_PROFILE_SIGNPOST_LOG.

Walks the begin/end event timeline and attributes each sub-phase
(attn_qkv, attn_sdpa, attn_out, v_ff_adaln, v_ff_inner) to its
currently-open parent phase (video_self_attn, video_text_ca,
audio_self_attn, audio_text_ca, a2v_cross, v2a_cross, video_ff,
audio_ff).  Produces:

1. Per-(parent, sub-phase) rollup with count/total/mean/p50/p90/p99/max.
2. Per-parent "unaccounted" budget = parent_wall - sum(sub-phases).
3. Top-N slowest individual sub-phase intervals across all parents.
4. Calls that fire OUTSIDE any parent (text encoder, AV connector) are
   bucketed under "[no_parent]".

Usage:
    python scripts/analyze_signpost_subphases.py <sidecar.log> [--top-n 20]

The sidecar.log is the file written when LTX_PROFILE_SIGNPOST_LOG is set;
each line is "<monotonic_ns> <begin|end> <phase>", with a comment header
line starting with "#".
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Phase taxonomy: which are parents (block-level) vs sub-phases (nested).
PARENT_PHASES = {
    "video_self_attn", "video_text_ca", "audio_self_attn", "audio_text_ca",
    "a2v_cross", "v2a_cross", "video_ff", "audio_ff",
}
SUB_PHASES = {
    "attn_qkv", "attn_sdpa", "attn_out",
    "v_ff_adaln", "v_ff_inner",
}
NO_PARENT = "[no_parent]"


def parse_log(path: Path):
    """Yield (ts_ns, event, phase) for each signpost line."""
    pat = re.compile(r"^(\d+)\s+(begin|end)\s+(\S+)\s*$")
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = pat.match(line)
            if m:
                yield int(m.group(1)), m.group(2), m.group(3)


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = int(round((p / 100.0) * (len(s) - 1)))
    return s[i]


def fmt_ms(ns):
    return f"{ns / 1e6:.2f}"


def fmt_s(ns):
    return f"{ns / 1e9:.3f}"


def analyze(events):
    # Per-(parent, sub) durations in ns.  Parent buckets also track their
    # own intervals for "total" and "unaccounted" calculations.
    bucket = defaultdict(list)  # (parent, sub) -> [dur_ns, ...]
    parent_total = defaultdict(list)  # parent -> [dur_ns of parent intervals]

    # Stack of currently-open phases.  Top entry is the innermost.
    # Each entry: (phase, ts_ns).  Sub-phase look-up uses
    # "closest open parent" walking down the stack.
    stack: list[tuple[str, int]] = []
    # All individual sub-phase calls, with parent attribution + duration.
    # Used for top-N slowest report.
    all_calls = []

    def find_open_parent():
        for phase, _ in reversed(stack):
            if phase in PARENT_PHASES:
                return phase
        return NO_PARENT

    for ts, ev, phase in events:
        if ev == "begin":
            stack.append((phase, ts))
        else:  # end
            # Pop the matching frame.  Signposts should be properly
            # nested, but defensively search if the top doesn't match.
            popped = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == phase:
                    popped = stack.pop(i)
                    break
            if popped is None:
                # Orphan end without begin - skip
                continue
            begin_ts = popped[1]
            dur = ts - begin_ts

            if phase in PARENT_PHASES:
                parent_total[phase].append(dur)
            elif phase in SUB_PHASES:
                parent = find_open_parent()
                bucket[(parent, phase)].append(dur)
                all_calls.append((dur, parent, phase))

    return bucket, parent_total, all_calls


def print_rollup(bucket, parent_total):
    """Per-parent table: each parent's wall + its sub-phase contributions
    + the unaccounted budget."""
    all_parents = sorted(parent_total.keys())
    # Also include any sub-phase buckets attributed to [no_parent].
    if any(p == NO_PARENT for p, _ in bucket.keys()):
        all_parents.append(NO_PARENT)

    print()
    print("=" * 86)
    print("PER-PARENT SUB-PHASE ATTRIBUTION")
    print("=" * 86)
    print(f"{'parent':<18} {'sub-phase':<14} {'n':>5} {'total_s':>9} "
          f"{'mean_ms':>9} {'p50_ms':>9} {'p99_ms':>9} {'max_ms':>9}")
    print("-" * 86)

    for parent in all_parents:
        parent_durs = parent_total.get(parent, [])
        parent_total_ns = sum(parent_durs)
        # Header row for the parent itself
        if parent_durs:
            n = len(parent_durs)
            print(
                f"{parent:<18} {'(total)':<14} {n:>5} "
                f"{fmt_s(parent_total_ns):>9} "
                f"{fmt_ms(parent_total_ns / n):>9} "
                f"{fmt_ms(pct(parent_durs, 50)):>9} "
                f"{fmt_ms(pct(parent_durs, 99)):>9} "
                f"{fmt_ms(max(parent_durs)):>9}"
            )
        else:
            print(f"{parent:<18} {'(total)':<14} {'-':>5} {'-':>9} "
                  f"{'-':>9} {'-':>9} {'-':>9} {'-':>9}")

        # Sub-phase rows for this parent
        sub_sum_ns = 0
        for sub in sorted(SUB_PHASES):
            durs = bucket.get((parent, sub), [])
            if not durs:
                continue
            n = len(durs)
            total_ns = sum(durs)
            sub_sum_ns += total_ns
            print(
                f"{'':<18} {sub:<14} {n:>5} "
                f"{fmt_s(total_ns):>9} "
                f"{fmt_ms(total_ns / n):>9} "
                f"{fmt_ms(pct(durs, 50)):>9} "
                f"{fmt_ms(pct(durs, 99)):>9} "
                f"{fmt_ms(max(durs)):>9}"
            )

        # Unaccounted budget for this parent
        if parent_durs and sub_sum_ns:
            unaccounted_ns = parent_total_ns - sub_sum_ns
            pct_unaccounted = 100 * unaccounted_ns / parent_total_ns
            print(
                f"{'':<18} {'[unaccounted]':<14} {'':>5} "
                f"{fmt_s(unaccounted_ns):>9} "
                f"{'':>9} {'':>9} {'':>9} "
                f"{pct_unaccounted:>8.1f}%"
            )
        print()
    print("=" * 86)


def print_top_n(all_calls, n):
    print()
    print(f"TOP-{n} SLOWEST INDIVIDUAL SUB-PHASE INTERVALS")
    print("=" * 60)
    print(f"{'ms':>9}  {'parent':<18} {'sub-phase':<14}")
    print("-" * 60)
    all_calls.sort(reverse=True)
    for dur, parent, sub in all_calls[:n]:
        print(f"{fmt_ms(dur):>9}  {parent:<18} {sub:<14}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="Sidecar log path (LTX_PROFILE_SIGNPOST_LOG)")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Show N slowest individual sub-phase intervals (default 20)")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"ERROR: log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    events = list(parse_log(args.log))
    if not events:
        print(f"ERROR: no signpost events found in {args.log}", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {args.log}")
    print(f"Events: {len(events)}")

    bucket, parent_total, all_calls = analyze(events)

    print_rollup(bucket, parent_total)
    print_top_n(all_calls, args.top_n)


if __name__ == "__main__":
    main()
