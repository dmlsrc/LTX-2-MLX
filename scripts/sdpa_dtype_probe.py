#!/usr/bin/env python3
"""Probe what dtypes mx.fast.scaled_dot_product_attention is actually called with
during the LTX-2-MLX denoise loop, and optionally measure per-call GPU time.

Runs by:
1. Monkey-patching mx.fast.scaled_dot_product_attention BEFORE any LTX_2_MLX
   imports happen, so it sees every call.
2. Logging (dtype, shape) signatures with caller info.
3. Optionally measuring per-call GPU duration with mx.eval barriers
   (LTX_PROBE_TIME_SDPA=1).
4. Invoking the existing generate.py with the remaining args.
5. Printing a histogram of (dtype, shape) signatures + timing summary at exit.

Usage (from the LTX-2-MLX repo root):
    python scripts/sdpa_dtype_probe.py <args you'd pass to scripts/generate.py>

Environment variables:
    LTX_PROBE_LIMIT=N       Cap unique signatures logged (default 50).
    LTX_PROBE_TIME_SDPA=1   Force mx.eval after every SDPA call and record
                            wall-time-per-call. Slows the run substantially
                            (breaks lazy graph at every SDPA). Use with
                            LTX_PROFILE_STOP_AFTER_STEPS=2 for a quick probe.
                            Requires LTX_DISABLE_COMPILED_ATTN=1 and
                            LTX_DISABLE_COMPILED_HELPERS=1 so mx.eval can
                            break MLX's compiled regions.
    LTX_PROBE_TIME_LOG=path Dump per-call timing to a JSONL file
                            (one record per SDPA call). Implies LTX_PROBE_TIME_SDPA=1.
    LTX_PROBE_SCRIPT=path   Override the default target script
                            (default: scripts/generate.py, resolved relative
                            to cwd).
    LTX_PROBE_MODULE=dotted Run a dotted module path instead of a script file
                            (e.g. mlx_video.models.ltx_2.generate for A/B
                            probes against mlx-video). Uses runpy.run_module
                            so package-relative imports work. Cwd is added
                            to sys.path automatically.

Recommended invocation for the video_self_attn outlier hunt (assumes you've
sourced docs/TEST_PROMPTS.md to get $BAKERY).  Log output goes to
$SHARED_TEMP_DIR/trace_analysis/ when set (preferred — shared between
accounts, survives reboots), otherwise falls back to ${TMPDIR:-/tmp}.

    LTX_PROBE_TIME_SDPA=1 \\
    LTX_PROBE_TIME_LOG="${SHARED_TEMP_DIR:-${TMPDIR:-/tmp}}/trace_analysis/sdpa_per_call.jsonl" \\
    LTX_PROFILE_STOP_AFTER_STEPS=2 \\
    LTX_DISABLE_COMPILED_ATTN=1 LTX_DISABLE_COMPILED_HELPERS=1 \\
    python scripts/sdpa_dtype_probe.py "$BAKERY" \\
      --pipeline one-stage --height 288 --width 512 --duration 20 --seed 124 \\
      --generate-audio --fast-mode --output-prefix probe_postfix
"""

from __future__ import annotations
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict

# Patch BEFORE any LTX_2_MLX import.
import mlx.core as mx

_orig_sdpa = mx.fast.scaled_dot_product_attention
_call_count = 0
_sig_counter: Counter = Counter()
_first_traces: dict = {}
_PROBE_LIMIT = int(os.environ.get("LTX_PROBE_LIMIT", "50"))

_TIME_LOG_PATH = os.environ.get("LTX_PROBE_TIME_LOG")
_TIME_SDPA = bool(os.environ.get("LTX_PROBE_TIME_SDPA")) or bool(_TIME_LOG_PATH)
_time_log_fh = None
_per_sig_ms: dict = defaultdict(list)  # sig -> [ms, ms, ...]

if _TIME_LOG_PATH:
    os.makedirs(os.path.dirname(_TIME_LOG_PATH) or ".", exist_ok=True)
    _time_log_fh = open(_TIME_LOG_PATH, "w")


def _sig_for(q, k, v, mask):
    mask_dtype = mask.dtype if mask is not None else None
    mask_shape = mask.shape if mask is not None else None
    return (
        ("q", q.dtype, q.shape),
        ("k", k.dtype, k.shape),
        ("v", v.dtype, v.shape),
        ("mask", mask_dtype, mask_shape),
    )


def _sig_key(sig):
    """Compact string key for a signature, suitable for JSON output."""
    q_dt, q_sh = sig[0][1], sig[0][2]
    k_dt, k_sh = sig[1][1], sig[1][2]
    v_dt, v_sh = sig[2][1], sig[2][2]
    m_dt, m_sh = sig[3][1], sig[3][2]
    return (
        f"q={q_dt}{tuple(q_sh)} k={k_dt}{tuple(k_sh)} "
        f"v={v_dt}{tuple(v_sh)} mask={m_dt}{tuple(m_sh) if m_sh else None}"
    )


def _patched_sdpa(q, k, v, scale=None, mask=None, **kwargs):
    global _call_count
    _call_count += 1
    sig = _sig_for(q, k, v, mask)
    _sig_counter[sig] += 1
    # Capture call site for the first occurrence of each signature
    if sig not in _first_traces and len(_first_traces) < _PROBE_LIMIT:
        stack = traceback.extract_stack(limit=8)
        callers = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                   for f in stack[:-1] if "sdpa_dtype_probe" not in f.filename]
        _first_traces[sig] = callers[-4:]

    if _TIME_SDPA:
        # mx.eval inputs first so the timer doesn't pick up upstream work.
        # If we're called from inside a compiled context, mx.eval raises
        # ValueError — fall back to untimed dispatch and continue.
        try:
            mx.eval(q, k, v)
            if mask is not None:
                mx.eval(mask)
            t0 = time.monotonic_ns()
            out = _orig_sdpa(q, k, v, scale=scale, mask=mask, **kwargs)
            mx.eval(out)
            ms = (time.monotonic_ns() - t0) / 1e6
            _per_sig_ms[sig].append(ms)
        except ValueError as e:
            if "eval" in str(e) and "compile" in str(e):
                # Inside a compiled function — can't break the graph.
                # Record the call signature but skip timing.
                return _orig_sdpa(q, k, v, scale=scale, mask=mask, **kwargs)
            raise
        if _time_log_fh is not None:
            rec = {
                "call_idx": _call_count,
                "ms": ms,
                "sig": _sig_key(sig),
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "mask_shape": list(mask.shape) if mask is not None else None,
                "dtype": str(q.dtype),
                "scale": scale,
            }
            _time_log_fh.write(json.dumps(rec) + "\n")
            _time_log_fh.flush()
        return out
    return _orig_sdpa(q, k, v, scale=scale, mask=mask, **kwargs)


mx.fast.scaled_dot_product_attention = _patched_sdpa

print("[probe] monkey-patched mx.fast.scaled_dot_product_attention", flush=True)
print(f"[probe] will log up to {_PROBE_LIMIT} unique signatures", flush=True)
if _TIME_SDPA:
    print("[probe] PER-CALL TIMING ENABLED (mx.eval barriers per SDPA call)", flush=True)
    if _TIME_LOG_PATH:
        print(f"[probe] writing per-call JSONL to {_TIME_LOG_PATH}", flush=True)


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = int(round((p / 100.0) * (len(s) - 1)))
    return s[i]


def _dump_report():
    print("\n" + "=" * 78, flush=True)
    print(f"[probe] TOTAL SDPA CALLS: {_call_count}", flush=True)
    print(f"[probe] UNIQUE SIGNATURES: {len(_sig_counter)}", flush=True)
    print("=" * 78, flush=True)
    for i, (sig, n) in enumerate(_sig_counter.most_common()):
        q_dt = sig[0][1]; q_sh = sig[0][2]
        k_dt = sig[1][1]; k_sh = sig[1][2]
        v_dt = sig[2][1]; v_sh = sig[2][2]
        m_dt = sig[3][1]; m_sh = sig[3][2]
        print(f"\n[{i:2d}] {n:5d} calls", flush=True)
        print(f"    q: {str(q_dt):12s} {q_sh}", flush=True)
        print(f"    k: {str(k_dt):12s} {k_sh}", flush=True)
        print(f"    v: {str(v_dt):12s} {v_sh}", flush=True)
        if m_dt is not None:
            print(f"    mask: {str(m_dt):9s} {m_sh}", flush=True)
        else:
            print("    mask: None", flush=True)
        trace = _first_traces.get(sig)
        if trace:
            print("    first caller stack:", flush=True)
            for f in trace:
                print(f"      {f}", flush=True)
        if _TIME_SDPA and sig in _per_sig_ms:
            times = _per_sig_ms[sig]
            print(
                f"    timing: n={len(times):4d} "
                f"sum={sum(times):8.1f}ms "
                f"mean={sum(times)/len(times):7.2f}ms "
                f"p50={_pct(times,50):7.2f}ms "
                f"p90={_pct(times,90):7.2f}ms "
                f"p99={_pct(times,99):7.2f}ms "
                f"max={max(times):7.2f}ms",
                flush=True,
            )
    print("=" * 78, flush=True)

    if _TIME_SDPA:
        # Sorted-by-wall-time rollup
        print("\n" + "=" * 78, flush=True)
        print("[probe] PER-SIGNATURE WALL TIME (sorted by total ms desc)", flush=True)
        print("=" * 78, flush=True)
        rows = []
        for sig, times in _per_sig_ms.items():
            rows.append((sum(times), len(times), max(times),
                         _pct(times, 99), _pct(times, 90), _pct(times, 50),
                         _sig_key(sig)))
        rows.sort(reverse=True)
        print(f"{'total_ms':>10s} {'count':>6s} {'max_ms':>8s} {'p99':>8s} {'p90':>8s} {'p50':>8s}  signature", flush=True)
        for total, n, mx_, p99, p90, p50, key in rows:
            print(f"{total:10.1f} {n:6d} {mx_:8.2f} {p99:8.2f} {p90:8.2f} {p50:8.2f}  {key}", flush=True)
        print("=" * 78, flush=True)

        # Top-N slowest individual calls across all signatures
        all_calls = []
        for sig, times in _per_sig_ms.items():
            for t in times:
                all_calls.append((t, _sig_key(sig)))
        all_calls.sort(reverse=True)
        print("\n[probe] TOP-30 SLOWEST INDIVIDUAL CALLS", flush=True)
        print(f"{'ms':>8s}  signature", flush=True)
        for t, key in all_calls[:30]:
            print(f"{t:8.2f}  {key}", flush=True)
        print("=" * 78, flush=True)

    if _time_log_fh is not None:
        _time_log_fh.close()


import atexit
atexit.register(_dump_report)


# Now invoke the target generate.py with the remaining args.
# Default to LTX-2-MLX's scripts/generate.py via run_path; set
# LTX_PROBE_MODULE to a dotted module path (e.g.
# mlx_video.models.ltx_2.generate) for projects that use
# package-relative imports — those need run_module so the package
# context is set up. LTX_PROBE_SCRIPT still works for standalone files.
import runpy

# When invoked as `python <path>/sdpa_dtype_probe.py ...`, sys.path[0]
# is the script's directory, not cwd. That means a sibling source tree
# (mlx-video, an unrelated mlx_* package, etc.) won't import unless we
# add cwd to sys.path explicitly. Required for `LTX_PROBE_MODULE=...`
# to find a target module installed only at cwd.
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)
    print(f"[probe] added cwd to sys.path: {_cwd}", flush=True)

_target_module = os.environ.get("LTX_PROBE_MODULE")
_target_script = os.environ.get("LTX_PROBE_SCRIPT", "scripts/generate.py")

if _target_module:
    sys.argv = [_target_module] + sys.argv[1:]
    print(f"[probe] running module: {_target_module} {' '.join(sys.argv[1:])}", flush=True)
    runpy.run_module(_target_module, run_name="__main__", alter_sys=True)
else:
    sys.argv = [_target_script] + sys.argv[1:]
    print(f"[probe] running script: {_target_script} {' '.join(sys.argv[1:])}", flush=True)
    runpy.run_path(_target_script, run_name="__main__")
