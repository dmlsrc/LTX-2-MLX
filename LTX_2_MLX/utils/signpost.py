"""os_signpost wrapper for Instruments / Metal System Trace.

Gated by env var ``LTX_PROFILE_SIGNPOSTS=1``.  Zero cost when unset:
the ``signpost(...)`` context manager returns a no-op object that has
no ctypes call, no string formatting, no dylib load.

Why this exists
---------------
The ``os_signpost`` C API is macro-based and requires the calling
image's ``__dso_handle`` as an implicit parameter, so it cannot be
invoked directly from Python ctypes.  We ship a tiny C shim
(``_signpost.c``) that exposes one ``begin_<phase>`` / ``end_<phase>``
pair per profiled phase.  The dylib auto-rebuilds if the source is
newer than the binary.

Usage
-----
::

    from LTX_2_MLX.utils.signpost import signpost

    with signpost("video_self_attn"):
        out = self.attn1(...)

Use Metal System Trace with the **Points of Interest** instrument added
to the recording.  Captured intervals appear under subsystem ``ltx`` on
a per-phase timeline you can correlate against GPU dispatches by
timestamp.  ``xctrace export`` exposes them under the ``os-signpost``
schema.

Recommended terminal capture command, after starting the generation with
``LTX_PROFILE_PAUSE_BEFORE_DENOISE=1`` and before pressing Enter::

    xcrun xctrace record \\
        --template "Metal System Trace" \\
        --instrument "Points of Interest" \\
        --attach <PID> \\
        --output <output.trace> \\
        --no-prompt

To export the signposts after capture::

    xcrun xctrace export --input <output.trace> \\
        --xpath '//trace-toc/run/data/table[@schema="os-signpost"]' \\
        --output signposts.xml

Adding a new phase
------------------
1. Add an ``LTX_PHASE(new_name)`` line to ``_signpost.c``.
2. Add the name to ``_PHASES`` below.
3. Remove the cached ``_signpost.dylib`` (the auto-rebuilder will
   recompile on next use).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, TextIO


# The 8 transformer sub-ops we instrument, plus event-only markers.
# Must match the LTX_PHASE() invocations in _signpost.c.
_PHASES = (
    # Block-level phase signposts (the 8 sub-ops of BasicAVTransformerBlock).
    "video_self_attn",
    "video_text_ca",
    "audio_self_attn",
    "audio_text_ca",
    "a2v_cross",
    "v2a_cross",
    "video_ff",
    "audio_ff",
    # Sub-phase signposts (nest inside the parent phase signposts).
    # Added 2026-05-17 for brutal-efficiency hunt: attribute the
    # ~9 s/step "non-SDPA in video_self_attn phase" + 17 s/step
    # video_ff buckets to specific sub-ops.  Aggregate across all
    # attention call sites; the dominant contributor is video_self_attn.
    "attn_qkv",       # V + Q + K + gate_logits projections + q_norm/k_norm + RoPE
    "attn_sdpa",      # mx.fast.scaled_dot_product_attention call only
    "attn_out",       # gate apply (V2) + output projection
    "v_ff_adaln",     # AdaLN modulation inside video_ff
    "v_ff_inner",     # self.ff(...) call: project_in + GELU + project_out
)

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR / "_signpost.c"
_LIB = _THIS_DIR / "_signpost.dylib"

# State
_lib: Optional[ctypes.CDLL] = None
_sids: dict[str, int] = {}
_init_lock = threading.Lock()
_init_attempted = False

# Optional sidecar log: timestamped phase begin/end records.  Works as a
# fallback when the Instruments trace template doesn't capture our
# os_signpost categories.  Path comes from LTX_PROFILE_SIGNPOST_LOG env var.
# Format: one line per event:
#     <ns_since_epoch> <begin|end> <phase>
# Timestamps are time.monotonic_ns() — same clock Metal traces use.
_log_fh = None
_log_block_counter = 0  # per-process; reset by caller if needed

# Sync mode: True when LTX_PROFILE_SIGNPOSTS_SYNC=1 is set.
#
# When enabled, callers should invoke ``signpost_barrier(out)`` at the end
# of each ``with signpost(...)`` block, passing the phase's output array.
# This forces mx.eval() on the array, draining MLX's lazy graph so the
# signpost end timestamp reflects "GPU done with this phase" rather than
# "Python finished enqueueing".  Without this, signposts fire microseconds
# apart while GPU work spans seconds and time-based attribution fails.
#
# NOTE: mx.synchronize() alone does NOT work — it only waits for ALREADY-
# DISPATCHED GPU work to finish.  Lazy ops that haven't been kicked off
# to the GPU yet (because nothing has eval'd them) are not affected.
_sync_mode = False


def _build_dylib() -> None:
    """Compile _signpost.c -> _signpost.dylib via clang."""
    cmd = [
        "clang",
        "-O2",
        "-shared",
        "-fPIC",
        "-Wall",
        "-o",
        str(_LIB),
        str(_SRC),
    ]
    subprocess.run(cmd, check=True)


def _needs_rebuild() -> bool:
    if not _LIB.exists():
        return True
    return _SRC.stat().st_mtime > _LIB.stat().st_mtime


def _try_init() -> None:
    """Idempotent.  Builds + loads the dylib, allocates one sid per phase.

    Also opens the sidecar log file if LTX_PROFILE_SIGNPOST_LOG is set.
    The dylib (os_signpost) and the sidecar log are independent — either
    or both can be active.
    """
    global _lib, _init_attempted, _log_fh
    if _init_attempted:
        return
    with _init_lock:
        if _init_attempted:
            return
        _init_attempted = True
        if not os.environ.get("LTX_PROFILE_SIGNPOSTS"):
            return
        # Open sidecar log (optional)
        log_path = os.environ.get("LTX_PROFILE_SIGNPOST_LOG")
        if log_path:
            try:
                _log_fh = open(log_path, "w", buffering=1)
                _log_fh.write(
                    f"# LTX signpost log  pid={os.getpid()}  "
                    f"phases={','.join(_PHASES)}\n"
                )
                print(
                    f"  [LTX_PROFILE_SIGNPOSTS] sidecar log: {log_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"  [LTX_PROFILE_SIGNPOSTS] sidecar log failed: {exc}",
                    flush=True,
                )
                _log_fh = None
        try:
            if _needs_rebuild():
                _build_dylib()
            lib = ctypes.CDLL(str(_LIB))
            lib.ltx_signpost_id_generate.restype = ctypes.c_uint64
            lib.ltx_signpost_enabled.restype = ctypes.c_int
            for phase in _PHASES:
                begin = getattr(lib, f"ltx_signpost_begin_{phase}")
                end = getattr(lib, f"ltx_signpost_end_{phase}")
                begin.argtypes = [ctypes.c_uint64]
                end.argtypes = [ctypes.c_uint64]
                begin.restype = None
                end.restype = None
                _sids[phase] = lib.ltx_signpost_id_generate()
            lib.ltx_signpost_event_step_begin.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            lib.ltx_signpost_event_step_end.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            lib.ltx_signpost_event_block.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            lib.ltx_signpost_event_step_begin.restype = None
            lib.ltx_signpost_event_step_end.restype = None
            lib.ltx_signpost_event_block.restype = None
            _sids["__events__"] = lib.ltx_signpost_id_generate()
            _lib = lib
            print(
                f"  [LTX_PROFILE_SIGNPOSTS] loaded {_LIB.name}, "
                f"{len(_PHASES)} phases, enabled={lib.ltx_signpost_enabled()}",
                flush=True,
            )
        except Exception as exc:
            print(f"  [LTX_PROFILE_SIGNPOSTS] disabled: {exc}", flush=True)
            _lib = None
        global _sync_mode
        if os.environ.get("LTX_PROFILE_SIGNPOSTS_SYNC"):
            _sync_mode = True
            print(
                "  [LTX_PROFILE_SIGNPOSTS_SYNC] phase-output mx.eval() at "
                "barrier — slower, but accurate attribution.",
                flush=True,
            )


def enabled() -> bool:
    _try_init()
    return _lib is not None


@contextmanager
def signpost(phase: str) -> Iterator[None]:
    """Wrap a region of code with an os_signpost interval.

    No-op (and free) when LTX_PROFILE_SIGNPOSTS is unset.
    """
    _try_init()
    if _lib is None and _log_fh is None:
        yield
        return
    sid = _sids.get(phase) if _lib is not None else None
    if _lib is not None and sid is None:
        # Unknown phase — silently no-op rather than crash a perf run
        yield
        return
    begin = getattr(_lib, f"ltx_signpost_begin_{phase}", None) if _lib is not None else None
    end = getattr(_lib, f"ltx_signpost_end_{phase}", None) if _lib is not None else None
    if begin is not None:
        begin(sid)
    if _log_fh is not None:
        _log_fh.write(f"{time.monotonic_ns()} begin {phase}\n")
    try:
        yield
    finally:
        # Caller is responsible for calling signpost_barrier(out_array)
        # inside the with block when LTX_PROFILE_SIGNPOSTS_SYNC=1 is set.
        # Without that, this signpost end fires immediately after Python
        # finishes enqueueing the phase's ops — long before GPU executes
        # them — and time-based attribution fails.
        if end is not None:
            end(sid)
        if _log_fh is not None:
            _log_fh.write(f"{time.monotonic_ns()} end {phase}\n")


def signpost_barrier(*arrays) -> None:
    """Force a phase's lazy ops to evaluate so the signpost end timestamp
    reflects "GPU done with this phase" rather than "Python done enqueueing".

    No-op unless LTX_PROFILE_SIGNPOSTS_SYNC=1.  Callers pass the array(s)
    that the phase produces; mx.eval(*arrays) forces MLX to dispatch and
    wait for the GPU work to complete.
    """
    if not _sync_mode:
        return
    if not arrays:
        return
    # Lazy import: only pulled in when sync mode is actually engaged.
    import mlx.core as _mx
    _mx.eval(*arrays)


def step_begin(step_idx: int) -> None:
    _try_init()
    if _lib is None:
        return
    _lib.ltx_signpost_event_step_begin(_sids["__events__"], int(step_idx))


def step_end(step_idx: int) -> None:
    _try_init()
    if _lib is None:
        return
    _lib.ltx_signpost_event_step_end(_sids["__events__"], int(step_idx))


def block_event(block_idx: int) -> None:
    _try_init()
    if _lib is None:
        return
    _lib.ltx_signpost_event_block(_sids["__events__"], int(block_idx))


# Eager init at module import.  Without this, _try_init() fires lazily on
# the first signpost(...) call — which lands inside the denoise loop, in
# the middle of progress-bar output.  The init prints get clobbered by
# the in-place progress bar.  Running init at import time makes the
# "[LTX_PROFILE_SIGNPOSTS] loaded ..." prints happen before any pipeline
# work starts.  No-op when env vars aren't set.
_try_init()
