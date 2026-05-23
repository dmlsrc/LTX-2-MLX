"""Stacked, phase-accurate progress bars for LTX-2-MLX.

Single home for the project's progress UI primitives.  Previously lived
under `LTX_2_MLX.videotoolbox.progress` — the implementation grew out
of the VSR harness work but is now used in three distinct contexts:

* `scripts/generate.py` denoise loops (stage 1 + stage 2 stacked).
* `scripts/vsr_harness.py` VAE chunks + VSR/OUT frames stacked.
* `LTX_2_MLX.videotoolbox.encode` VT-encode frame bar (alone or stacked
  with a caller-owned VAE chunks bar in the streaming path).

The `videotoolbox/` subpackage is for Apple-specific AV plumbing —
keeping the progress UI here under a dedicated `progress/` package
means callers don't have to import PyObjC-touching modules just to
draw a bar.

Public surface:

    from LTX_2_MLX.progress import PhaseBar, StackedPhaseBars

Submodule layout:

    bars.py     PhaseBar / StackedPhaseBars implementation.
"""

from __future__ import annotations

from .bars import (
    PhaseBar,
    StackedPhaseBars,
)

__all__ = [
    "PhaseBar",
    "StackedPhaseBars",
]
