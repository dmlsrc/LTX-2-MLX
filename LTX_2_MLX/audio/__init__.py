"""Audio utilities for LTX-2-MLX outputs (post-VAE-decode).

Currently only the sequence-start onset-spike mitigation lives here; if
more audio post-processing accrues (notch filters, dynamic range checks,
loudness measurement, ...) it joins this package.

The mitigation is documented in `docs/AUDIO_ISSUES.md` ->
"Sequence-Start Audio Spike".  See `onset.py` for the implementation
and the threshold rationale.
"""

from .onset import (
    DEFAULT_DETECT_THRESHOLD_RATIO,
    DEFAULT_DETECT_WINDOW_MS,
    DEFAULT_SILENCE_END_MS,
    DEFAULT_SILENCE_RATIO,
    DEFAULT_SILENCE_START_MS,
    DEFAULT_TRIM_MS,
    OnsetTrimResult,
    detect_onset_latent_spike,
    detect_onset_spike,
    flatten_onset_latent,
    mitigate_onset,
    mitigate_onset_latent,
    parse_trim_mode,
    trim_onset,
)

__all__ = [
    "DEFAULT_DETECT_THRESHOLD_RATIO",
    "DEFAULT_DETECT_WINDOW_MS",
    "DEFAULT_SILENCE_END_MS",
    "DEFAULT_SILENCE_RATIO",
    "DEFAULT_SILENCE_START_MS",
    "DEFAULT_TRIM_MS",
    "OnsetTrimResult",
    "detect_onset_latent_spike",
    "detect_onset_spike",
    "flatten_onset_latent",
    "mitigate_onset",
    "mitigate_onset_latent",
    "parse_trim_mode",
    "trim_onset",
]
