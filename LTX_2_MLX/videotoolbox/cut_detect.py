"""Scene-cut detector for resetting VSR's prev-frame chain on hard cuts.

Pure numpy on the CPU. VSR's `balanced` mode propagates previous source +
output frames into the upscaler for temporal coherence; across a hard cut
that's the wrong context and produces ghosting around the cut frame. For
single-shot LTX latents there are no cuts (mode="off" is correct). For
arbitrary `--video` input on edited footage, enable one of the detection
modes so VSR resets at cut boundaries.

Two algorithms:
  simple  Downsampled-pixel mean absolute difference. ~1ms/frame. Catches
          hard cuts cleanly; doesn't flag dissolves (which don't ghost in
          VSR anyway because the temporal coherence kinda matches).
  hist    Per-channel 32-bin histogram chi-squared distance. ~3ms/frame.
          More robust to fast motion than simple-pixel diff.

False positives are cheap (one frame of "no temporal context", visually
invisible). False negatives let a cut ghost — tune the threshold down.
"""

from __future__ import annotations

import numpy as np


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Coerce any frame format (uint8 RGB, uint8 RGBA, fp16 RGBA, fp32 RGBA)
    to uint8 RGB for histogram / thumbnail work.
    """
    if frame.dtype == np.float16 or frame.dtype == np.float32:
        return np.clip(frame[..., :3] * 255.0, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        return frame[..., :3]
    return frame


def _frame_thumbnail(frame: np.ndarray, target_size: int = 32) -> np.ndarray:
    rgb = _to_uint8_rgb(frame)
    h, w = rgb.shape[:2]
    step_h = max(1, h // target_size)
    step_w = max(1, w // target_size)
    return np.ascontiguousarray(rgb[::step_h, ::step_w])


def _frame_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    rgb = _to_uint8_rgb(frame)
    hists = [np.histogram(rgb[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
    return np.concatenate(hists).astype(np.float32)


class CutDetector:
    """Detects hard cuts between consecutive frames.

    Modes:
        "off"     no-op
        "simple"  downsampled-pixel MAD. threshold ~0.2-0.35 typical.
        "hist"    per-channel histogram chi-squared. threshold ~0.4-0.8.

    Always returns False on the first frame (no previous to compare).
    """

    def __init__(self, mode: str, threshold: float):
        self.mode = mode
        self.threshold = float(threshold)
        self._prev: np.ndarray | None = None

    def is_cut(self, frame: np.ndarray) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "simple":
            curr = _frame_thumbnail(frame)
            if self._prev is None:
                self._prev = curr
                return False
            diff = np.abs(curr.astype(np.int16) - self._prev.astype(np.int16))
            mad = diff.mean() / 255.0
            self._prev = curr
            return bool(mad > self.threshold)
        if self.mode == "hist":
            curr = _frame_histogram(frame)
            if self._prev is None:
                self._prev = curr
                return False
            a, b = self._prev, curr
            eps = 1e-6
            chi2 = ((a - b) ** 2 / (a + b + eps)).sum()
            norm = chi2 / (a.sum() + eps)
            self._prev = curr
            return bool(norm > self.threshold)
        raise ValueError(f"Unknown cut-detect mode: {self.mode!r}")
