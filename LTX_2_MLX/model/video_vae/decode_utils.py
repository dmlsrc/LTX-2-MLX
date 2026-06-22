"""Whole-clip VAE decode helper, decoder-agnostic (``decoder(latent, timestep=...)``).

``decode_single_pass`` decodes the entire latent in one call (the unsafe,
seam-free path). The chunked, streaming path lives in ``tiling.decode_streaming``;
the legacy accumulate-and-blend decode now lives only in
``scripts/decode_ab_harness.py`` as the A/B baseline. Typed ``Any`` to keep this
module concrete-class-free.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

_log = logging.getLogger(__name__)


def decode_single_pass(
    latent: Any, decoder: Any, *, timestep: float | None = 0.05
) -> mx.array:
    """Decode the WHOLE latent in one decoder call -- no temporal chunking.

    The unsafe / max-quality path. Returns the raw ``(B, C, T, H, W)`` decoder
    output in [-1, 1]; the caller converts. Holds the entire decoded video in
    memory, and on sequences longer than ~7 latent frames it triggers the MLX
    3D-convolution bug that injects noise into early frames -- exactly what the
    chunked ``decode_streaming`` path exists to avoid. Use only for clips you
    know fit and want decoded seam-free, or to exercise the conv3d path directly.
    """
    if latent.ndim == 4:
        latent = latent[None]
    _log.warning(
        "decode_single_pass: decoding %d latent frames in ONE pass (no temporal "
        "chunking). Peak memory is the whole video, and >~7 frames hits the MLX "
        "conv3d early-frame noise bug -- use decode_streaming unless it fits.",
        int(latent.shape[2]),
    )
    return decoder(latent, timestep=timestep)
