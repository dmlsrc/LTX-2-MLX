"""VideoToolbox-backed video post-processing for LTX-2-MLX.

This subpackage bridges MLX-decoded video frames to Apple's hardware video
pipeline: VideoToolbox Super Resolution (`VsrSession`), Frame Rate
Conversion (`VtfrcSession`), and AVAssetWriter (`AVWriter`). All modules
under this namespace require the PyObjC frameworks listed in the `[vsr]`
optional install group; importing them on a base install raises a
SystemExit with the install hint.

Public surface:

    from LTX_2_MLX.videotoolbox import (
        VsrSession,        # spatial upscale via VTSuperResolutionScaler*
        VtfrcSession,      # temporal frame-rate conversion via VTFrameRateConversion*
        AVWriter,          # HEVC + audio encoder via AVAssetWriter
        AudioTrack,        # in-memory PCM -> CMSampleBuffer wrapper
        CutDetector,       # numpy-only scene-cut detector for VSR reset
    )

Submodules expose lower-level helpers:

    pixel_buffers   CVPixelBuffer create/read/write, CMTime helpers
    comparison      Side-by-side composite for `comparison.mp4`

The stacked progress-bar primitives moved to `LTX_2_MLX.progress`
(was previously re-exported here when the bars only had VSR callers).
"""

from __future__ import annotations

from ._compat import autorelease_pool, require_pyobjc

# Re-export the main classes. Each submodule imports its own PyObjC bits via
# _compat, so importing this package only forces the pyobjc check at the
# point a class is actually constructed (via require_pyobjc in each ctor).
from .audio import AudioTrack
from .cut_detect import CutDetector
from .encode import encode_video_videotoolbox
from .vsr import VsrSession
from .temporal import VtfrcSession
from .writer import AVWriter

__all__ = [
    "AudioTrack",
    "AVWriter",
    "CutDetector",
    "VsrSession",
    "VtfrcSession",
    "autorelease_pool",
    "encode_video_videotoolbox",
    "require_pyobjc",
]
