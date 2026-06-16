"""PyObjC framework imports, guarded for environments without the `[vsr]` extra.

All other submodules `from ._compat import av, CoreAudio, ...` and call
`require_pyobjc()` at the top of any user-facing entry point so the missing
dependency surfaces as a clean SystemExit instead of an `ImportError` deep
inside a Metal call.
"""

from __future__ import annotations

from typing import Any

# Sentinel-typed slots so submodules can `from ._compat import av` even when
# PyObjC isn't installed — they just get None and need to call
# `require_pyobjc()` before touching anything.
av: Any = None
CoreAudio: Any = None
CoreMedia: Any = None
Foundation: Any = None
Quartz: Any = None
VideoToolbox: Any = None
libdispatch: Any = None

_IMPORT_ERROR: Exception | None = None

try:
    import objc as _objc  # type: ignore
    import AVFoundation as _av  # type: ignore
    import CoreAudio as _CoreAudio  # type: ignore
    import CoreMedia as _CoreMedia  # type: ignore
    import Foundation as _Foundation  # type: ignore
    import libdispatch as _libdispatch  # type: ignore
    import Quartz as _Quartz  # type: ignore
    import VideoToolbox as _VideoToolbox  # type: ignore

    av = _av
    CoreAudio = _CoreAudio
    CoreMedia = _CoreMedia
    Foundation = _Foundation
    libdispatch = _libdispatch
    Quartz = _Quartz
    VideoToolbox = _VideoToolbox
    objc = _objc
except ImportError as _e:
    _IMPORT_ERROR = _e
    objc = None  # type: ignore


# Alias to keep `vt.` references readable inside submodules.
vt = VideoToolbox


def autorelease_pool():
    """`with autorelease_pool():` to drain transient PyObjC objects per iter.

    PyObjC autoreleased objects (NSData, CIImage, ...) accumulate in the
    process's top-level autorelease pool, which doesn't drain until the
    interpreter exits. Long Python loops that allocate many such objects
    per iteration grow RSS unboundedly. Wrapping the inner-loop body in
    a fresh pool forces drainage at the end of each iteration.
    """
    require_pyobjc()
    return objc.autorelease_pool()


def require_pyobjc() -> None:
    """Raise SystemExit with the install hint if pyobjc isn't importable.

    Call this at the top of every user-facing entry point (CLI main(),
    public class __init__) so the missing-dep error surfaces with a clean
    message instead of an AttributeError on `None.foo`.
    """
    if _IMPORT_ERROR is not None:
        raise SystemExit(
            "LTX_2_MLX.videotoolbox requires PyObjC (VideoToolbox / "
            "AVFoundation / CoreMedia / CoreAudio / Quartz / Foundation / "
            "libdispatch). Install the optional 'vsr' extra:\n"
            "    uv pip install '.[vsr]'\n"
            "or install pyobjc directly:\n"
            "    uv pip install pyobjc\n"
            f"\nUnderlying ImportError: {_IMPORT_ERROR}"
        )
