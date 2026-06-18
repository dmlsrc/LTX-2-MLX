"""Native safetensors header reading - no safetensors lib, no numpy, no tensors.

The safetensors format prefixes the file with an 8-byte little-endian u64 header
length, followed by that many bytes of UTF-8 JSON. The JSON maps each tensor name
to ``{dtype, shape, data_offsets}`` plus an optional ``__metadata__`` string dict.

Reading just the header is cheap even on a 40 GB checkpoint, so it is the right
tool for version / config detection and dtype inspection. (``mx.load`` would
otherwise materialize every tensor just to surface the header metadata.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Parse and return the full safetensors JSON header (no tensor data)."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(header_len))


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    """Return the ``__metadata__`` string dict from a safetensors header.

    Native replacement for ``safetensors.safe_open(path).metadata()``; empty
    dict when the file carries no metadata block.
    """
    return read_safetensors_header(path).get("__metadata__", {}) or {}


def read_safetensors_dtypes(path: str | Path) -> dict[str, str]:
    """Map tensor key -> raw dtype string (``F32``, ``BF16``, ``F8_E4M3``, ...).

    Works for dtypes MLX cannot represent, since the tensors are never read.
    """
    header = read_safetensors_header(path)
    return {
        key: entry["dtype"]
        for key, entry in header.items()
        if key != "__metadata__" and isinstance(entry, dict) and "dtype" in entry
    }
