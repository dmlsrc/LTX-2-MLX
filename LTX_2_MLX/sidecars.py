"""Sidecar I/O for run dumps (latents, text conditioning).

A *sidecar* is a named bundle of MLX arrays plus string metadata written next to
a generation's output. Two on-disk formats are supported:

- ``safetensors`` (modern): :func:`mlx.core.save_safetensors` stores arrays in
  their native dtype (bfloat16 included) with the string metadata in the
  safetensors header. NumPy is not involved.
- ``npz`` (legacy): ``numpy.savez``. NumPy has no bfloat16, so bf16 arrays are
  widened to float32 and their original dtype recorded in a ``<key>_mlx_dtype``
  entry; string metadata is stored as 0-d string arrays. NumPy is imported
  lazily, only when this path actually runs.

This module is the single place NumPy touches the runtime. Importing it (and the
rest of the library) does not import NumPy; only the legacy save/load path does,
lazily. Keep all sidecar serialization here so NumPy stays reasoned-about and
isolated rather than sprinkled across pipelines and scripts.
"""

from __future__ import annotations

import os

import mlx.core as mx

SIDECAR_FORMAT_ENV = "LTX_SIDECAR_FORMAT"
# Default to the modern safetensors format: native dtypes (bf16 included) + header
# metadata, NumPy-free. All readers go through load_sidecar, which also reads the
# legacy npz format, so existing .npz sidecars keep loading. Set
# LTX_SIDECAR_FORMAT=npz to write the legacy NumPy format.
_DEFAULT_FORMAT = "safetensors"
_FORMATS = ("npz", "safetensors")


def default_format() -> str:
    """Resolve the sidecar format from ``$LTX_SIDECAR_FORMAT`` (else the default)."""
    fmt = os.environ.get(SIDECAR_FORMAT_ENV, _DEFAULT_FORMAT).lower()
    if fmt not in _FORMATS:
        raise ValueError(
            f"{SIDECAR_FORMAT_ENV}={fmt!r}; expected one of {_FORMATS}"
        )
    return fmt


def _ext_for(fmt: str) -> str:
    return ".safetensors" if fmt == "safetensors" else ".npz"


def sidecar_path(base_path: str, suffix: str = "", fmt: str | None = None) -> str:
    """Path for a sidecar next to ``base_path`` (extension stripped), with an
    optional ``suffix`` (e.g. ``"_text"``) and the format's extension."""
    fmt = fmt or default_format()
    stem = os.path.splitext(base_path)[0]
    return f"{stem}{suffix}{_ext_for(fmt)}"


def find_sidecar(base_path: str, suffix: str = "") -> str | None:
    """Return the existing sidecar next to ``base_path`` (extension stripped) with
    the given ``suffix``, trying safetensors then npz; None if neither exists.

    Use this when deriving a sidecar path from a run's output stem so a reader
    finds the file regardless of which format it was written in.
    """
    stem = os.path.splitext(str(base_path))[0]
    for ext in (".safetensors", ".npz"):
        candidate = f"{stem}{suffix}{ext}"
        if os.path.exists(candidate):
            return candidate
    return None


def save_sidecar(
    path: str,
    arrays: dict[str, mx.array],
    metadata: dict[str, str] | None = None,
    *,
    fmt: str | None = None,
) -> str:
    """Write ``arrays`` plus string ``metadata`` to ``path``.

    ``fmt`` defaults to :func:`default_format`. The path's extension is forced to
    match the chosen format. Returns the path actually written.
    """
    metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
    fmt = (fmt or default_format()).lower()
    if fmt not in _FORMATS:
        raise ValueError(f"fmt={fmt!r}; expected one of {_FORMATS}")
    out = os.path.splitext(path)[0] + _ext_for(fmt)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if fmt == "safetensors":
        if arrays:
            mx.eval(*arrays.values())
        mx.save_safetensors(out, dict(arrays), metadata=metadata)
    else:
        _save_npz(out, arrays, metadata)
    return out


def load_sidecar(path: str) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Load a sidecar as ``(arrays, metadata)``, detecting the format from the
    extension. Arrays come back in their original MLX dtype either way."""
    if path.endswith(".safetensors"):
        arrays, metadata = mx.load(path, return_metadata=True)
        return dict(arrays), {str(k): str(v) for k, v in metadata.items()}
    return _load_npz(path)


# --- legacy npz backend: NumPy imported lazily, only here ---


def _save_npz(out: str, arrays: dict[str, mx.array], metadata: dict[str, str]) -> None:
    import numpy as np  # lazy: NumPy loads only for the legacy path

    payload = {}
    for key, arr in arrays.items():
        mx.eval(arr)
        # NumPy has no bf16: widen to float32 and record the original dtype.
        if arr.dtype == mx.bfloat16:
            payload[key] = np.array(arr.astype(mx.float32))
        else:
            payload[key] = np.array(arr)
        payload[f"{key}_mlx_dtype"] = np.array(str(arr.dtype))
    for mkey, mval in metadata.items():
        payload[mkey] = np.array(mval)
    np.savez(out, **payload)


def _load_npz(path: str) -> tuple[dict[str, mx.array], dict[str, str]]:
    import numpy as np  # lazy

    arrays: dict[str, mx.array] = {}
    metadata: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as data:
        dtype_tags = {
            k[: -len("_mlx_dtype")]: str(data[k])
            for k in data.files
            if k.endswith("_mlx_dtype")
        }
        for key in data.files:
            if key.endswith("_mlx_dtype"):
                continue
            val = data[key]
            if val.dtype.kind in ("U", "S"):  # string metadata
                metadata[key] = str(val)
                continue
            arr = mx.array(val)
            target = _mlx_dtype_from_name(dtype_tags.get(key))
            if target is not None:
                arr = arr.astype(target)
            arrays[key] = arr
    # Surface the dtype tags in metadata too, for callers that want them.
    metadata.update({f"{k}_mlx_dtype": v for k, v in dtype_tags.items()})
    return arrays, metadata


def _mlx_dtype_from_name(name: str | None) -> mx.Dtype | None:
    """``"mlx.core.bfloat16"`` / ``"bfloat16"`` -> ``mx.bfloat16``; None if unknown."""
    if not name:
        return None
    candidate = getattr(mx, name.rsplit(".", 1)[-1], None)
    return candidate if isinstance(candidate, mx.Dtype) else None
