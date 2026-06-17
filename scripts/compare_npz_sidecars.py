#!/usr/bin/env python3
"""Compare two NumPy NPZ sidecars.

Designed for LTX generation sidecars, but intentionally generic:

  - compares every common NPZ key by default
  - reports exact equality plus numeric drift stats for array payloads
  - can compare adjacent ``*_text.npz`` sidecars with ``--with-text``
  - can compare selected fields from adjacent ``*_run.json`` logs

Examples:
    scripts/compare_npz_sidecars.py a.npz b.npz
    scripts/compare_npz_sidecars.py a.npz b.npz --with-text --require-exact
    scripts/compare_npz_sidecars.py a.npz b.npz --keys final_video_latent final_audio_latent
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Run as scripts/compare_npz_sidecars.py: put the repo root on the path so the
# centralized sidecar loader (which transparently handles both the npz and
# safetensors backends) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from LTX_2_MLX.sidecars import load_sidecar  # noqa: E402

DEFAULT_RUN_KEYS = (
    "prompt",
    "negative_prompt",
    "parameters.height",
    "parameters.width",
    "parameters.num_frames",
    "parameters.seed",
    "parameters.generate_audio",
    "parameters.fast_mode",
    "parameters.cfg_scale",
    "parameters.compute_dtype",
    "parameters.config_weights_path",
    "parameters.transformer_weights_path",
    "parameters.gemma_path",
)

FALLBACK_WRAP_WIDTH = 120
MIN_WRAP_WIDTH = 60
FIELD_LABEL_WIDTH = len("right")
MIN_KEY_WIDTH = 32


@dataclass(frozen=True)
class CompareResult:
    name: str
    exact: bool
    comparable: bool = True
    missing: bool = False


@dataclass(frozen=True)
class Colors:
    enabled: bool

    def _paint(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def green(self, text: str) -> str:
        return self._paint("32", text)

    def yellow(self, text: str) -> str:
        return self._paint("33", text)

    def red(self, text: str) -> str:
        return self._paint("31", text)


def _colors(mode: str) -> Colors:
    if mode == "always":
        return Colors(enabled=True)
    if mode == "never":
        return Colors(enabled=False)
    enabled = (
        sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )
    return Colors(enabled=enabled)


def _exact_status(
    exact: bool,
    *,
    colors: Colors,
    comparable: bool = True,
    missing: bool = False,
) -> str:
    text = f"exact={exact}"
    if exact:
        return colors.green(text)
    if missing or not comparable:
        return colors.red(text)
    return colors.yellow(text)


def _missing_status(colors: Colors) -> str:
    return colors.red("missing")


def _adjacent_text_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_text{path.suffix}")


def _adjacent_run_log_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_run.json")


def _is_numeric(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)


def _key_width(keys: Iterable[str]) -> int:
    return max([MIN_KEY_WIDTH, *(len(key) for key in keys)])


def _dtype_label(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.str_):
        return "str"
    if np.issubdtype(dtype, np.bytes_):
        return "bytes"
    return str(dtype)


def _array_descriptor(array: np.ndarray) -> str:
    dtype = _dtype_label(array.dtype)
    if array.shape == ():
        return f"scalar[{dtype}]"
    return f"shape={array.shape} dtype={dtype}"


def _pair_descriptor(left: np.ndarray, right: np.ndarray) -> str:
    left_desc = _array_descriptor(left)
    right_desc = _array_descriptor(right)
    if left_desc == right_desc:
        return left_desc
    return f"left={left_desc} right={right_desc}"


def _scalar_repr(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return f"array(shape={value.shape}, dtype={value.dtype})"
    return str(value)


def _terminal_width() -> int:
    return max(shutil.get_terminal_size((FALLBACK_WRAP_WIDTH, 20)).columns, MIN_WRAP_WIDTH)


def _print_wrapped_field(
    label: str,
    value: Any,
    *,
    indent: str = "  ",
    label_width: int = FIELD_LABEL_WIDTH,
    wrap_width: int | None = None,
) -> None:
    text = _scalar_repr(value)
    prefix = f"{indent}{label}: " + (" " * max(label_width - len(label), 0))
    width = wrap_width or _terminal_width()
    wrapped = textwrap.wrap(
        text,
        width=max(width - len(prefix), 20),
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped:
        print(prefix)
        return
    print(prefix + wrapped[0])
    continuation = " " * len(prefix)
    for line in wrapped[1:]:
        print(continuation + line)


def _safe_float_array(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.float64)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float64)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float64)
    if np.issubdtype(array.dtype, np.complexfloating):
        return array.astype(np.complex128)
    raise TypeError(f"cannot convert dtype {array.dtype} to float stats")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.iscomplexobj(a_flat) or np.iscomplexobj(b_flat):
        dot = np.vdot(a_flat, b_flat)
        a_norm = np.sqrt(np.vdot(a_flat, a_flat).real)
        b_norm = np.sqrt(np.vdot(b_flat, b_flat).real)
        if a_norm == 0 or b_norm == 0:
            return 1.0 if a_norm == b_norm else math.nan
        return float((dot / (a_norm * b_norm)).real)
    a_norm = np.linalg.norm(a_flat)
    b_norm = np.linalg.norm(b_flat)
    if a_norm == 0 or b_norm == 0:
        return 1.0 if a_norm == b_norm else math.nan
    return float(np.dot(a_flat, b_flat) / (a_norm * b_norm))


def _compare_numeric(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    key_width: int,
    colors: Colors,
) -> CompareResult:
    exact = bool(np.array_equal(left, right))
    if left.shape != right.shape:
        print(
            f"{name:<{key_width}} "
            f"{_exact_status(exact, colors=colors, comparable=False)} "
            f"{colors.red('shape_mismatch')} "
            f"left={_array_descriptor(left)} right={_array_descriptor(right)}"
        )
        return CompareResult(name, exact=False, comparable=False)

    left_f = _safe_float_array(left)
    right_f = _safe_float_array(right)
    delta = left_f - right_f
    abs_delta = np.abs(delta)
    max_abs = float(abs_delta.max()) if abs_delta.size else 0.0
    mean_abs = float(abs_delta.mean()) if abs_delta.size else 0.0
    rms_abs = float(np.sqrt(np.mean(abs_delta * abs_delta))) if abs_delta.size else 0.0
    denom = np.maximum(np.abs(right_f), 1e-9)
    max_rel = float((abs_delta / denom).max()) if abs_delta.size else 0.0
    cos = _cosine(left_f, right_f)
    nonfinite_left = int((~np.isfinite(left_f)).sum())
    nonfinite_right = int((~np.isfinite(right_f)).sum())

    print(
        f"{name:<{key_width}} {_exact_status(exact, colors=colors)} "
        f"{_pair_descriptor(left, right)} "
        f"max_abs={max_abs:.12g} mean_abs={mean_abs:.12g} "
        f"rms={rms_abs:.12g} max_rel={max_rel:.12g} cos={cos:.12f}"
    )
    if nonfinite_left or nonfinite_right:
        print(f"{'':<{key_width}} {colors.red('nonfinite')} left={nonfinite_left} right={nonfinite_right}")
    return CompareResult(name, exact=exact)


def _compare_value(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    key_width: int,
    wrap_width: int | None,
    colors: Colors,
) -> CompareResult:
    exact = bool(np.array_equal(left, right))
    if left.shape == right.shape and _is_numeric(left) and _is_numeric(right):
        return _compare_numeric(name, left, right, key_width=key_width, colors=colors)

    print(
        f"{name:<{key_width}} {_exact_status(exact, colors=colors)} "
        f"{_pair_descriptor(left, right)}"
    )
    _print_wrapped_field("left", left, wrap_width=wrap_width)
    _print_wrapped_field("right", right, wrap_width=wrap_width)
    return CompareResult(name, exact=exact)


def _resolve_keys(
    left_keys: Iterable[str],
    right_keys: Iterable[str],
    requested: list[str] | None,
    include_missing: bool,
) -> list[str]:
    left_set = set(left_keys)
    right_set = set(right_keys)
    if requested:
        return requested
    keys = left_set | right_set if include_missing else left_set & right_set
    return sorted(keys)


def _load_sidecar_as_numpy(path: Path) -> dict[str, np.ndarray]:
    """Load a sidecar via the centralized loader and return a NumPy-keyed dict.

    The comparison math below is NumPy, so each MLX array is converted once.
    bf16 has no NumPy dtype, so widen it to float32 (which is also what the npz
    backend stored on disk) before the stats code touches it. The ``_mlx_dtype``
    tags and string metadata come back through ``load_sidecar``'s metadata dict,
    so they remain comparable keys exactly as the raw npz exposed them.
    """
    import mlx.core as mx

    arrays, metadata = load_sidecar(str(path))
    out: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float32)
        out[key] = np.array(value)
    for key, value in metadata.items():
        out[key] = np.array(value)
    return out


def _npz_selected_keys(
    left_path: Path,
    right_path: Path,
    *,
    keys: list[str] | None,
    include_missing: bool,
) -> list[str]:
    left = _load_sidecar_as_numpy(left_path)
    right = _load_sidecar_as_numpy(right_path)
    return _resolve_keys(left.keys(), right.keys(), keys, include_missing)


def compare_npz(
    left_path: Path,
    right_path: Path,
    *,
    label: str,
    keys: list[str] | None,
    include_missing: bool,
    wrap_width: int | None,
    key_width: int,
    colors: Colors,
) -> list[CompareResult]:
    print(f"\n== {label} ==")
    _print_wrapped_field("left", left_path, indent="", wrap_width=wrap_width)
    _print_wrapped_field("right", right_path, indent="", wrap_width=wrap_width)

    left = _load_sidecar_as_numpy(left_path)
    right = _load_sidecar_as_numpy(right_path)
    selected_keys = _resolve_keys(left.keys(), right.keys(), keys, include_missing)
    if not selected_keys:
        print("No comparable keys.")
        return []

    results: list[CompareResult] = []
    for key in selected_keys:
        if key not in left or key not in right:
            print(
                f"{key:<{key_width}} {_missing_status(colors)} "
                f"left={key not in left} right={key not in right}"
            )
            results.append(CompareResult(key, exact=False, comparable=False, missing=True))
            continue
        results.append(
            _compare_value(
                key,
                left[key],
                right[key],
                key_width=key_width,
                wrap_width=wrap_width,
                colors=colors,
            )
        )
    return results


def _json_get(data: dict[str, Any], dotted_key: str) -> Any:
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def compare_run_logs(
    left_path: Path,
    right_path: Path,
    keys: list[str],
    *,
    wrap_width: int | None,
    key_width: int,
    colors: Colors,
) -> list[CompareResult]:
    print("\n== run logs ==")
    _print_wrapped_field("left", left_path, indent="", wrap_width=wrap_width)
    _print_wrapped_field("right", right_path, indent="", wrap_width=wrap_width)
    with left_path.open("r", encoding="utf-8") as f:
        left = json.load(f)
    with right_path.open("r", encoding="utf-8") as f:
        right = json.load(f)

    results: list[CompareResult] = []
    for key in keys:
        try:
            left_value = _json_get(left, key)
            right_value = _json_get(right, key)
        except KeyError:
            print(f"{key:<{key_width}} {_missing_status(colors)}")
            results.append(CompareResult(key, exact=False, comparable=False, missing=True))
            continue
        exact = left_value == right_value
        print(f"{key:<{key_width}} {_exact_status(exact, colors=colors)}")
        _print_wrapped_field("left", left_value, wrap_width=wrap_width)
        _print_wrapped_field("right", right_value, wrap_width=wrap_width)
        results.append(CompareResult(key, exact=exact))
    return results


def _print_summary(results: list[CompareResult], *, colors: Colors) -> None:
    if not results:
        return
    exact = sum(1 for r in results if r.exact)
    missing = sum(1 for r in results if r.missing)
    mismatched = len(results) - exact
    print("\n== summary ==")
    exact_text = colors.green(str(exact)) if mismatched == 0 else str(exact)
    if mismatched == 0:
        mismatched_text = colors.green(str(mismatched))
    elif missing:
        mismatched_text = colors.red(str(mismatched))
    else:
        mismatched_text = colors.yellow(str(mismatched))
    missing_text = colors.red(str(missing)) if missing else colors.green(str(missing))
    print(
        f"compared={len(results)} exact={exact_text} "
        f"mismatched={mismatched_text} missing={missing_text}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("left", type=Path, help="Left .npz sidecar.")
    parser.add_argument("right", type=Path, help="Right .npz sidecar.")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Specific NPZ keys to compare. Defaults to all common keys.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Report keys present in only one file. Defaults to common keys only.",
    )
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="Also compare adjacent *_text.npz sidecars.",
    )
    parser.add_argument("--left-text", type=Path, default=None, help="Explicit left text sidecar.")
    parser.add_argument("--right-text", type=Path, default=None, help="Explicit right text sidecar.")
    parser.add_argument(
        "--with-run-log",
        action="store_true",
        help="Also compare selected fields from adjacent *_run.json logs.",
    )
    parser.add_argument("--left-run-log", type=Path, default=None, help="Explicit left run log JSON.")
    parser.add_argument("--right-run-log", type=Path, default=None, help="Explicit right run log JSON.")
    parser.add_argument(
        "--run-keys",
        nargs="+",
        default=list(DEFAULT_RUN_KEYS),
        help="Dotted JSON keys to compare when --with-run-log is set.",
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=None,
        help=(
            "Output wrap width. Defaults to the detected terminal width "
            f"(fallback {FALLBACK_WRAP_WIDTH} when redirected)."
        ),
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize status by severity: green exact, yellow mismatch, red missing/structural.",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help=(
            "Retained for compatibility; now a no-op. Sidecars load through the "
            "centralized loader, which reads object-free npz (and safetensors)."
        ),
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Exit non-zero if any compared value differs or is missing.",
    )
    args = parser.parse_args()
    colors = _colors(args.color)

    left_text = args.left_text or _adjacent_text_path(args.left)
    right_text = args.right_text or _adjacent_text_path(args.right)
    left_run_log = args.left_run_log or _adjacent_run_log_path(args.left)
    right_run_log = args.right_run_log or _adjacent_run_log_path(args.right)

    key_groups = [
        _npz_selected_keys(
            args.left,
            args.right,
            keys=args.keys,
            include_missing=args.include_missing,
        )
    ]
    if args.with_text:
        key_groups.append(
            _npz_selected_keys(
                left_text,
                right_text,
                keys=None,
                include_missing=args.include_missing,
            )
        )
    if args.with_run_log:
        key_groups.append(args.run_keys)
    global_key_width = _key_width(key for group in key_groups for key in group)

    results: list[CompareResult] = []
    results.extend(
        compare_npz(
            args.left,
            args.right,
            label="latents",
            keys=args.keys,
            include_missing=args.include_missing,
            wrap_width=args.wrap_width,
            key_width=global_key_width,
            colors=colors,
        )
    )

    if args.with_text:
        results.extend(
            compare_npz(
                left_text,
                right_text,
                label="text conditioning",
                keys=None,
                include_missing=args.include_missing,
                wrap_width=args.wrap_width,
                key_width=global_key_width,
                colors=colors,
            )
        )

    if args.with_run_log:
        results.extend(
            compare_run_logs(
                left_run_log,
                right_run_log,
                args.run_keys,
                wrap_width=args.wrap_width,
                key_width=global_key_width,
                colors=colors,
            )
        )

    _print_summary(results, colors=colors)
    if args.require_exact and any(not r.exact for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
