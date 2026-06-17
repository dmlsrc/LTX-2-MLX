#!/usr/bin/env python3
"""Compare two generation sidecars (safetensors or npz).

Recognizes the format of each file it is given (through the centralized sidecar
loader) and behaves the same on either - you can even compare a safetensors
sidecar against an npz one:

  - compares every common key by default
  - reports exact equality plus numeric drift stats for array payloads
  - can compare adjacent ``*_text.*`` sidecars with ``--with-text`` (mirroring
    the given file's extension)
  - can compare selected fields from adjacent ``*_run.json`` logs

Examples:
    scripts/compare_sidecars.py a.safetensors b.safetensors
    scripts/compare_sidecars.py a.npz b.npz --with-text --require-exact
    scripts/compare_sidecars.py a.safetensors b.npz --keys final_video_latent
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

import mlx.core as mx

# Run as scripts/compare_sidecars.py: put the repo root on the path so the
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


def _key_width(keys: Iterable[str]) -> int:
    return max([MIN_KEY_WIDTH, *(len(key) for key in keys)])


def _dtype_label(arr: mx.array) -> str:
    return str(arr.dtype).replace("mlx.core.", "")


def _array_descriptor(arr: mx.array) -> str:
    if arr.ndim == 0:
        return f"scalar[{_dtype_label(arr)}]"
    return f"shape={tuple(arr.shape)} dtype={_dtype_label(arr)}"


def _value_descriptor(value: object) -> str:
    if isinstance(value, mx.array):
        return _array_descriptor(value)
    return f"str[len={len(str(value))}]"


def _pair_descriptor(left: object, right: object) -> str:
    left_desc = _value_descriptor(left)
    right_desc = _value_descriptor(right)
    if left_desc == right_desc:
        return left_desc
    return f"left={left_desc} right={right_desc}"


def _scalar_repr(value: Any) -> str:
    if isinstance(value, mx.array):
        if value.ndim == 0:
            return str(value.item())
        return f"array(shape={tuple(value.shape)}, dtype={_dtype_label(value)})"
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


def _cosine(a: mx.array, b: mx.array) -> float:
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    a_norm = float(mx.sqrt(mx.sum(a_flat * a_flat)).item())
    b_norm = float(mx.sqrt(mx.sum(b_flat * b_flat)).item())
    if a_norm == 0 or b_norm == 0:
        return 1.0 if a_norm == b_norm else math.nan
    return float(mx.sum(a_flat * b_flat).item() / (a_norm * b_norm))


def _compare_array(
    name: str,
    left: mx.array,
    right: mx.array,
    *,
    key_width: int,
    colors: Colors,
) -> CompareResult:
    if left.shape != right.shape:
        print(
            f"{name:<{key_width}} "
            f"{_exact_status(False, colors=colors, comparable=False)} "
            f"{colors.red('shape_mismatch')} "
            f"left={_array_descriptor(left)} right={_array_descriptor(right)}"
        )
        return CompareResult(name, exact=False, comparable=False)

    exact = bool(mx.all(left == right).item())
    left_f = left.astype(mx.float32)
    right_f = right.astype(mx.float32)
    abs_delta = mx.abs(left_f - right_f)
    if left_f.size:
        max_abs = float(mx.max(abs_delta).item())
        mean_abs = float(mx.mean(abs_delta).item())
        rms_abs = float(mx.sqrt(mx.mean(abs_delta * abs_delta)).item())
        denom = mx.maximum(mx.abs(right_f), 1e-9)
        max_rel = float(mx.max(abs_delta / denom).item())
    else:
        max_abs = mean_abs = rms_abs = max_rel = 0.0
    cos = 1.0 if exact else _cosine(left_f, right_f)
    nonfinite_left = int(mx.sum(mx.logical_or(mx.isnan(left_f), mx.isinf(left_f)).astype(mx.int32)).item())
    nonfinite_right = int(mx.sum(mx.logical_or(mx.isnan(right_f), mx.isinf(right_f)).astype(mx.int32)).item())

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
    left: object,
    right: object,
    *,
    key_width: int,
    wrap_width: int | None,
    colors: Colors,
) -> CompareResult:
    left_is_array = isinstance(left, mx.array)
    right_is_array = isinstance(right, mx.array)
    if left_is_array and right_is_array:
        return _compare_array(name, left, right, key_width=key_width, colors=colors)
    if left_is_array != right_is_array:
        print(
            f"{name:<{key_width}} "
            f"{_exact_status(False, colors=colors, comparable=False)} "
            f"{colors.red('type_mismatch')} {_pair_descriptor(left, right)}"
        )
        return CompareResult(name, exact=False, comparable=False)

    exact = left == right
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


def _load(path: Path) -> dict[str, object]:
    """Load a sidecar through the centralized loader and merge its native MLX
    arrays with its string metadata into one comparable namespace.

    ``load_sidecar`` already normalizes either on-disk format to native dtypes
    (bf16 restored) plus a string metadata dict (pipeline tag, ``*_mlx_dtype``,
    prompts), so the comparison operates on MLX arrays and strings directly -
    no NumPy, no re-widening.
    """
    arrays, metadata = load_sidecar(str(path))
    merged: dict[str, object] = dict(arrays)
    merged.update(metadata)
    return merged


def _selected_keys(
    left_path: Path,
    right_path: Path,
    *,
    keys: list[str] | None,
    include_missing: bool,
) -> list[str]:
    left = _load(left_path)
    right = _load(right_path)
    return _resolve_keys(left.keys(), right.keys(), keys, include_missing)


def compare_sidecar(
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

    left = _load(left_path)
    right = _load(right_path)
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
    parser.add_argument("left", type=Path, help="Left sidecar (safetensors or npz).")
    parser.add_argument("right", type=Path, help="Right sidecar (safetensors or npz).")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Specific keys to compare. Defaults to all common keys.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Report keys present in only one file. Defaults to common keys only.",
    )
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="Also compare adjacent *_text.* sidecars.",
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
        _selected_keys(
            args.left,
            args.right,
            keys=args.keys,
            include_missing=args.include_missing,
        )
    ]
    if args.with_text:
        key_groups.append(
            _selected_keys(
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
        compare_sidecar(
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
            compare_sidecar(
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
