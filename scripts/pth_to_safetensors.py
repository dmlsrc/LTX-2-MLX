#!/usr/bin/env python3
"""Convert a PyTorch .pth/.pt checkpoint to safetensors, directly MLX-loadable.

Safe by construction - nothing in the checkpoint is executed:

1. A static pickle scan (pure pickletools, no unpickling) lists every global the
   file would invoke and refuses anything outside the tensor-rebuild / container
   allowlist (no os/subprocess/eval/exec/import machinery).
2. torch.load(weights_only=True) loads it with PyTorch's restricted unpickler,
   which only runs that same allowlist of tensor rebuilders - it does NOT execute
   the pickle's __reduce__, so a malicious checkpoint raises instead of running.

Then it makes the weights MLX-friendly: strips DataParallel 'module.' prefixes,
demotes float64 -> float32, drops non-tensor entries, and verifies the output
loads with mlx.core.load().

    scripts/pth_to_safetensors.py model.pth                  # -> model.safetensors
    scripts/pth_to_safetensors.py model.pth -o weights.safetensors
    scripts/pth_to_safetensors.py ckpt.pth --strip-prefix "" --keep-fp64
    scripts/pth_to_safetensors.py ckpt.pth --only-prefix generator_ema. --strip-prefix generator_ema.
"""

from __future__ import annotations

import argparse
import pickletools
import sys
from pathlib import Path

# Pickle opcodes that push the module/name strings a STACK_GLOBAL then consumes.
_STR_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING",
}
# Modules a plain weights checkpoint legitimately references.
_SAFE_MODULES = {"torch", "collections", "numpy"}
# Substrings that mark a global as code-execution capable.
_DANGER = (
    "subprocess", "posix", "eval", "exec", "system", "popen", "__import__",
    "importlib", "runpy", "socket", "shutil", "compile", "getattr", "pty",
    "builtins", "os.", "commands", "webbrowser",
)


def _pickle_globals(data: bytes) -> set[str]:
    """Collect every GLOBAL / STACK_GLOBAL the (possibly concatenated) pickles
    reference, without unpickling. The legacy torch format is several pickles
    back to back followed by raw storage bytes, so walk pickle-by-pickle and
    stop when genops hits the non-pickle tail."""
    refs: set[str] = set()
    off, npk = 0, 0
    while off < len(data) and npk < 16:
        strs: list = []
        stop = None
        try:
            for op, arg, _pos in pickletools.genops(data[off:]):
                if op.name in _STR_OPS:
                    strs.append(arg)
                elif op.name == "GLOBAL":
                    refs.add(str(arg).replace("\n", " "))
                elif op.name == "STACK_GLOBAL" and len(strs) >= 2:
                    refs.add(f"{strs[-2]} {strs[-1]}")
                if op.name == "STOP":
                    stop = _pos
                    break
        except Exception:
            break
        if stop is None:
            break
        off += stop + 1
        npk += 1
    return refs


def _suspicious(refs: set[str]) -> list[str]:
    bad = []
    for r in refs:
        rl = r.lower()
        module = r.split()[0] if " " in r else r          # "torch._utils name" -> "torch._utils"
        top = module.split(".")[0]                          # -> "torch"
        if any(d in rl for d in _DANGER) or top not in _SAFE_MODULES:
            bad.append(r)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="Path to the .pth / .pt checkpoint.")
    ap.add_argument("-o", "--output", help="Output .safetensors (default: input with .safetensors).")
    ap.add_argument(
        "--strip-prefix", default="module.",
        help="Key prefix to strip from every weight (default 'module.'; '' to keep).",
    )
    ap.add_argument(
        "--only-prefix", default="",
        help="Keep only tensor keys with this prefix before applying --strip-prefix.",
    )
    ap.add_argument("--keep-fp64", action="store_true", help="Keep float64 tensors as-is (default: demote to float32).")
    ap.add_argument("--force", action="store_true", help="Convert even if the static scan flags non-tensor globals.")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        ap.error(f"no such file: {src}")
    out = Path(args.output) if args.output else src.with_suffix(".safetensors")

    # ---- 1. static safety scan (no execution) ------------------------------
    refs = _pickle_globals(src.read_bytes())
    print(f"[scan] pickle references {len(refs)} global(s):")
    for r in sorted(refs):
        print(f"        {r}")
    bad = _suspicious(refs)
    if bad:
        print(f"[scan] SUSPICIOUS (outside tensor-rebuild allowlist): {bad}", file=sys.stderr)
        if not args.force:
            print("[scan] refusing to load. Re-run with --force only if you trust this file "
                  "(weights_only=True still gates the load).", file=sys.stderr)
            return 2
    else:
        print("[scan] clean - only tensor-rebuild / container globals.")

    # ---- 2. safe load (restricted unpickler, no code execution) ------------
    try:
        import torch
    except ImportError:
        print("error: this converter needs PyTorch (a dev dependency) to read .pth files.", file=sys.stderr)
        return 1
    try:
        obj = torch.load(str(src), map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"error: torch.load(weights_only=True) refused/failed: {e}", file=sys.stderr)
        print("       The checkpoint contains non-tensor objects (optimizer state, custom "
              "classes, ...) the safe loader won't run. Extract just the weights first.", file=sys.stderr)
        return 1

    # ---- 3. find the state_dict (handle common nesting) --------------------
    sd = obj
    if not (hasattr(sd, "items") and any(torch.is_tensor(v) for v in sd.values())):
        for key in ("state_dict", "model", "net", "weights", "params_ema", "params"):
            if isinstance(obj, dict) and key in obj and hasattr(obj[key], "items"):
                sd = obj[key]
                print(f"[load] using nested checkpoint key '{key}'")
                break

    # ---- 4. make MLX-friendly: strip prefix, demote fp64, drop non-tensors -
    prefix = args.strip_prefix
    only_prefix = args.only_prefix
    tensors, dropped, stripped, filtered = {}, [], 0, 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            dropped.append(k)
            continue
        if only_prefix and not k.startswith(only_prefix):
            filtered += 1
            continue
        nk = k[len(prefix):] if prefix and k.startswith(prefix) else k
        stripped += (nk != k)
        t = v.detach().cpu().contiguous()
        if t.dtype == torch.float64 and not args.keep_fp64:
            t = t.float()
        tensors[nk] = t.clone()
    if not tensors:
        print("error: no tensors found in the checkpoint.", file=sys.stderr)
        return 1
    n_params = sum(t.numel() for t in tensors.values())
    print(f"[convert] {len(tensors)} tensors, {n_params/1e6:.3f}M params, "
          f"dtypes={sorted({str(t.dtype) for t in tensors.values()})}")
    if stripped:
        print(f"[convert] stripped '{prefix}' from {stripped} keys")
    if filtered:
        print(f"[convert] filtered out {filtered} tensor keys outside '{only_prefix}'")
    if dropped:
        print(f"[convert] dropped {len(dropped)} non-tensor entries: {dropped[:6]}"
              f"{'...' if len(dropped) > 6 else ''}")

    # ---- 5. save + verify it loads in MLX ----------------------------------
    from safetensors.torch import save_file
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    try:
        import mlx.core as mx
        loaded = mx.load(str(out))
        ok = len(loaded) == len(tensors)
        sample = next(iter(loaded.items()))
        print(f"[verify] mlx.core.load OK: {len(loaded)} arrays "
              f"(e.g. {sample[0]} {tuple(sample[1].shape)} {sample[1].dtype}); match={ok}")
    except ImportError:
        print("[verify] (mlx not importable here; safetensors written, but MLX load unverified)")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
