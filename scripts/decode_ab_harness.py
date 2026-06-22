#!/usr/bin/env python3
"""A/B harness: accumulating decode_latent vs streamed decode_streaming (matched chunks).

The streaming decode is being decoupled from tiling: the no-tiling path now streams
through ``decode_streaming`` with a temporal config matched to ``decode_latent``'s own
chunking (7 latent frames / 2 overlap == 56 / 16 in pixel-space units, given the 8x
temporal scale). The two outputs WILL differ at chunk boundaries (different blend
code); this harness answers the only question that matters: is the streamed output
*worse*, i.e. does it introduce the artifacts the chunking exists to prevent?

It does NOT do a pixel-equality check. It looks for the two named failure modes:

  1. Conv3d early-frame noise -- the bug decode_latent's small chunking suppresses.
     Surfaces as the A/B difference concentrating in the first frames of each chunk.
  2. Temporal seams -- a blend discontinuity at chunk boundaries. Surfaces as a
     spike in the streamed output's frame-to-frame gradient that the accumulated
     output doesn't have.

`decode_latent` (the accumulating baseline) is treated as ground truth: the question
is purely whether the stream regresses against what ships today. Note decode_streaming is
already the production decoder for every tiled (large) clip, so its blend is already
validated; this just extends it to the no-tiling case.

Produce a latent first by saving a sidecar from a real run, then::

    python scripts/decode_ab_harness.py \
        --latent  /path/to/run.sidecar.safetensors \
        --weights "$LTX_DEFAULT_WEIGHTS_PATH" \
        --out     "$SHARED_TEMP_DIR/decode_ab"

Outputs: a summary.json + a printed verdict, and stacked [A | B | 5x|A-B|] PNG crops
of the worst-diff frame, the first frames of the clip, and any detected seam frames.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

# Run as `python scripts/decode_ab_harness.py`: put the repo root and scripts/ on
# the path so we can reuse decode_latent_debug.make_decoder (the split-checkpoint
# VAE loader: config from the combined checkpoint, weights from the split VAE)
# instead of reinventing the loading.
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS.parent), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decode_latent_debug import make_decoder  # noqa: E402

from LTX_2_MLX import sidecars  # noqa: E402
from LTX_2_MLX.model.video_vae.decode_utils import decode_latent  # noqa: E402
from LTX_2_MLX.model.video_vae.tiling import (  # noqa: E402
    TemporalChunkConfig,
    TilingConfig,
    decode_streaming,
)

# Matches decode_latent(temporal_chunk_size=7, temporal_overlap=2); tile sizes are
# pixel-space, multiplied through the 8x temporal VAE scale.
_MATCHED_TILE = TemporalChunkConfig(chunk_size_in_frames=56, chunk_overlap_in_frames=16)


def _to_thwc_unit(video: mx.array) -> mx.array:
    """(B,C,T,H,W) in [-1,1] -> (T,H,W,C) in [0,1], batch 0."""
    v = mx.clip((video + 1.0) * 0.5, 0.0, 1.0)
    return mx.transpose(v[0], (1, 2, 3, 0))


def _save_stack(a_t: mx.array, b_t: mx.array, diff_t: mx.array, path: Path) -> None:
    """Save a horizontal [A | B | 5x|diff|] strip; PNG if the saver is available."""
    strip = mx.concatenate([a_t, b_t, mx.clip(diff_t * 5.0, 0.0, 1.0)], axis=1)
    u8 = (strip * 255.0 + 0.5).astype(mx.uint8)
    mx.eval(u8)
    try:
        from LTX_2_MLX.videotoolbox.images import save_image

        save_image(u8, str(path.with_suffix(".png")))
    except Exception:  # noqa: BLE001 - fall back to a raw dump the user can view
        mx.save_safetensors(str(path.with_suffix(".safetensors")), {"strip_rgb_u8": u8})


def compare_decodes(
    latent: mx.array,
    decoder: Any,
    *,
    out_dir: Path,
    timestep: float = 0.05,
    n_early: int = 4,
) -> dict[str, Any]:
    """Decode `latent` both ways, score the difference, write crops + summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # A: current accumulating path. B: streamed, chunk-matched.
    video_a = decode_latent(latent, decoder, timestep=timestep, dtype=mx.float32)
    cfg = TilingConfig(spatial_config=None, temporal_config=_MATCHED_TILE)
    chunks = list(decode_streaming(latent, decoder, cfg, timestep=timestep, show_progress=False))
    video_b = mx.concatenate(chunks, axis=2)

    a = _to_thwc_unit(video_a)
    b = _to_thwc_unit(video_b)
    t = min(a.shape[0], b.shape[0])
    a, b = a[:t], b[:t]

    diff = mx.abs(a - b)
    diff_pf = mx.mean(diff, axis=(1, 2, 3))            # per-frame mean |A-B|
    grad_a = mx.mean(mx.abs(a[1:] - a[:-1]), axis=(1, 2, 3))
    grad_b = mx.mean(mx.abs(b[1:] - b[:-1]), axis=(1, 2, 3))
    mx.eval(diff_pf, grad_a, grad_b)

    dpf, ga, gb = diff_pf.tolist(), grad_a.tolist(), grad_b.tolist()

    # Seam detection: spikes in B's temporal gradient that A doesn't share. A frame i
    # in the gradient arrays is the step from frame i to i+1.
    med = statistics.median(gb)
    mad = statistics.median([abs(x - med) for x in gb]) or 1e-9
    seams = [
        {"frame": i + 1, "grad_b": gb[i], "grad_a": ga[i], "b_worse": gb[i] > ga[i] * 1.5}
        for i in range(len(gb))
        if gb[i] > med + 6.0 * mad
    ]
    seam_regressions = [s for s in seams if s["b_worse"]]

    max_diff_frame = max(range(t), key=lambda i: dpf[i]) if t else 0

    # Crops: worst-diff frame, the first n_early frames, and frames around seams.
    crop_frames = sorted({max_diff_frame, *range(min(n_early, t)), *(s["frame"] for s in seams)})
    for f in crop_frames:
        _save_stack(a[f], b[f], diff[f], out_dir / f"frame_{f:04d}")

    summary = {
        "frames_compared": t,
        "frame_count_a": int(a.shape[0]) if t == 0 else int(video_a.shape[2]),
        "frame_count_b": int(video_b.shape[2]),
        "mean_abs_diff": sum(dpf) / len(dpf) if dpf else 0.0,
        "max_abs_diff_per_frame": max(dpf) if dpf else 0.0,
        "max_diff_frame": max_diff_frame,
        "max_grad_a": max(ga) if ga else 0.0,
        "max_grad_b": max(gb) if gb else 0.0,
        "seam_candidates": seams,
        "seam_regressions": seam_regressions,
        "crops_written": [str(out_dir / f"frame_{f:04d}") for f in crop_frames],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _load_decoder(weights_path: str, config_weights_path: str | None) -> Any:
    # Reuse the existing split-checkpoint loader: it reads decoder_blocks from
    # config_weights_path (the combined checkpoint) and the weights from
    # weights_path (the split video_vae), and defaults timestep_conditioning off.
    return make_decoder(weights_path, mx.bfloat16, config_weights_path=config_weights_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent", required=True, help="sidecar .safetensors holding 'final_video_latent'")
    ap.add_argument("--weights", required=True, help="VAE weights (the split video_vae.safetensors)")
    ap.add_argument("--config-weights", default=os.environ.get("LTX_DEFAULT_WEIGHTS_PATH"),
                    help="checkpoint carrying the VAE config/decoder_blocks (the combined checkpoint)")
    ap.add_argument("--out", required=True, help="output directory for crops + summary.json")
    ap.add_argument("--timestep", type=float, default=0.05)
    ap.add_argument("--latent-key", default="final_video_latent")
    ap.add_argument("--crop-t", type=int, default=None,
                    help="keep only the first N latent frames (cheaper run; needs >7 to multi-chunk)")
    ap.add_argument("--crop-w", type=int, default=None,
                    help="keep only the first N latent W columns; <=16 keeps pixel width <=512 so the "
                         "no-tiling arm avoids the spatial conv3d overflow")
    args = ap.parse_args()

    arrays, _meta = sidecars.load_sidecar(args.latent)
    if args.latent_key not in arrays:
        ap.error(f"latent key {args.latent_key!r} not in sidecar; keys: {sorted(arrays)}")
    latent = arrays[args.latent_key]
    if args.crop_t:
        latent = latent[:, :, : args.crop_t]
    if args.crop_w:
        latent = latent[:, :, :, :, : args.crop_w]
    print(f"latent {tuple(latent.shape)} (pixel ~ {latent.shape[-2] * 32} x {latent.shape[-1] * 32})")

    decoder = _load_decoder(args.weights, args.config_weights)
    summary = compare_decodes(latent, decoder, out_dir=Path(args.out), timestep=args.timestep)

    print(f"\nframes: A={summary['frame_count_a']} B={summary['frame_count_b']} compared={summary['frames_compared']}")
    print(f"mean |A-B| = {summary['mean_abs_diff']:.5f}   max per-frame = {summary['max_abs_diff_per_frame']:.5f} (frame {summary['max_diff_frame']})")
    print(f"max temporal gradient: A={summary['max_grad_a']:.5f}  B={summary['max_grad_b']:.5f}")
    if summary["seam_regressions"]:
        print(f"VERDICT: possible seam regression at frames {[s['frame'] for s in summary['seam_regressions']]} "
              "(B gradient spike >> A). Inspect those crops.")
    elif summary["seam_candidates"]:
        print("VERDICT: seams detected but not worse than A (B gradient spike <= A). Likely fine.")
    else:
        print("VERDICT: no seam spikes; differences are blend/numerical. Inspect the max-diff + early-chunk crops to confirm no noise.")
    print(f"crops + summary.json in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
