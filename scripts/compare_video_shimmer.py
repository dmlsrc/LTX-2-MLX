#!/usr/bin/env python3
"""Compare two videos by temporal second-difference (TSD) — a frame-to-frame
"shimmer" / non-smoothness metric.

Why TSD?

  The naive frame-to-frame difference |frame[i] - frame[i-1]| picks up both
  real motion and shimmer in one number, so it can't tell them apart.

  TSD = |frame[i] - (frame[i-1] + frame[i+1]) / 2|

  cancels smooth (linear) motion — real motion is approximately linear
  frame-to-frame at small scales, so the midpoint average predicts frame[i]
  well — and leaves only non-smooth temporal variation: shimmer, flicker,
  cuts, encoding artifacts.  Useful for A/B-ing two videos that show the
  same source content but went through different processing (VSR modes,
  encoder settings, temporal upscaling, etc.).

Streams frames through ffmpeg at full source resolution as gray8 luma —
keeps a 3-frame ring buffer in memory, computes per-frame TSD plus an
8x8 spatial grid of per-patch shimmer, never materializes more than ~30 MB
of luma at any time regardless of video resolution.

Usage:
    scripts/compare_video_shimmer.py <video-a> <video-b>
    scripts/compare_video_shimmer.py a.mp4 b.mp4 --label-a image --label-b balanced
    scripts/compare_video_shimmer.py a.mp4 b.mp4 --grid 16  # finer spatial grid
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np


def probe_dimensions(path: str) -> tuple[int, int, float, int]:
    """Get (width, height, fps, total_frames) from a video via ffprobe.

    Returns total_frames=0 if it's not in the metadata (some containers
    don't store it; we'll just count what ffmpeg streams to us).
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "default=noprint_wrappers=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    fields = dict(line.split("=", 1) for line in out if "=" in line)
    w = int(fields["width"])
    h = int(fields["height"])
    num, den = fields.get("r_frame_rate", "24/1").split("/")
    fps = float(num) / float(den) if float(den) != 0 else 24.0
    try:
        nframes = int(fields.get("nb_frames", "0"))
    except ValueError:
        nframes = 0
    return w, h, fps, nframes


def stream_tsd(path: str, w: int, h: int, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream the video as gray8 luma via ffmpeg; return per-frame TSD and
    per-patch (grid x grid) mean TSD.

    Memory: 3 frames * w * h bytes (e.g. 3 * 4096 * 2304 = ~28 MB at 4K).
    """
    frame_bytes = w * h
    patch_h = h // grid
    patch_w = w // grid
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", path,
        "-vf", "format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None

    ring: list[np.ndarray] = []
    tsd_per_frame: list[float] = []
    tsd_per_patch = np.zeros((grid, grid), dtype=np.float64)
    n_pairs = 0

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w).astype(np.int16)
            ring.append(frame)
            if len(ring) > 3:
                ring.pop(0)
            if len(ring) == 3:
                tsd = np.abs(ring[1] - (ring[0] + ring[2]) // 2)
                tsd_per_frame.append(float(tsd.mean()))
                # Average over each grid cell.  Trim trailing pixels that
                # don't fit a full row/column of patches.
                trimmed = tsd[: patch_h * grid, : patch_w * grid]
                patches = trimmed.reshape(grid, patch_h, grid, patch_w).mean(axis=(1, 3))
                tsd_per_patch += patches
                n_pairs += 1
    finally:
        proc.wait()
        if proc.returncode not in (0, None):
            sys.stderr.write(f"ffmpeg exited with {proc.returncode} on {path}\n")

    tsd_per_patch /= max(n_pairs, 1)
    return np.array(tsd_per_frame), tsd_per_patch


def _patch_glyph(value: float) -> str:
    """Compact 2-char glyph for the per-patch ascii heatmap."""
    if value >= 0.10:    return "##"
    if value >= 0.05:    return "++"
    if value >= 0.02:    return "+ "
    if value <= -0.10:   return "@@"
    if value <= -0.05:   return "--"
    if value <= -0.02:   return "- "
    return ". "


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video_a", help="First video to analyze.")
    p.add_argument("video_b", help="Second video to analyze.")
    p.add_argument("--label-a", default=None, help="Label for video A (defaults to filename stem).")
    p.add_argument("--label-b", default=None, help="Label for video B (defaults to filename stem).")
    p.add_argument(
        "--grid", type=int, default=8,
        help="Spatial grid resolution for the per-patch heatmap (default 8 = 8x8 grid).",
    )
    p.add_argument(
        "--per-second", action="store_true",
        help="Also print mean TSD bucketed by output-clip second.",
    )
    args = p.parse_args()

    import os.path
    label_a = args.label_a or os.path.splitext(os.path.basename(args.video_a))[0]
    label_b = args.label_b or os.path.splitext(os.path.basename(args.video_b))[0]

    wa, ha, fps_a, _ = probe_dimensions(args.video_a)
    wb, hb, fps_b, _ = probe_dimensions(args.video_b)
    if (wa, ha) != (wb, hb):
        p.error(
            f"video dimensions differ: {args.video_a} is {wa}x{ha}, "
            f"{args.video_b} is {wb}x{hb}. TSD comparison requires matched resolutions."
        )
    fps = fps_a  # for per-second bucketing
    w, h = wa, ha

    print(f"Resolution:  {w}x{h}")
    print(f"Grid:        {args.grid}x{args.grid}  ({h // args.grid}x{w // args.grid} pixels per patch)")
    print(f"FPS:         {fps:.3f}")
    print()

    print(f"Streaming {label_a}: {args.video_a}")
    tsd_a, patches_a = stream_tsd(args.video_a, w, h, args.grid)
    print(f"  {len(tsd_a)} TSD samples")

    print(f"Streaming {label_b}: {args.video_b}")
    tsd_b, patches_b = stream_tsd(args.video_b, w, h, args.grid)
    print(f"  {len(tsd_b)} TSD samples")

    n = min(len(tsd_a), len(tsd_b))
    if n < len(tsd_a) or n < len(tsd_b):
        print(f"  (truncating both series to {n} frame pairs for comparison)")
    tsd_a = tsd_a[:n]
    tsd_b = tsd_b[:n]
    diff = tsd_a - tsd_b

    print()
    print("=== Temporal second-difference (TSD) per frame ===")
    print(f"{'video':<20s}{'mean':>10s}{'median':>10s}{'std':>10s}{'p95':>10s}{'max':>10s}")
    print(
        f"{label_a:<20s}"
        f"{tsd_a.mean():>10.4f}{np.median(tsd_a):>10.4f}{tsd_a.std():>10.4f}"
        f"{np.percentile(tsd_a, 95):>10.4f}{tsd_a.max():>10.4f}"
    )
    print(
        f"{label_b:<20s}"
        f"{tsd_b.mean():>10.4f}{np.median(tsd_b):>10.4f}{tsd_b.std():>10.4f}"
        f"{np.percentile(tsd_b, 95):>10.4f}{tsd_b.max():>10.4f}"
    )

    print()
    print(f"=== Shimmer signal: TSD({label_a}) − TSD({label_b}) ===")
    print(f"  mean:    {diff.mean():+.4f}")
    print(f"  median:  {np.median(diff):+.4f}")
    print(f"  p5:      {np.percentile(diff, 5):+.4f}")
    print(f"  p95:     {np.percentile(diff, 95):+.4f}")
    print(f"  max:     {diff.max():+.4f}  (frame pair {diff.argmax() + 1})")
    print(f"  min:     {diff.min():+.4f}  (frame pair {diff.argmin() + 1})")
    a_higher = (diff > 0).mean() * 100
    print(f"  fraction of frame pairs where {label_a} > {label_b}: {a_higher:.1f}%")

    if args.per_second:
        print()
        print("=== Per-second mean TSD ===")
        sec_buckets = np.arange(n) // int(round(fps))
        max_sec = int(sec_buckets.max()) + 1
        print(f"  {'sec':>4s}  {label_a:>16s}  {label_b:>16s}  {'diff':>8s}")
        for s in range(max_sec):
            mask = sec_buckets == s
            if not mask.any():
                continue
            sa, sb = tsd_a[mask].mean(), tsd_b[mask].mean()
            print(f"  {s:>4d}  {sa:>16.4f}  {sb:>16.4f}  {sa - sb:>+8.4f}")

    diff_patches = patches_a - patches_b
    print()
    print(f"=== Per-patch shimmer ({label_a} − {label_b}, {args.grid}x{args.grid} grid) ===")
    print("  (rows top→bottom, cols left→right)")
    for r in range(args.grid):
        row = "  "
        for c in range(args.grid):
            row += _patch_glyph(diff_patches[r, c]) + " "
        print(row)
    print(
        "  legend: ##≥0.10  ++≥0.05  + ≥0.02  . ≈0  "
        "- ≤−0.02  --≤−0.05  @@≤−0.10"
    )
    mx_idx = np.unravel_index(int(diff_patches.argmax()), diff_patches.shape)
    mn_idx = np.unravel_index(int(diff_patches.argmin()), diff_patches.shape)
    print(f"  max:  {diff_patches.max():+.4f} at row {mx_idx[0]}, col {mx_idx[1]}")
    print(f"  min:  {diff_patches.min():+.4f} at row {mn_idx[0]}, col {mn_idx[1]}")


if __name__ == "__main__":
    main()
