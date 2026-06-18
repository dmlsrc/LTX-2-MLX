#!/usr/bin/env python3
"""Inspect edge/boundary artifacts in a rendered video.

This is a lightweight visual diagnostic for the VAE boundary artifacts that are
most visible near frame edges. It extracts representative frames, writes contact
sheets for full frames and edge strips, and records simple edge-vs-center
statistics so future runs can be compared without redoing scratch work.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from LTX_2_MLX.videotoolbox.images import draw_labels, load_image_rgb, resize_lanczos, save_image

np = None


def load_image_deps() -> None:
    global np
    if np is not None:
        return

    import numpy as _np

    np = _np


def run_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(video: Path) -> tuple[int | None, float | None, int | None, int | None]:
    data = json.loads(
        run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-print_format",
                "json",
                str(video),
            ]
        )
    )
    stream = data["streams"][0]
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    width = int(stream["width"]) if "width" in stream else None
    height = int(stream["height"]) if "height" in stream else None

    frames = stream.get("nb_frames")
    if frames is not None and frames != "N/A":
        return int(frames), fps, width, height

    duration = stream.get("duration")
    if duration is not None and fps:
        return max(1, int(round(float(duration) * fps))), fps, width, height

    return None, fps, width, height


def choose_frame_indices(frame_count: int | None, num_samples: int, frames_arg: str | None) -> list[int]:
    if frames_arg:
        indices = sorted({int(part.strip()) for part in frames_arg.split(",") if part.strip()})
        if frame_count is not None:
            indices = [idx for idx in indices if 0 <= idx < frame_count]
        if not indices:
            raise ValueError("--frames did not leave any valid frame indices")
        return indices

    if frame_count is None:
        return [0]

    if num_samples <= 1:
        return [frame_count // 2]

    return sorted(
        {
            min(frame_count - 1, max(0, int(round(i * (frame_count - 1) / (num_samples - 1)))))
            for i in range(num_samples)
        }
    )


def extract_frame(video: Path, frame_index: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            str(output_path),
        ]
    )


def load_image(path: Path) -> np.ndarray:
    return np.asarray(load_image_rgb(path))


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    new_h = max(1, round(image.shape[0] * width / image.shape[1]))
    return np.asarray(resize_lanczos(image, width, new_h))


def draw_sheet(cells: list[tuple[str, np.ndarray]], output_path: Path, columns: int, cell_width: int) -> None:
    if not cells:
        return

    label_h = 18
    padding = 8
    images = [(label, resize_to_width(image, cell_width)) for label, image in cells]
    cell_h = max(image.shape[0] for _, image in images) + label_h
    rows = math.ceil(len(images) / columns)
    sheet_w = columns * cell_width + (columns + 1) * padding
    sheet_h = rows * cell_h + (rows + 1) * padding
    sheet = np.full((sheet_h, sheet_w, 3), (24, 24, 24), dtype=np.uint8)

    labels: list[tuple[int, int, str]] = []
    for idx, (label, image) in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_h + padding)
        labels.append((x, y, label))
        h, w = image.shape[0], image.shape[1]
        sheet[y + label_h:y + label_h + h, x:x + w] = image

    sheet = np.asarray(draw_labels(sheet, labels, color=(235, 235, 235)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(sheet, output_path)


def edge_strip(image: np.ndarray, edge_pct: float, side_pct: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[0], image.shape[1]
    top_bottom_px = max(1, int(round(height * edge_pct / 100.0)))
    side_px = max(1, int(round(width * side_pct / 100.0)))

    top = image[0:top_bottom_px, 0:width]
    bottom = image[height - top_bottom_px:height, 0:width]
    left = image[0:height, 0:side_px]
    right = image[0:height, width - side_px:width]

    horizontal = np.full((top.shape[0] + bottom.shape[0] + 2, width, 3), (255, 0, 255), dtype=np.uint8)
    horizontal[0:top.shape[0], 0:width] = top
    horizontal[top.shape[0] + 2:top.shape[0] + 2 + bottom.shape[0], 0:width] = bottom

    vertical = np.full((height, left.shape[1] + right.shape[1] + 2, 3), (255, 0, 255), dtype=np.uint8)
    vertical[0:height, 0:left.shape[1]] = left
    vertical[0:height, left.shape[1] + 2:left.shape[1] + 2 + right.shape[1]] = right
    return horizontal, vertical


def band_metrics(frames: list[np.ndarray], edge_pct: float, side_pct: float) -> list[str]:
    if not frames:
        return []

    height, width = frames[0].shape[:2]
    edge_px = max(1, int(round(height * edge_pct / 100.0)))
    side_px = max(1, int(round(width * side_pct / 100.0)))
    center_y0 = height // 2 - edge_px // 2
    center_y1 = center_y0 + edge_px
    center_x0 = width // 2 - side_px // 2
    center_x1 = center_x0 + side_px

    lines = [
        f"edge_pct={edge_pct:g} top/bottom band={edge_px}px of {height}px ({edge_px / height * 100:.2f}%)",
        f"side_pct={side_pct:g} left/right band={side_px}px of {width}px ({side_px / width * 100:.2f}%)",
    ]

    def region_stats(name: str, values: list[np.ndarray]) -> None:
        stacked = np.concatenate([value.reshape(-1, 3) for value in values], axis=0).astype(np.float32)
        lines.append(f"{name:<14} mean={stacked.mean():7.2f} std={stacked.std():7.2f}")

    region_stats("top", [frame[:edge_px] for frame in frames])
    region_stats("bottom", [frame[-edge_px:] for frame in frames])
    region_stats("left", [frame[:, :side_px] for frame in frames])
    region_stats("right", [frame[:, -side_px:] for frame in frames])
    region_stats("center_y", [frame[center_y0:center_y1] for frame in frames])
    region_stats("center_x", [frame[:, center_x0:center_x1] for frame in frames])

    if len(frames) > 1:
        deltas = [np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)) for i in range(1, len(frames))]
        lines.append("sampled-frame mean absolute change:")
        for name, slices in (
            ("top", (slice(None, edge_px), slice(None))),
            ("bottom", (slice(-edge_px, None), slice(None))),
            ("left", (slice(None), slice(None, side_px))),
            ("right", (slice(None), slice(-side_px, None))),
            ("center", (slice(center_y0, center_y1), slice(center_x0, center_x1))),
        ):
            value = np.mean([delta[slices[0], slices[1]].mean() for delta in deltas])
            lines.append(f"  {name:<8} {value:7.3f}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--frames", help="Comma-separated frame indices to inspect.")
    parser.add_argument("--edge-pct", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    parser.add_argument("--side-pct", type=float, default=5.0)
    parser.add_argument("--sheet-width", type=int, default=320)
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args()
    load_image_deps()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")

    frame_count, fps, width, height = probe_video(args.video)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.video.with_suffix("")
        output_dir = output_dir.parent / f"{output_dir.name}_edge_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = choose_frame_indices(frame_count, args.num_samples, args.frames)
    extracted_dir = output_dir / "extracted_frames"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for frame_index in indices:
        path = extracted_dir / f"frame_{frame_index:05d}.png"
        extract_frame(args.video, frame_index, path)
        frame_paths.append(path)

    images = [load_image(path) for path in frame_paths]
    arrays = [np.array(image) for image in images]

    draw_sheet(
        [(f"f{idx}", image) for idx, image in zip(indices, images, strict=True)],
        output_dir / "full_frames.png",
        columns=min(4, len(images)),
        cell_width=args.sheet_width,
    )

    for pct in args.edge_pct:
        h_cells = []
        v_cells = []
        for frame_index, image in zip(indices, images, strict=True):
            horizontal, vertical = edge_strip(image, pct, args.side_pct)
            h_cells.append((f"f{frame_index} top+bottom {pct:g}%", horizontal))
            v_cells.append((f"f{frame_index} left+right {args.side_pct:g}%", vertical))
        draw_sheet(
            h_cells,
            output_dir / f"top_bottom_{pct:g}pct.png",
            columns=min(3, len(h_cells)),
            cell_width=args.sheet_width,
        )
        draw_sheet(
            v_cells,
            output_dir / f"left_right_{args.side_pct:g}pct_for_{pct:g}pct_run.png",
            columns=min(4, len(v_cells)),
            cell_width=max(96, args.sheet_width // 3),
        )

    metrics_lines = [
        f"video={args.video}",
        f"frames={frame_count if frame_count is not None else 'unknown'} fps={fps if fps is not None else 'unknown'}",
        f"size={width if width is not None else '?'}x{height if height is not None else '?'}",
        f"sampled_indices={','.join(str(idx) for idx in indices)}",
        "",
    ]
    for pct in args.edge_pct:
        metrics_lines.extend(band_metrics(arrays, pct, args.side_pct))
        metrics_lines.append("")

    (output_dir / "metrics.txt").write_text("\n".join(metrics_lines), encoding="utf-8")
    print(f"Wrote edge analysis to {output_dir}")

    if not args.keep_extracted:
        shutil.rmtree(extracted_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
