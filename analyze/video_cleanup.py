# Uses opencv and ffmpeg to remove letterboxing, overlays and other artifacts

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CropRect:
    """Crop rectangle in source pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    def as_ffmpeg_crop(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"


@dataclass(frozen=True)
class OverlayRect:
    """Static overlay region to remove (best-effort)."""

    x: int
    y: int
    w: int
    h: int

    def as_ffmpeg_delogo(self) -> str:
        # https://ffmpeg.org/ffmpeg-filters.html#delogo
        return f"delogo=x={self.x}:y={self.y}:w={self.w}:h={self.h}:show=0"


class VideoCleanupError(RuntimeError):
    pass


def _pick_sample_frame_indices(
    frame_count: int,
    sample_frames: int,
    *,
    start_fraction: float = 0.05,
    end_fraction: float = 0.95,
) -> list[int]:
    if frame_count <= 0:
        return []
    if sample_frames <= 1:
        return [max(0, min(frame_count - 1, int(frame_count * 0.5)))]

    start = int(frame_count * start_fraction)
    end = max(start + 1, int(frame_count * end_fraction))
    end = min(end, frame_count - 1)
    if end <= start:
        return [start]

    return np.linspace(start, end, num=sample_frames, dtype=int).tolist()


def _detect_letterbox_crop(
    video_path: str | os.PathLike[str],
    *,
    sample_frames: int = 20,
    black_threshold: int = 16,
    nonblack_row_fraction: float = 0.015,
    min_crop_px: int = 2,
) -> CropRect | None:
    """
    Detect black bars (letterboxing/pillarboxing) and return a crop rect.

    Heuristic:
    - Sample frames through the video
    - For each frame, compute per-row and per-column mean luma
    - Consider a row/col "content" if mean luma > black_threshold
    - Find first/last content rows/cols; take a robust median across frames
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoCleanupError(f"Failed to open video with OpenCV: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise VideoCleanupError("Could not determine video dimensions via OpenCV.")

        indices = _pick_sample_frame_indices(frame_count, sample_frames)
        if not indices:
            # Some containers don't expose CAP_PROP_FRAME_COUNT reliably; fall back to sequential sampling.
            indices = list(range(0, sample_frames))

        tops: list[int] = []
        bottoms: list[int] = []
        lefts: list[int] = []
        rights: list[int] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row_mean = gray.mean(axis=1)  # (H,)
            col_mean = gray.mean(axis=0)  # (W,)

            row_content = row_mean > black_threshold
            col_content = col_mean > black_threshold

            # Require a minimum amount of "content" to avoid treating mostly-black frames as signal.
            if (row_content.sum() / height) < nonblack_row_fraction:
                continue
            if (col_content.sum() / width) < nonblack_row_fraction:
                continue

            top = int(np.argmax(row_content))
            bottom = int(height - 1 - np.argmax(row_content[::-1]))
            left = int(np.argmax(col_content))
            right = int(width - 1 - np.argmax(col_content[::-1]))

            if bottom <= top or right <= left:
                continue

            tops.append(top)
            bottoms.append(bottom)
            lefts.append(left)
            rights.append(right)

        if not tops:
            return None

        top = int(np.median(tops))
        bottom = int(np.median(bottoms))
        left = int(np.median(lefts))
        right = int(np.median(rights))

        # Convert inclusive bounds to x,y,w,h
        x = max(0, left)
        y = max(0, top)
        w = min(width, right + 1) - x
        h = min(height, bottom + 1) - y

        if w <= 0 or h <= 0:
            return None

        # Ignore tiny crops (noise)
        if x < min_crop_px and y < min_crop_px and (width - (x + w)) < min_crop_px and (height - (y + h)) < min_crop_px:
            return None

        # Ensure even dimensions for common codecs (H.264), keeping crop centered on detected rect.
        w_even = w - (w % 2)
        h_even = h - (h % 2)
        if w_even <= 0 or h_even <= 0:
            return None
        if w_even != w:
            x = min(x, width - w_even)
            w = w_even
        if h_even != h:
            y = min(y, height - h_even)
            h = h_even

        return CropRect(x=x, y=y, w=w, h=h)
    finally:
        cap.release()


def _build_filter_chain(crop: CropRect | None, overlays: Sequence[OverlayRect]) -> str | None:
    filters: list[str] = []
    if crop is not None:
        filters.append(crop.as_ffmpeg_crop())
    for o in overlays:
        filters.append(o.as_ffmpeg_delogo())
    if not filters:
        return None
    return ",".join(filters)


def _run_ffmpeg(args: Sequence[str]) -> None:
    try:
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise VideoCleanupError("ffmpeg not found on PATH. Please install ffmpeg.") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        raise VideoCleanupError(f"ffmpeg failed:\n{stderr}") from e


def cleanup_video(
    video_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    overlays: Sequence[OverlayRect] = (),
    sample_frames: int = 20,
    suffix: str = "_cleaned",
) -> str:
    """
    Clean up a video and return the output file path.

    - **As a module**: call with `output_path=None` to write to a temporary file and return it.
    - **As a CLI**: call with an explicit `output_path` (the CLI does this by default).

    Current best-effort behaviors:
    - Detect letterboxing/pillarboxing and crop it away.
    - Optionally remove static overlays using ffmpeg's `delogo` filter.
    """
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video file not found: {src}")
    if not src.is_file():
        raise FileNotFoundError(f"Not a file: {src}")

    temp_output = False
    if output_path is None:
        # Use a deterministic suffix and a common container.
        # We keep the extension when possible; fallback to mp4 for broader compatibility.
        ext = src.suffix if src.suffix else ".mp4"
        fd, tmp_path = tempfile.mkstemp(prefix=f"{src.stem}{suffix}_", suffix=ext)
        os.close(fd)
        out = Path(tmp_path)
        temp_output = True
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)

    crop = _detect_letterbox_crop(src, sample_frames=sample_frames)
    vf = _build_filter_chain(crop, overlays)

    base_args: list[str] = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
    ]
    if vf is not None:
        base_args += ["-vf", vf]

    # First try: copy audio (fast, preserves original).
    args_copy_audio = [
        *base_args,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(out),
    ]

    try:
        try:
            _run_ffmpeg(args_copy_audio)
        except VideoCleanupError:
            # Fallback: re-encode audio to aac for compatibility.
            args_aac = [
                *base_args,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out),
            ]
            _run_ffmpeg(args_aac)
    except Exception:
        if temp_output:
            try:
                out.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    return str(out)


# Backwards-compatible entry point expected by the rest of the project.
def analyze(video_path: str) -> str:
    return cleanup_video(video_path)


def _parse_overlay_rects(values: Iterable[str]) -> list[OverlayRect]:
    rects: list[OverlayRect] = []
    for v in values:
        parts = v.split(",")
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                f"Overlay must be 'x,y,w,h' but got: {v!r}"
            )
        try:
            x, y, w, h = (int(p) for p in parts)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Overlay must be integers 'x,y,w,h' but got: {v!r}"
            ) from e
        rects.append(OverlayRect(x=x, y=y, w=w, h=h))
    return rects


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean up a video (crop letterboxing, remove overlays).")
    parser.add_argument("video_path", help="Path to input video.")
    parser.add_argument(
        "--suffix",
        default="_cleaned",
        help="Suffix for CLI output filename (default: %(default)s).",
    )
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Static overlay region to remove via delogo: 'x,y,w,h' (repeatable).",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=20,
        help="Number of frames to sample for letterbox detection (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    src = Path(args.video_path)
    if not src.exists():
        print(f"Video file not found: {src}", file=sys.stderr)
        return 1

    overlays = _parse_overlay_rects(args.overlay)

    out_name = f"{src.stem}{args.suffix}{src.suffix or '.mp4'}"
    out_path = Path.cwd() / out_name

    try:
        out = cleanup_video(
            src,
            output_path=out_path,
            overlays=overlays,
            sample_frames=args.sample_frames,
            suffix=args.suffix,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())