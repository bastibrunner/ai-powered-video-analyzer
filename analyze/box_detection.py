# This module is used to detect pillar and letterboxing. Input is a video file, 
# the video gets analyzed frame by frame. The color of all pixels on the edge of the video 
# are compared, if the color is the same on one side, it is considered a pillar or letterbox.
# Next, the thickness of the box is calculated by checking the pixels towards the center of
# the video. The size of the box is then saved for every frame into a text file.
# Example file entry: 
# > Frame 0: 100 0 100 0
# This would be the output for frame 0. The first number is the frame number, the second number is the thickness 
# of the box on the left side, then top, right, bottom.



from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


def _scan_bar_thickness_from_edge(
    means: np.ndarray,
    ref_mean: float,
    ref_std: float,
    *,
    start_from: str,
    outer_idx: int = 0,
    max_scan_px: int,
    color_diff_threshold: float,
    edge_uniformity_std_threshold: float,
    patience: int,
) -> int:
    """
    Scan inward from an edge and estimate the thickness of a near-uniform bar.

    - `means` is per-column (left/right) or per-row (top/bottom) mean intensity.
    - `ref_mean`/`ref_std` describe the intensity distribution in a small
      region at that edge.
    """
    # If the edge isn't uniform enough, we treat it as "no bar" for this side.
    if ref_std > edge_uniformity_std_threshold:
        return 0

    adaptive_thr = max(color_diff_threshold, ref_std * 2.5 + 1.0)
    if max_scan_px <= 0:
        return 0

    frame_len = len(means)

    if start_from in {"left", "top"}:
        start_idx = outer_idx
        end_idx = min(frame_len, outer_idx + max_scan_px)
        indices = range(start_idx, end_idx)
    elif start_from in {"right", "bottom"}:
        start_idx = outer_idx
        end_idx_exclusive = max(-1, outer_idx - max_scan_px)
        indices = range(start_idx, end_idx_exclusive, -1)
    else:
        raise ValueError(f"Unknown start_from={start_from!r}")

    last_good_pos: Optional[int] = None
    misses = 0

    for idx in indices:
        diff = abs(float(means[idx]) - ref_mean)
        if diff <= adaptive_thr:
            last_good_pos = idx
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                break

    if last_good_pos is None:
        return 0

    if start_from in {"left", "top"}:
        return int(last_good_pos - outer_idx + 1)

    return int(outer_idx - last_good_pos + 1)


def _scan_multi_bar_thickness_from_edge(
    gray: np.ndarray,
    means: np.ndarray,
    *,
    start_from: str,
    max_scan_px_total: int,
    edge_ref_cols: int,
    max_segments: int,
    color_diff_threshold: float,
    edge_uniformity_std_threshold: float,
    patience: int,
) -> int:
    """
    Scan inward from an edge and sum the thicknesses of multiple adjacent
    near-uniform bar segments (e.g. nested borders).
    """
    h, w = gray.shape[:2]
    frame_len = len(means)

    total_thickness = 0
    offset = 0  # how far we've already moved inward from the edge

    for _ in range(max_segments):
        if offset >= max_scan_px_total:
            break

        remaining = max_scan_px_total - offset
        if remaining <= 0:
            break

        # Reference slice for the current edge (used to estimate ref_mean/ref_std).
        ref_slice = None
        if start_from == "left":
            outer_idx = offset
            ref_slice = gray[:, offset : min(w, offset + edge_ref_cols)]
        elif start_from == "right":
            outer_idx = w - offset - 1
            left = max(0, w - offset - edge_ref_cols)
            ref_slice = gray[:, left : w - offset]
        elif start_from == "top":
            outer_idx = offset
            ref_slice = gray[offset : min(h, offset + edge_ref_cols), :]
        elif start_from == "bottom":
            outer_idx = h - offset - 1
            top = max(0, h - offset - edge_ref_cols)
            ref_slice = gray[top : h - offset, :]
        else:
            raise ValueError(f"Unknown start_from={start_from!r}")

        # If we have no pixels left to build a reference, stop.
        if ref_slice is None or ref_slice.size == 0:
            break

        # Guard against tiny frame lengths where means and indices can mismatch.
        if outer_idx < 0 or outer_idx >= frame_len:
            break

        ref_mean = float(ref_slice.mean())
        ref_std = float(ref_slice.std())

        seg_th = _scan_bar_thickness_from_edge(
            means,
            ref_mean,
            ref_std,
            start_from=start_from,
            outer_idx=outer_idx,
            max_scan_px=remaining,
            color_diff_threshold=color_diff_threshold,
            edge_uniformity_std_threshold=edge_uniformity_std_threshold,
            patience=patience,
        )

        if seg_th <= 0:
            break

        total_thickness += seg_th
        offset += seg_th

    return int(total_thickness)


def analyze(
    video_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """
    Detect pillar/letterbox thickness per frame and write results to a text file.

    Output format (one line per frame):
    Frame N: <left_thickness> <top_thickness> <right_thickness> <bottom_thickness>
    """
    src = Path(str(video_path))
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Video file not found: {src}")

    if output_path is None:
        output_path = src.with_name(f"{src.stem}_box_detection.txt")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src}")

    # Heuristic parameters for typical 8-bit video frames.
    edge_cols = 4
    color_diff_threshold = 12.0
    edge_uniformity_std_threshold = 25.0
    patience = 2
    max_scan_fraction = 0.5  # never scan more than half the frame from one edge
    single_color_band = 6.0  # grayscale intensity band around median
    single_color_fraction_threshold = 0.99  # center mostly one color -> ignore frame
    single_color_std_threshold = 7.5  # extra guard for very flat frames
    max_segments_per_side = 3  # avoid overfitting on complex content

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar_total = total_frames if total_frames > 0 else None

    try:
        with tqdm(
            total=pbar_total,
            desc="Box detection",
            unit="frame",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            with out.open("w", encoding="utf-8") as f:
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    h, w = gray.shape[:2]

                    # Clamp edge region so slicing stays safe for tiny frames.
                    effective_edge_cols = max(
                        1, min(edge_cols, max(1, min(w, h) // 2))
                    )

                    # Ignore frames where the center is essentially single-color
                    # (e.g. black screens or solid-title cards).
                    # This prevents the edge scanner from "discovering" a bar that
                    # doesn't exist.
                    margin_x = min(w // 4, max(1, int(w * 0.05)))
                    margin_y = min(h // 4, max(1, int(h * 0.05)))
                    if w > 2 * margin_x and h > 2 * margin_y:
                        center = gray[margin_y : h - margin_y, margin_x : w - margin_x]
                    else:
                        center = gray

                    center_median = float(np.median(center))
                    center_std = float(center.std())
                    center_single_frac = float(
                        np.mean(np.abs(center - center_median) <= single_color_band)
                    )

                    # Require whole-frame single-color dominance to avoid skipping
                    # legitimately letterboxed videos with a dark (but not uniform) scene.
                    frame_median = float(np.median(gray))
                    frame_single_frac = float(
                        np.mean(np.abs(gray - frame_median) <= single_color_band)
                    )
                    frame_std = float(gray.std())
                    if (
                        frame_single_frac >= single_color_fraction_threshold
                        and (
                            center_single_frac >= single_color_fraction_threshold
                            or center_std <= single_color_std_threshold
                        )
                        and frame_std <= (single_color_std_threshold * 2.0)
                    ):
                        f.write(f"Frame {frame_idx}: 0 0 0 0\n")
                        frame_idx += 1
                        pbar.update(1)
                        continue

                    # Per-column/per-row means of intensity.
                    col_means = gray.mean(axis=0)  # (W,)
                    row_means = gray.mean(axis=1)  # (H,)

                    max_scan_left = int(min(w // 2, max_scan_fraction * w))
                    max_scan_top = int(min(h // 2, max_scan_fraction * h))

                    # Sum multiple adjacent box segments on each side.
                    left_th = _scan_multi_bar_thickness_from_edge(
                        gray,
                        col_means,
                        start_from="left",
                        max_scan_px_total=max_scan_left,
                        edge_ref_cols=effective_edge_cols,
                        max_segments=max_segments_per_side,
                        color_diff_threshold=color_diff_threshold,
                        edge_uniformity_std_threshold=edge_uniformity_std_threshold,
                        patience=patience,
                    )
                    right_th = _scan_multi_bar_thickness_from_edge(
                        gray,
                        col_means,
                        start_from="right",
                        max_scan_px_total=max_scan_left,
                        edge_ref_cols=effective_edge_cols,
                        max_segments=max_segments_per_side,
                        color_diff_threshold=color_diff_threshold,
                        edge_uniformity_std_threshold=edge_uniformity_std_threshold,
                        patience=patience,
                    )
                    top_th = _scan_multi_bar_thickness_from_edge(
                        gray,
                        row_means,
                        start_from="top",
                        max_scan_px_total=max_scan_top,
                        edge_ref_cols=effective_edge_cols,
                        max_segments=max_segments_per_side,
                        color_diff_threshold=color_diff_threshold,
                        edge_uniformity_std_threshold=edge_uniformity_std_threshold,
                        patience=patience,
                    )
                    bottom_th = _scan_multi_bar_thickness_from_edge(
                        gray,
                        row_means,
                        start_from="bottom",
                        max_scan_px_total=max_scan_top,
                        edge_ref_cols=effective_edge_cols,
                        max_segments=max_segments_per_side,
                        color_diff_threshold=color_diff_threshold,
                        edge_uniformity_std_threshold=edge_uniformity_std_threshold,
                        patience=patience,
                    )

                    f.write(
                        f"Frame {frame_idx}: {left_th} {top_th} {right_th} {bottom_th}\n"
                    )
                    frame_idx += 1
                    pbar.update(1)
    finally:
        cap.release()

    return str(out)


def main(video_path: str) -> None:
    if not os.path.exists(video_path):
        print("Video file not found: ", video_path, file=sys.stderr)
        sys.exit(1)
    result = analyze(video_path)
    print(result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python box_detection.py <video_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])