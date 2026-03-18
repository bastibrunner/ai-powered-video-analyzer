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

    if start_from in {"left", "top"}:
        indices = range(0, max_scan_px)
    elif start_from in {"right", "bottom"}:
        indices = range(len(means) - 1, len(means) - 1 - max_scan_px, -1)
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
        return int(last_good_pos + 1)

    # Right/bottom scan counts inward from index len(means)-1.
    return int((len(means) - 1) - last_good_pos + 1)


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

                    # Per-column/per-row means of intensity.
                    col_means = gray.mean(axis=0)  # (W,)
                    row_means = gray.mean(axis=1)  # (H,)

                    # Reference region stats near each edge (mean and stddev).
                    left_region = gray[:, :effective_edge_cols]
                    right_region = gray[:, max(0, w - effective_edge_cols) : w]
                    top_region = gray[:effective_edge_cols, :]
                    bottom_region = gray[max(0, h - effective_edge_cols) : h, :]

                    left_ref_mean = float(left_region.mean())
                    left_ref_std = float(left_region.std())
                    right_ref_mean = float(right_region.mean())
                    right_ref_std = float(right_region.std())
                    top_ref_mean = float(top_region.mean())
                    top_ref_std = float(top_region.std())
                    bottom_ref_mean = float(bottom_region.mean())
                    bottom_ref_std = float(bottom_region.std())

                    max_scan_left = int(min(w // 2, max_scan_fraction * w))
                    max_scan_top = int(min(h // 2, max_scan_fraction * h))

                    left_th = _scan_bar_thickness_from_edge(
                        col_means,
                        left_ref_mean,
                        left_ref_std,
                        start_from="left",
                        max_scan_px=max_scan_left,
                        color_diff_threshold=color_diff_threshold,
                        edge_uniformity_std_threshold=edge_uniformity_std_threshold,
                        patience=patience,
                    )
                    right_th = _scan_bar_thickness_from_edge(
                        col_means,
                        right_ref_mean,
                        right_ref_std,
                        start_from="right",
                        max_scan_px=max_scan_left,
                        color_diff_threshold=color_diff_threshold,
                        edge_uniformity_std_threshold=edge_uniformity_std_threshold,
                        patience=patience,
                    )
                    top_th = _scan_bar_thickness_from_edge(
                        row_means,
                        top_ref_mean,
                        top_ref_std,
                        start_from="top",
                        max_scan_px=max_scan_top,
                        color_diff_threshold=color_diff_threshold,
                        edge_uniformity_std_threshold=edge_uniformity_std_threshold,
                        patience=patience,
                    )
                    bottom_th = _scan_bar_thickness_from_edge(
                        row_means,
                        bottom_ref_mean,
                        bottom_ref_std,
                        start_from="bottom",
                        max_scan_px=max_scan_top,
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