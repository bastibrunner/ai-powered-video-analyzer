import logging
import os
import sys

from .pipeline import *
from moviepy import VideoFileClip, AudioFileClip


def analyze(video_path):
    pipeline.audio_event_detection.analyze(video_path)
    audio_transcript.analyze(video_path)
    object_detection.analyze(video_path)
    scene_description.analyze(video_path)
    video_cleanup.analyze(video_path)
    video_cut_detection.analyze(video_path)
    video_processing.analyze(video_path)


def main():
    """
    Entry point that supports both CLI (headless) and GUI usage.

    - If a video path is provided as a positional CLI argument, the script runs
        `process_video` directly without creating any GUI window. This is safe on
        headless servers without a DISPLAY.
    - If no video argument is provided, the Tkinter GUI is started (if possible).
    """
    import argparse

    parser = argparse.ArgumentParser(description="AI-powered video analyzer.")
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to the video file.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32,
        help="Process every n-th frame (default: 32).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video alongside the input (default: disabled in CLI).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        logging.error("Video file not found: %s", args.video)
        sys.exit(1)




if __name__ == "__main__":
    main()
