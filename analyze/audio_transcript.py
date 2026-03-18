"""
This module transcribes the audio of a video (or audio) file using Whisper.

It writes a timestamped transcript to a text file with entries like:

> 00:00:00 - 00:00:05: This is the transcript of the audio for the first 5 seconds.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Any

import whisper


def _seconds_to_timestr(seconds: float) -> str:
    # Keep formatting consistent with the GUI's helper.
    seconds_i = max(0, int(seconds))
    hrs = int(seconds_i // 3600)
    mins = int((seconds_i % 3600) // 60)
    secs = int(seconds_i % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _get_preferred_whisper_device() -> str:
    """
    Choose Whisper device with an explicit override.

    WHISPER_DEVICE:
      - "cpu" forces CPU
      - "cuda" forces CUDA (will fall back to CPU if CUDA initialization fails)
      - unset/other uses CUDA if available, else CPU
    """
    try:
        import torch
    except Exception:
        return "cpu"

    override = (os.environ.get("WHISPER_DEVICE") or "").strip().lower()
    if override in {"cpu", "cuda"}:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_whisper_model(model_name: str) -> Any:
    device = _get_preferred_whisper_device()
    logging.info("Loading Whisper model %s (%s)...", model_name, device)
    try:
        return whisper.load_model(model_name, device=device)
    except RuntimeError as e:
        # Common case: the installed CUDA build isn't compatible with the GPU.
        msg_l = str(e).lower()
        if device == "cuda" and ("no kernel image" in msg_l or "cudaerror" in msg_l or "cuda error" in msg_l):
            logging.error("Whisper failed on CUDA; falling back to CPU: %s", e)
            return whisper.load_model(model_name, device="cpu")
        raise


def _is_audio_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def _extract_audio_from_video(video_path: str, audio_path: str) -> None:
    # MoviePy is already a project dependency (used by the GUI and CLI).
    from moviepy import VideoFileClip

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        # Ensure we close the reader even if there's no audio track.
        try:
            clip.reader.close()
        except Exception:
            pass
        raise ValueError("No audio track found in the video.")

    clip.audio.write_audiofile(audio_path, logger=None)
    try:
        clip.reader.close()
    except Exception:
        pass
    # MoviePy's underlying audio reader API differs across versions.
    # Best-effort cleanup must never break the main path.
    try:
        if clip.audio is not None and getattr(clip.audio, "reader", None) is not None:
            reader = clip.audio.reader
            if hasattr(reader, "close_proc"):
                reader.close_proc()
            elif hasattr(reader, "close"):
                reader.close()
    except Exception:
        pass


def _transcribe_audio(model: Any, audio_path: str, language: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """
    Returns (combined_transcript_text, segments).

    segments are Whisper segment dicts (typically containing: start, end, text).
    """
    logging.info("Transcribing audio via Whisper: %s", audio_path)
    whisper_language = language
    if whisper_language:
        # The GUI historically uses some extended/3-letter codes; map them to Whisper's expected ISO-ish codes.
        # Whisper typically accepts e.g. "en", "fa", "de", "fr", etc.
        _lang_map = {
            "none": None,
            "eng": "en",
            "fas": "fa",
            "chi_sim": "zh",
            "chi_tra": "zh",
            "spa": "es",
            "fra": "fr",
            "deu": "de",
            "ara": "ar",
            "jpn": "ja",
            "kor": "ko",
            "rus": "ru",
            "ita": "it",
        }
        whisper_language = _lang_map.get(whisper_language, whisper_language)
    result = model.transcribe(audio_path, task="transcribe", language=whisper_language)
    segments = result.get("segments") or []

    # Join into a single text blob for callers that want "just the transcript".
    combined = " ".join((s.get("text") or "").strip() for s in segments).strip()
    return combined, segments


def _write_timestamped_transcript(segments: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or 0.0)
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            f.write(f"> {_seconds_to_timestr(start)} - {_seconds_to_timestr(end)}: {text}\n")


def analyze(video_or_audio_path: str, *, language: str | None = None, model_name: str = "large-v2") -> str:
    """
    Transcribe audio and write `*_audio_transcript.txt` in the current working directory.

    Returns the combined transcript text.
    """
    if not os.path.exists(video_or_audio_path):
        raise FileNotFoundError(f"Input file not found: {video_or_audio_path}")

    base_name = os.path.splitext(os.path.basename(video_or_audio_path))[0]
    output_path = os.path.abspath(f"{base_name}_audio_transcript.txt")

    # Lazily load a Whisper model (kept within a call) to avoid surprises at import-time.
    model = _load_whisper_model(model_name)

    tmp_audio: str | None = None
    try:
        if _is_audio_file(video_or_audio_path):
            audio_path = video_or_audio_path
        else:
            # Extract audio to a temporary wav file for Whisper.
            fd, tmp_audio = tempfile.mkstemp(prefix=f"{base_name}_", suffix=".wav")
            os.close(fd)
            _extract_audio_from_video(video_or_audio_path, tmp_audio)
            audio_path = tmp_audio

        combined, segments = _transcribe_audio(model, audio_path, language=language)
        _write_timestamped_transcript(segments, output_path=output_path)
        logging.info("Wrote timestamped transcript: %s", output_path)
        return combined
    finally:
        if tmp_audio:
            try:
                os.remove(tmp_audio)
            except Exception:
                pass


def main(input_path: str) -> None:
    if not os.path.exists(input_path):
        print("Input file not found:", input_path)
        sys.exit(1)
    transcript = analyze(input_path)
    print(transcript)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_transcript.py <video_path|audio_path>")
        sys.exit(1)
    main(sys.argv[1])