from .audio_event_detection import analyze as audio_event_detection
from .audio_transcript import analyze as audio_transcript
from .object_detection import analyze as object_detection
from .scene_description import analyze as scene_description
from .summarization import analyze as summarization
from .video_cleanup import analyze as video_cleanup
from .video_cut_detection import analyze as video_cut_detection

__all__ = ["audio_event_detection", "audio_transcript", "object_detection", "scene_description", "video_cleanup", "video_cut_detection", "video_processing"]

def analyze(video_path):
    audio_event_detection.analyze(video_path)
    audio_transcript.analyze(video_path)
    object_detection.analyze(video_path)
    scene_description.analyze(video_path)
    video_cleanup.analyze(video_path)
    video_cut_detection.analyze(video_path)
    video_processing.analyze(video_path)