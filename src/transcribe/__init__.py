"""Audio transcription module"""
from .whisper import WhisperTranscriber, transcribe_audio

__all__ = ["WhisperTranscriber", "transcribe_audio"]
