"""Whisper transcription for accurate subtitle timing"""
import subprocess
import shutil
import json
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class WordTiming:
    """A word with its timing"""
    word: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class SubtitleSegment:
    """A subtitle segment with timing"""
    text: str
    start: float  # seconds
    end: float  # seconds
    words: list[WordTiming] | None = None


@dataclass
class TranscriptionResult:
    """Result of audio transcription"""
    text: str
    segments: list[SubtitleSegment]
    language: str
    duration: float


class WhisperTranscriber:
    """
    Whisper-based audio transcription.

    Supports multiple backends:
    1. whisper (OpenAI's Python package) - CPU/GPU
    2. faster-whisper (CTranslate2) - faster, less memory
    3. whisper.cpp (via CLI) - efficient CPU inference
    """

    MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    def __init__(
        self,
        model: str = "base",
        backend: str = "auto",  # auto, whisper, faster-whisper, whisper-cpp
        device: str = "auto",  # auto, cpu, cuda
    ):
        self.model_name = model
        self.backend = backend
        self.device = device
        self._model = None
        self._backend_name = None

    def _detect_backend(self) -> str:
        """Detect available backend"""
        if self.backend != "auto":
            return self.backend

        # Try faster-whisper first (faster, less memory)
        try:
            import faster_whisper
            return "faster-whisper"
        except ImportError:
            pass

        # Try standard whisper
        try:
            import whisper
            return "whisper"
        except ImportError:
            pass

        # Check for whisper.cpp CLI
        if shutil.which("whisper-cpp") or shutil.which("whisper.cpp"):
            return "whisper-cpp"

        raise RuntimeError(
            "No Whisper backend found. Install one of:\n"
            "  pip install openai-whisper\n"
            "  pip install faster-whisper\n"
            "  or install whisper.cpp"
        )

    def _load_model(self):
        """Load the Whisper model"""
        if self._model is not None:
            return

        self._backend_name = self._detect_backend()
        console.print(f"[dim]Using Whisper backend: {self._backend_name}[/dim]")

        if self._backend_name == "faster-whisper":
            from faster_whisper import WhisperModel

            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            compute_type = "float16" if device == "cuda" else "int8"

            console.print(f"[dim]Loading {self.model_name} model on {device}...[/dim]")
            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
            )

        elif self._backend_name == "whisper":
            import whisper

            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            console.print(f"[dim]Loading {self.model_name} model on {device}...[/dim]")
            self._model = whisper.load_model(self.model_name, device=device)

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (e.g., "en", "es") or None for auto-detect
            word_timestamps: Include word-level timing

        Returns:
            TranscriptionResult with segments and timing
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        if self._backend_name == "faster-whisper":
            return self._transcribe_faster_whisper(audio_path, language, word_timestamps)
        elif self._backend_name == "whisper":
            return self._transcribe_whisper(audio_path, language, word_timestamps)
        else:
            return self._transcribe_whisper_cpp(audio_path, language)

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper"""
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )

        segments = []
        full_text = []

        for segment in segments_gen:
            words = None
            if word_timestamps and segment.words:
                words = [
                    WordTiming(word=w.word.strip(), start=w.start, end=w.end)
                    for w in segment.words
                ]

            segments.append(SubtitleSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=words,
            ))
            full_text.append(segment.text.strip())

        return TranscriptionResult(
            text=" ".join(full_text),
            segments=segments,
            language=info.language,
            duration=info.duration,
        )

    def _transcribe_whisper(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI whisper"""
        import whisper

        result = self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
        )

        segments = []
        for segment in result["segments"]:
            words = None
            if word_timestamps and "words" in segment:
                words = [
                    WordTiming(word=w["word"].strip(), start=w["start"], end=w["end"])
                    for w in segment["words"]
                ]

            segments.append(SubtitleSegment(
                text=segment["text"].strip(),
                start=segment["start"],
                end=segment["end"],
                words=words,
            ))

        # Get duration from audio
        import wave
        try:
            with wave.open(str(audio_path), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
        except Exception:
            duration = segments[-1].end if segments else 0.0

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", "en"),
            duration=duration,
        )

    def _transcribe_whisper_cpp(
        self,
        audio_path: Path,
        language: str | None,
    ) -> TranscriptionResult:
        """Transcribe using whisper.cpp CLI"""
        # Find whisper.cpp executable
        whisper_cmd = shutil.which("whisper-cpp") or shutil.which("whisper.cpp") or "whisper"

        cmd = [
            whisper_cmd,
            "-m", f"models/ggml-{self.model_name}.bin",
            "-f", str(audio_path),
            "-oj",  # Output JSON
        ]

        if language:
            cmd.extend(["-l", language])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"whisper.cpp failed: {result.stderr}")

        data = json.loads(result.stdout)

        segments = []
        for seg in data.get("transcription", []):
            segments.append(SubtitleSegment(
                text=seg["text"].strip(),
                start=seg["timestamps"]["from"] / 1000.0,
                end=seg["timestamps"]["to"] / 1000.0,
            ))

        full_text = " ".join(s.text for s in segments)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language or "en",
            duration=segments[-1].end if segments else 0.0,
        )


def transcribe_audio(
    audio_path: Path | str,
    model: str = "base",
    **kwargs,
) -> TranscriptionResult:
    """Convenience function to transcribe audio"""
    transcriber = WhisperTranscriber(model=model)
    return transcriber.transcribe(audio_path, **kwargs)


def generate_srt(
    result: TranscriptionResult,
    output_path: Path | str,
    max_chars_per_line: int = 42,
) -> Path:
    """Generate SRT subtitle file from transcription"""
    output_path = Path(output_path)

    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result.segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n")

            # Word wrap
            text = segment.text
            if len(text) > max_chars_per_line:
                words = text.split()
                lines = []
                current_line = []
                current_len = 0

                for word in words:
                    if current_len + len(word) + 1 > max_chars_per_line and current_line:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_len = len(word)
                    else:
                        current_line.append(word)
                        current_len += len(word) + 1

                if current_line:
                    lines.append(" ".join(current_line))

                text = "\n".join(lines)

            f.write(f"{text}\n\n")

    return output_path
