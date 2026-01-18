"""Piper TTS - Fast local text-to-speech"""
import subprocess
import shutil
import wave
import json
from pathlib import Path
from typing import BinaryIO
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class AudioSegment:
    """Generated audio segment"""
    path: Path
    duration: float  # seconds
    text: str


class PiperTTS:
    """Piper TTS client for local voice synthesis"""

    # Voice model download URLs
    VOICE_MODELS = {
        "en_US-lessac-medium": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        },
        "en_US-amy-medium": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        },
        "en_GB-alan-medium": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        },
        "es_ES-davefx-medium": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json",
        },
    }

    def __init__(
        self,
        model_name: str = "en_US-lessac-medium",
        models_dir: Path | str = "models/piper",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w

        self._piper_path = self._find_piper()

    def _find_piper(self) -> Path | None:
        """Find piper executable"""
        # Check if piper is in PATH
        piper_path = shutil.which("piper")
        if piper_path:
            return Path(piper_path)

        # Check common locations
        common_paths = [
            Path.home() / ".local" / "bin" / "piper",
            Path("/usr/local/bin/piper"),
            Path("/usr/bin/piper"),
            self.models_dir.parent / "piper" / "piper",
        ]

        for path in common_paths:
            if path.exists():
                return path

        return None

    def is_available(self) -> bool:
        """Check if Piper is installed"""
        return self._piper_path is not None

    def get_model_path(self) -> Path:
        """Get path to voice model"""
        return self.models_dir / f"{self.model_name}.onnx"

    def get_config_path(self) -> Path:
        """Get path to model config"""
        return self.models_dir / f"{self.model_name}.onnx.json"

    def is_model_downloaded(self) -> bool:
        """Check if model files exist"""
        return self.get_model_path().exists() and self.get_config_path().exists()

    def download_model(self) -> bool:
        """Download voice model if not present"""
        if self.is_model_downloaded():
            return True

        if self.model_name not in self.VOICE_MODELS:
            console.print(f"[red]Unknown model: {self.model_name}[/red]")
            console.print(f"Available models: {list(self.VOICE_MODELS.keys())}")
            return False

        urls = self.VOICE_MODELS[self.model_name]

        try:
            import httpx

            console.print(f"[cyan]Downloading voice model: {self.model_name}...[/cyan]")

            # Download model
            with httpx.stream("GET", urls["model"], follow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))

                with open(self.get_model_path(), "wb") as f:
                    downloaded = 0
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = (downloaded / total) * 100
                            console.print(f"\r[cyan]Model: {percent:.1f}%[/cyan]", end="")

            console.print()

            # Download config
            with httpx.stream("GET", urls["config"], follow_redirects=True) as response:
                response.raise_for_status()
                with open(self.get_config_path(), "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

            console.print(f"[green]Model downloaded successfully![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Failed to download model: {e}[/red]")
            return False

    def synthesize(self, text: str, output_path: Path | str) -> AudioSegment | None:
        """Synthesize speech from text"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            console.print("[red]Piper not found. Install with: pip install piper-tts[/red]")
            return None

        if not self.is_model_downloaded():
            if not self.download_model():
                return None

        # Run piper
        cmd = [
            str(self._piper_path),
            "--model", str(self.get_model_path()),
            "--output_file", str(output_path),
            "--length_scale", str(self.length_scale),
            "--noise_scale", str(self.noise_scale),
            "--noise_w", str(self.noise_w),
        ]

        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                check=True,
            )

            # Get audio duration
            duration = self._get_wav_duration(output_path)

            return AudioSegment(
                path=output_path,
                duration=duration,
                text=text,
            )

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Piper error: {e.stderr.decode()}[/red]")
            return None
        except FileNotFoundError:
            console.print("[red]Piper executable not found[/red]")
            return None

    def _get_wav_duration(self, path: Path) -> float:
        """Get duration of WAV file in seconds"""
        try:
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0

    def synthesize_segments(
        self,
        segments: list[dict],
        output_dir: Path | str,
        silence_padding: float = 0.3,
    ) -> list[AudioSegment]:
        """Synthesize multiple text segments"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, segment in enumerate(segments):
            text = segment.get("text", segment) if isinstance(segment, dict) else segment
            output_path = output_dir / f"segment_{i:03d}.wav"

            console.print(f"[dim]Synthesizing segment {i + 1}/{len(segments)}...[/dim]")

            audio = self.synthesize(text, output_path)
            if audio:
                results.append(audio)

        return results


def synthesize_speech(
    text: str,
    output_path: Path | str,
    model: str = "en_US-lessac-medium",
    **kwargs,
) -> AudioSegment | None:
    """Convenience function to synthesize speech"""
    tts = PiperTTS(model_name=model, **kwargs)
    return tts.synthesize(text, output_path)
