"""Video assembly and rendering"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .effects import ken_burns_effect, add_subtitles, crossfade_frames

console = Console()


@dataclass
class VideoSegment:
    """A segment of the final video"""
    image_path: Path | None = None
    audio_path: Path | None = None
    text: str = ""
    duration: float = 5.0


@dataclass
class VideoConfig:
    """Video output configuration"""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    codec: str = "libx264"
    bitrate: str = "8M"
    enable_ken_burns: bool = True
    ken_burns_zoom: float = 1.2
    transition_duration: float = 0.5
    subtitle_font_size: int = 60
    subtitle_color: str = "white"
    subtitle_stroke_color: str = "black"
    subtitle_stroke_width: int = 3
    subtitle_position: str = "bottom"


class VideoAssembler:
    """Assembles video from images, audio, and text"""

    def __init__(self, config: VideoConfig | None = None):
        self.config = config or VideoConfig()
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        self.ffmpeg_path = shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            console.print("[yellow]Warning: FFmpeg not found in PATH[/yellow]")
            return False
        return True

    def create_segment_frames(
        self,
        segment: VideoSegment,
        direction: str = "in",
    ) -> list[np.ndarray]:
        """Create frames for a single segment"""
        frames = []

        if segment.image_path and segment.image_path.exists():
            # Apply Ken Burns effect to image
            if self.config.enable_ken_burns:
                frames = ken_burns_effect(
                    segment.image_path,
                    duration=segment.duration,
                    target_size=(self.config.width, self.config.height),
                    zoom_ratio=self.config.ken_burns_zoom,
                    fps=self.config.fps,
                    direction=direction,
                )
            else:
                # Static image
                from PIL import Image
                img = Image.open(segment.image_path)
                img = img.resize(
                    (self.config.width, self.config.height),
                    Image.Resampling.LANCZOS
                )
                frame = np.array(img)
                num_frames = int(segment.duration * self.config.fps)
                frames = [frame] * num_frames
        else:
            # Create solid color background
            num_frames = int(segment.duration * self.config.fps)
            frame = np.zeros(
                (self.config.height, self.config.width, 3),
                dtype=np.uint8
            )
            frame[:] = (30, 30, 30)  # Dark gray
            frames = [frame] * num_frames

        # Add subtitles if text is provided
        if segment.text:
            frames = add_subtitles(
                frames,
                segment.text,
                font_size=self.config.subtitle_font_size,
                color=self.config.subtitle_color,
                stroke_color=self.config.subtitle_stroke_color,
                stroke_width=self.config.subtitle_stroke_width,
                position=self.config.subtitle_position,
            )

        return frames

    def assemble(
        self,
        segments: list[VideoSegment],
        output_path: Path | str,
        background_music_path: Path | str | None = None,
        music_volume: float = 0.3,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Assemble all segments into final video"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate all frames
            all_frames = []
            directions = ["in", "out"]  # Alternate Ken Burns direction

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Rendering segments...", total=len(segments))

                for i, segment in enumerate(segments):
                    direction = directions[i % 2]
                    frames = self.create_segment_frames(segment, direction)

                    if all_frames and self.config.transition_duration > 0:
                        # Add crossfade transition
                        transition_frames = int(
                            self.config.transition_duration * self.config.fps
                        )
                        all_frames = crossfade_frames(
                            all_frames, frames, transition_frames
                        )
                    else:
                        all_frames.extend(frames)

                    progress.update(task, advance=1)

                    if progress_callback:
                        progress_callback(i + 1, len(segments))

            # Write frames to video
            console.print("[cyan]Encoding video...[/cyan]")
            frames_video = temp_path / "frames.mp4"
            self._write_frames_to_video(all_frames, frames_video)

            # Combine audio files
            audio_paths = [s.audio_path for s in segments if s.audio_path]
            if audio_paths:
                console.print("[cyan]Processing audio...[/cyan]")
                combined_audio = temp_path / "combined_audio.wav"
                self._combine_audio(audio_paths, combined_audio)

                # Merge video and audio
                console.print("[cyan]Merging audio and video...[/cyan]")
                self._merge_audio_video(
                    frames_video,
                    combined_audio,
                    output_path,
                    background_music_path,
                    music_volume,
                )
            else:
                # Just copy video
                shutil.copy(frames_video, output_path)

        console.print(f"[green]Video saved to: {output_path}[/green]")
        return output_path

    def _write_frames_to_video(
        self,
        frames: list[np.ndarray],
        output_path: Path,
    ) -> None:
        """Write frames to video file using FFmpeg"""
        if not frames:
            raise ValueError("No frames to write")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.config.width}x{self.config.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.config.fps),
            "-i", "-",
            "-c:v", self.config.codec,
            "-pix_fmt", "yuv420p",
            "-b:v", self.config.bitrate,
            "-preset", "medium",
            str(output_path),
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        for frame in frames:
            # Ensure frame is RGB
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg error: {stderr}")

    def _combine_audio(
        self,
        audio_paths: list[Path],
        output_path: Path,
    ) -> None:
        """Combine multiple audio files with silence between"""
        # Create concat file
        concat_file = output_path.parent / "concat.txt"
        with open(concat_file, "w") as f:
            for path in audio_paths:
                if path and path.exists():
                    f.write(f"file '{path.absolute()}'\n")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)

    def _merge_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        background_music_path: Path | str | None = None,
        music_volume: float = 0.3,
    ) -> None:
        """Merge audio and video tracks"""
        if background_music_path and Path(background_music_path).exists():
            # Mix voice with background music
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-i", str(background_music_path),
                "-filter_complex",
                f"[1:a]volume=1.0[voice];[2:a]volume={music_volume}[music];[voice][music]amix=inputs=2:duration=shortest[a]",
                "-map", "0:v",
                "-map", "[a]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_path),
            ]
        else:
            # Just voice audio
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_path),
            ]

        subprocess.run(cmd, capture_output=True, check=True)


def create_video(
    segments: list[VideoSegment],
    output_path: Path | str,
    config: VideoConfig | None = None,
    **kwargs,
) -> Path:
    """Convenience function to create video from segments"""
    assembler = VideoAssembler(config)
    return assembler.assemble(segments, output_path, **kwargs)
