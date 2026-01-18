"""Main video generation pipeline"""
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .utils.config import Config, load_config
from .llm.ollama import OllamaClient, generate_script, VideoScript
from .tts.piper import PiperTTS, AudioSegment
from .video.assembler import VideoAssembler, VideoSegment, VideoConfig
from .media.library import MediaLibrary, download_sample_assets

console = Console()


@dataclass
class GeneratedVideo:
    """Result of video generation"""
    video_path: Path
    script: VideoScript
    audio_segments: list[AudioSegment]
    duration: float
    metadata: dict = field(default_factory=dict)


class Pipeline:
    """Main video generation pipeline - fully local, no API keys"""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self._setup_directories()
        self._setup_components()

    def _setup_directories(self) -> None:
        """Create necessary directories"""
        self.output_dir = Path(self.config.general.output_dir)
        self.temp_dir = Path(self.config.general.temp_dir)

        for dir_path in [self.output_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_components(self) -> None:
        """Initialize pipeline components"""
        # LLM
        self.llm = OllamaClient(
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
        )

        # TTS
        self.tts = PiperTTS(
            model_name=self.config.tts.model,
            length_scale=self.config.tts.length_scale,
            noise_scale=self.config.tts.noise_scale,
            noise_w=self.config.tts.noise_w,
        )

        # Video assembler
        video_config = VideoConfig(
            width=self.config.video.width,
            height=self.config.video.height,
            fps=self.config.video.fps,
            codec=self.config.video.codec,
            bitrate=self.config.video.bitrate,
            enable_ken_burns=self.config.video.enable_ken_burns,
            ken_burns_zoom=self.config.video.ken_burns_zoom,
            transition_duration=self.config.video.transition_duration,
            subtitle_font_size=self.config.subtitles.font_size,
            subtitle_color=self.config.subtitles.color,
            subtitle_stroke_color=self.config.subtitles.stroke_color,
            subtitle_stroke_width=self.config.subtitles.stroke_width,
            subtitle_position=self.config.subtitles.position,
        )
        self.video_assembler = VideoAssembler(video_config)

        # Media library
        self.media = MediaLibrary(
            images_dir=self.config.media.images_dir,
            footage_dir=self.config.media.footage_dir,
            music_dir=self.config.media.music_dir,
        )

    def check_dependencies(self) -> dict[str, bool]:
        """Check if all dependencies are available"""
        checks = {
            "Ollama": self.llm.is_available(),
            "Piper TTS": self.tts.is_available(),
            "FFmpeg": self.video_assembler.ffmpeg_path is not None,
            "Media Library": self.media.stats()["images"] > 0,
        }
        return checks

    def print_status(self) -> None:
        """Print pipeline status"""
        checks = self.check_dependencies()

        table = Table(title="Pipeline Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status")
        table.add_column("Notes", style="dim")

        for name, available in checks.items():
            status = "[green]✓ Ready[/green]" if available else "[red]✗ Not Ready[/red]"

            notes = ""
            if name == "Ollama" and not available:
                notes = "Run: ollama serve"
            elif name == "Piper TTS" and not available:
                notes = "Run: pip install piper-tts"
            elif name == "FFmpeg" and not available:
                notes = "Install FFmpeg"
            elif name == "Media Library" and not available:
                notes = "Add images to assets/images/"

            table.add_row(name, status, notes)

        console.print(table)

        # Show available models
        if checks["Ollama"]:
            models = self.llm.list_models()
            if models:
                console.print(f"\n[dim]Available Ollama models: {', '.join(models[:5])}[/dim]")

        # Show media stats
        self.media.print_stats()

    def generate(
        self,
        topic: str,
        style: str = "informative",
        duration: int = 30,
        output_name: str | None = None,
        background_music: bool = True,
        music_volume: float = 0.3,
    ) -> GeneratedVideo | None:
        """
        Generate a complete video from a topic.

        Args:
            topic: The video topic/subject
            style: Content style (informative, entertaining, educational)
            duration: Target duration in seconds
            output_name: Output filename (without extension)
            background_music: Whether to add background music
            music_volume: Background music volume (0.0-1.0)

        Returns:
            GeneratedVideo object or None if generation fails
        """
        console.print(Panel(f"[bold]Generating video about: {topic}[/bold]"))

        # Check dependencies
        checks = self.check_dependencies()
        missing = [name for name, ok in checks.items() if not ok]

        if "Ollama" in missing:
            console.print("[red]Error: Ollama is not running. Start with: ollama serve[/red]")
            return None

        if "FFmpeg" in missing:
            console.print("[red]Error: FFmpeg is not installed[/red]")
            return None

        # Create unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_name:
            output_name = output_name.replace(" ", "_")
        else:
            output_name = topic.replace(" ", "_")[:30]

        run_dir = self.temp_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Generate script
            console.print("\n[bold cyan]Step 1/4: Generating script...[/bold cyan]")
            script = generate_script(
                self.llm,
                topic=topic,
                style=style,
                duration=duration,
            )
            console.print(f"  Title: {script.title}")
            console.print(f"  Segments: {len(script.segments)}")

            # Step 2: Generate audio
            console.print("\n[bold cyan]Step 2/4: Synthesizing voice...[/bold cyan]")
            audio_segments = []

            if not self.tts.is_available():
                console.print("[yellow]Warning: Piper not available, skipping audio[/yellow]")
            else:
                audio_dir = run_dir / "audio"
                audio_dir.mkdir(exist_ok=True)

                for i, segment in enumerate(script.segments):
                    audio = self.tts.synthesize(
                        segment.text,
                        audio_dir / f"segment_{i:03d}.wav",
                    )
                    if audio:
                        audio_segments.append(audio)
                        console.print(f"  [dim]Segment {i+1}: {audio.duration:.1f}s[/dim]")

            # Step 3: Get images
            console.print("\n[bold cyan]Step 3/4: Selecting images...[/bold cyan]")

            if self.media.stats()["images"] == 0:
                console.print("[yellow]No images in library - creating samples...[/yellow]")
                download_sample_assets(self.media)

            images = self.media.get_images_for_segments(
                len(script.segments),
                tags=topic.lower().split(),
            )
            console.print(f"  Selected {len(images)} images")

            # Step 4: Assemble video
            console.print("\n[bold cyan]Step 4/4: Assembling video...[/bold cyan]")

            video_segments = []
            for i, segment in enumerate(script.segments):
                # Get duration from audio or use hint
                if i < len(audio_segments):
                    seg_duration = audio_segments[i].duration + self.config.audio.silence_padding
                    audio_path = audio_segments[i].path
                else:
                    seg_duration = segment.duration_hint or 5.0
                    audio_path = None

                # Get image
                image_path = images[i].path if i < len(images) else None

                video_segments.append(VideoSegment(
                    image_path=image_path,
                    audio_path=audio_path,
                    text=segment.text if self.config.subtitles.enabled else "",
                    duration=seg_duration,
                ))

            # Get background music
            music_path = None
            if background_music:
                music_asset = self.media.get_random_music()
                if music_asset:
                    music_path = music_asset.path
                    console.print(f"  Background music: {music_asset.path.name}")

            # Generate output path
            output_path = self.output_dir / f"{output_name}_{run_id}.mp4"

            # Assemble
            final_path = self.video_assembler.assemble(
                video_segments,
                output_path,
                background_music_path=music_path,
                music_volume=music_volume,
            )

            # Calculate total duration
            total_duration = sum(s.duration for s in video_segments)

            result = GeneratedVideo(
                video_path=final_path,
                script=script,
                audio_segments=audio_segments,
                duration=total_duration,
                metadata={
                    "topic": topic,
                    "style": style,
                    "run_id": run_id,
                    "title": script.title,
                    "description": script.description,
                    "hashtags": script.hashtags,
                },
            )

            # Print summary
            console.print(Panel(
                f"[bold green]Video generated successfully![/bold green]\n\n"
                f"[bold]Title:[/bold] {script.title}\n"
                f"[bold]Duration:[/bold] {total_duration:.1f}s\n"
                f"[bold]Output:[/bold] {final_path}\n\n"
                f"[dim]Description:[/dim] {script.description}\n"
                f"[dim]Hashtags:[/dim] {' '.join('#' + h for h in script.hashtags)}",
                title="Generation Complete",
            ))

            return result

        except Exception as e:
            console.print(f"[red]Error during generation: {e}[/red]")
            import traceback
            traceback.print_exc()
            return None


def create_video(
    topic: str,
    config: Config | None = None,
    **kwargs,
) -> GeneratedVideo | None:
    """Convenience function to generate a video"""
    pipeline = Pipeline(config)
    return pipeline.generate(topic, **kwargs)
