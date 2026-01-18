"""CLI interface for Local YouTube Automation"""
import typer
from pathlib import Path
from rich.console import Console

from .pipeline import Pipeline, create_video
from .utils.config import load_config
from .media.library import download_sample_assets

app = typer.Typer(
    name="ytauto",
    help="Local YouTube Automation - No API keys required",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    topic: str = typer.Argument(..., help="Video topic or subject"),
    style: str = typer.Option("informative", "--style", "-s", help="Content style"),
    duration: int = typer.Option(30, "--duration", "-d", help="Target duration in seconds"),
    output: str = typer.Option(None, "--output", "-o", help="Output filename"),
    no_music: bool = typer.Option(False, "--no-music", help="Disable background music"),
    music_volume: float = typer.Option(0.3, "--music-volume", help="Background music volume"),
    config: Path = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Generate a video from a topic"""
    cfg = load_config(config) if config else load_config()
    pipeline = Pipeline(cfg)

    result = pipeline.generate(
        topic=topic,
        style=style,
        duration=duration,
        output_name=output,
        background_music=not no_music,
        music_volume=music_volume,
    )

    if result:
        console.print(f"\n[green]Video saved to: {result.video_path}[/green]")
    else:
        console.print("\n[red]Video generation failed[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Check pipeline status and dependencies"""
    config = load_config()
    pipeline = Pipeline(config)
    pipeline.print_status()


@app.command()
def init():
    """Initialize project with sample assets"""
    config = load_config()
    pipeline = Pipeline(config)

    console.print("[cyan]Initializing project...[/cyan]\n")

    # Create sample images
    download_sample_assets(pipeline.media)

    # Download TTS model
    if pipeline.tts.is_available():
        console.print("\n[cyan]Downloading TTS model...[/cyan]")
        pipeline.tts.download_model()

    console.print("\n[green]Initialization complete![/green]")
    console.print("\nNext steps:")
    console.print("  1. Start Ollama: [bold]ollama serve[/bold]")
    console.print("  2. Pull a model: [bold]ollama pull llama3.1:8b[/bold]")
    console.print("  3. Add images to: [bold]assets/images/[/bold]")
    console.print("  4. Add music to: [bold]assets/music/[/bold]")
    console.print("  5. Generate: [bold]python -m src.main generate 'Your topic'[/bold]")


@app.command()
def models():
    """List available Ollama models"""
    config = load_config()
    from .llm.ollama import OllamaClient

    client = OllamaClient(base_url=config.llm.base_url)

    if not client.is_available():
        console.print("[red]Ollama is not running. Start with: ollama serve[/red]")
        raise typer.Exit(1)

    available = client.list_models()

    if available:
        console.print("[bold]Available Ollama Models:[/bold]\n")
        for model in available:
            marker = " [green](current)[/green]" if model == config.llm.model else ""
            console.print(f"  • {model}{marker}")
    else:
        console.print("[yellow]No models found. Pull one with: ollama pull llama3.1:8b[/yellow]")


@app.command()
def voices():
    """List available TTS voices"""
    from .tts.piper import PiperTTS

    console.print("[bold]Available Piper Voices:[/bold]\n")
    for voice_name in PiperTTS.VOICE_MODELS.keys():
        console.print(f"  • {voice_name}")

    console.print("\n[dim]More voices at: https://rhasspy.github.io/piper-samples/[/dim]")


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
