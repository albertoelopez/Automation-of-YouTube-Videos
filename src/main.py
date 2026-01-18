"""CLI interface for Local YouTube Automation"""
import typer
from pathlib import Path
from typing import Optional
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
    generate_images: bool = typer.Option(False, "--generate-images", "-g", help="Generate images with AI"),
    upload: bool = typer.Option(False, "--upload", "-u", help="Upload to YouTube after generation"),
    privacy: str = typer.Option("private", "--privacy", help="YouTube privacy: public, private, unlisted"),
    config: Path = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Generate a video from a topic"""
    cfg = load_config(config) if config else load_config()
    pipeline = Pipeline(cfg)

    # Generate images if requested
    if generate_images:
        console.print("[cyan]Generating images with AI...[/cyan]")
        from .images.generator import ImageGenerator
        img_gen = ImageGenerator()
        # This will be handled by the enhanced pipeline

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

        # Upload if requested
        if upload:
            console.print("\n[cyan]Uploading to YouTube...[/cyan]")
            from .upload.youtube import YouTubeUploader

            uploader = YouTubeUploader()
            upload_result = uploader.upload_from_result(
                video_path=result.video_path,
                metadata=result.metadata,
                privacy_status=privacy,
            )

            if upload_result:
                console.print(f"[green]Uploaded: {upload_result.url}[/green]")
    else:
        console.print("\n[red]Video generation failed[/red]")
        raise typer.Exit(1)


@app.command()
def batch(
    file: Path = typer.Argument(..., help="Batch jobs file (JSON, CSV, or text)"),
    continue_on_error: bool = typer.Option(True, "--continue", help="Continue on errors"),
    no_music: bool = typer.Option(False, "--no-music", help="Disable background music"),
    config: Path = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Process multiple videos from a batch file"""
    from .batch import BatchProcessor

    cfg = load_config(config) if config else load_config()
    processor = BatchProcessor(cfg)

    results = processor.process_file(
        file,
        continue_on_error=continue_on_error,
        background_music=not no_music,
    )

    processor.print_summary(results)


@app.command("batch-sample")
def batch_sample(
    output: Path = typer.Option("batch_jobs.json", "--output", "-o", help="Output file path"),
):
    """Create a sample batch jobs file"""
    from .batch import create_sample_batch_file
    create_sample_batch_file(output)


@app.command()
def status():
    """Check pipeline status and dependencies"""
    config = load_config()
    pipeline = Pipeline(config)
    pipeline.print_status()

    # Check additional components
    console.print("\n[bold]Additional Components:[/bold]")

    # Image generator
    try:
        from .images.generator import ImageGenerator
        gen = ImageGenerator()
        backend = gen._get_backend()
        console.print(f"  Image Generation: [green]✓[/green] ({backend})")
    except Exception as e:
        console.print(f"  Image Generation: [yellow]⚠ {e}[/yellow]")

    # YouTube uploader
    try:
        from .upload.youtube import YouTubeUploader
        uploader = YouTubeUploader()
        if uploader.is_configured():
            auth_status = "[green]✓ Authenticated[/green]" if uploader.is_authenticated() else "[yellow]Not authenticated[/yellow]"
            console.print(f"  YouTube Upload: {auth_status}")
        else:
            console.print("  YouTube Upload: [dim]Not configured (client_secrets.json missing)[/dim]")
    except Exception as e:
        console.print(f"  YouTube Upload: [dim]Not available[/dim]")


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


@app.command("generate-image")
def generate_image_cmd(
    prompt: str = typer.Argument(..., help="Image prompt"),
    width: int = typer.Option(1920, "--width", "-w", help="Image width"),
    height: int = typer.Option(1080, "--height", "-h", help="Image height"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path"),
):
    """Generate an image using AI (Stable Diffusion or Pollinations)"""
    from .images.generator import ImageGenerator

    gen = ImageGenerator()
    console.print(f"[cyan]Using backend: {gen._get_backend()}[/cyan]")

    result = gen.generate(prompt, width=width, height=height)

    if result:
        console.print(f"[green]Image saved to: {result.path}[/green]")
    else:
        console.print("[red]Image generation failed[/red]")
        raise typer.Exit(1)


@app.command("youtube-auth")
def youtube_auth():
    """Authenticate with YouTube for uploads"""
    from .upload.youtube import YouTubeUploader

    uploader = YouTubeUploader()

    if not uploader.is_configured():
        console.print("[red]YouTube not configured.[/red]")
        console.print("\nTo enable YouTube uploads:")
        console.print("  1. Go to https://console.cloud.google.com/")
        console.print("  2. Create a project")
        console.print("  3. Enable YouTube Data API v3")
        console.print("  4. Create OAuth 2.0 credentials (Desktop app)")
        console.print("  5. Download client_secrets.json to config/")
        raise typer.Exit(1)

    if uploader.authenticate():
        console.print("[green]YouTube authentication successful![/green]")
    else:
        console.print("[red]Authentication failed[/red]")
        raise typer.Exit(1)


@app.command("upload")
def upload_cmd(
    video: Path = typer.Argument(..., help="Path to video file"),
    title: str = typer.Option(..., "--title", "-t", help="Video title"),
    description: str = typer.Option("", "--description", "-d", help="Video description"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    privacy: str = typer.Option("private", "--privacy", "-p", help="Privacy: public, private, unlisted"),
):
    """Upload a video to YouTube"""
    from .upload.youtube import YouTubeUploader

    uploader = YouTubeUploader()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    result = uploader.upload(
        video_path=video,
        title=title,
        description=description,
        tags=tag_list,
        privacy_status=privacy,
    )

    if result:
        console.print(f"\n[green]Upload complete![/green]")
        console.print(f"URL: {result.url}")
    else:
        console.print("[red]Upload failed[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
