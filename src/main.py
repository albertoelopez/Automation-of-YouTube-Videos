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
def voices(
    language: str = typer.Option(None, "--language", "-l", help="Filter by language"),
):
    """List available TTS voices"""
    from .tts.piper import PiperTTS
    from rich.table import Table

    voices_list = PiperTTS.list_voices()

    if language:
        voices_list = [v for v in voices_list if language.lower() in v["language"].lower()]

    table = Table(title="Available Piper Voices")
    table.add_column("Voice Name", style="cyan")
    table.add_column("Language")
    table.add_column("Gender")

    for voice in voices_list:
        table.add_row(voice["name"], voice["language"], voice["gender"])

    console.print(table)
    console.print(f"\n[dim]Total: {len(voices_list)} voices[/dim]")
    console.print("[dim]More voices at: https://rhasspy.github.io/piper-samples/[/dim]")


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


@app.command("transcribe")
def transcribe_cmd(
    audio: Path = typer.Argument(..., help="Path to audio file"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model: tiny, base, small, medium, large"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g., en, es)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output SRT file path"),
):
    """Transcribe audio to subtitles using Whisper"""
    try:
        from .transcribe.whisper import WhisperTranscriber, generate_srt
    except ImportError:
        console.print("[red]Whisper not installed. Install with:[/red]")
        console.print('  pip install -e ".[whisper]"')
        raise typer.Exit(1)

    transcriber = WhisperTranscriber(model=model)
    console.print(f"[cyan]Transcribing {audio.name}...[/cyan]")

    result = transcriber.transcribe(audio, language=language)

    console.print(f"[green]Transcription complete![/green]")
    console.print(f"  Language: {result.language}")
    console.print(f"  Duration: {result.duration:.1f}s")
    console.print(f"  Segments: {len(result.segments)}")

    if output:
        srt_path = generate_srt(result, output)
        console.print(f"  SRT saved: {srt_path}")
    else:
        console.print(f"\n[bold]Text:[/bold]\n{result.text}")


@app.command("templates")
def templates_cmd(
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
):
    """List available video templates"""
    from .templates.manager import TemplateManager

    manager = TemplateManager()

    if category:
        templates = manager.list_by_category(category)
        if not templates:
            console.print(f"[yellow]No templates found in category: {category}[/yellow]")
            return
    else:
        manager.print_list()


@app.command("schedule")
def schedule_cmd(
    topic: str = typer.Argument(..., help="Video topic"),
    time: str = typer.Option(..., "--time", "-t", help="Scheduled time (ISO format or 'now+1h')"),
    template: str = typer.Option(None, "--template", help="Template to use"),
    upload_youtube: bool = typer.Option(False, "--youtube", "-y", help="Upload to YouTube"),
    upload_tiktok: bool = typer.Option(False, "--tiktok", "-k", help="Upload to TikTok"),
    privacy: str = typer.Option("private", "--privacy", "-p", help="Privacy setting"),
):
    """Schedule a video for later generation"""
    from .scheduler.scheduler import Scheduler
    from datetime import datetime, timedelta

    scheduler = Scheduler()

    # Parse time
    if time.startswith("now+"):
        # Parse relative time like "now+1h" or "now+30m"
        delta_str = time[4:]
        if delta_str.endswith("h"):
            delta = timedelta(hours=int(delta_str[:-1]))
        elif delta_str.endswith("m"):
            delta = timedelta(minutes=int(delta_str[:-1]))
        elif delta_str.endswith("d"):
            delta = timedelta(days=int(delta_str[:-1]))
        else:
            delta = timedelta(minutes=int(delta_str))
        scheduled_time = datetime.now() + delta
    else:
        scheduled_time = datetime.fromisoformat(time)

    job = scheduler.add_job(
        topic=topic,
        scheduled_time=scheduled_time,
        template=template,
        upload_youtube=upload_youtube,
        upload_tiktok=upload_tiktok,
        privacy=privacy,
    )

    console.print(f"[green]Scheduled job {job.id} for {scheduled_time}[/green]")


@app.command("schedule-list")
def schedule_list_cmd():
    """List scheduled jobs"""
    from .scheduler.scheduler import Scheduler
    scheduler = Scheduler()
    scheduler.print_status()


@app.command("schedule-run")
def schedule_run_cmd(
    foreground: bool = typer.Option(True, "--foreground", "-f", help="Run in foreground"),
):
    """Run the scheduler"""
    from .scheduler.scheduler import run_scheduler
    run_scheduler(foreground=foreground)


@app.command("analytics")
def analytics_cmd(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze"),
    recent: bool = typer.Option(False, "--recent", "-r", help="Show recent logs"),
    performance: bool = typer.Option(False, "--performance", "-p", help="Show video performance"),
):
    """View generation analytics"""
    from .analytics.tracker import AnalyticsTracker

    tracker = AnalyticsTracker()

    if recent:
        tracker.print_recent_logs()
    elif performance:
        tracker.print_video_performance()
    else:
        tracker.print_summary(days=days)


@app.command("tiktok-login")
def tiktok_login_cmd():
    """Login to TikTok for uploads"""
    try:
        from .upload.tiktok import TikTokUploader
    except ImportError:
        console.print("[red]Playwright not installed. Install with:[/red]")
        console.print("  pip install playwright && playwright install chromium")
        raise typer.Exit(1)

    uploader = TikTokUploader(headless=False)
    uploader.login_manual()


@app.command("trending")
def trending_cmd(
    sources: str = typer.Option("google,reddit,hackernews", "--sources", "-s", help="Comma-separated sources"),
    niche: str = typer.Option(None, "--niche", "-n", help="Search specific niche (tech, gaming, finance, etc.)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of results"),
    save: bool = typer.Option(False, "--save", help="Save results to cache"),
):
    """Get trending topics from various sources"""
    from .research.trending import TrendingTopics

    trending = TrendingTopics()

    if niche:
        console.print(f"[cyan]Searching trending in niche: {niche}[/cyan]")
        topics = trending.search_niche(niche)
    else:
        source_list = [s.strip() for s in sources.split(",")]
        console.print(f"[cyan]Fetching from: {', '.join(source_list)}[/cyan]")
        topics = trending.get_all_trending(sources=source_list)

    if topics:
        trending.print_trending(topics, limit=limit)
        if save:
            cache_file = trending.save_cache(topics)
            console.print(f"\n[dim]Saved to: {cache_file}[/dim]")
    else:
        console.print("[yellow]No trending topics found[/yellow]")


@app.command("ideas")
def ideas_cmd(
    topic: str = typer.Argument(..., help="Topic or niche for idea generation"),
    style: str = typer.Option("educational", "--style", "-s", help="Content style"),
    duration: int = typer.Option(30, "--duration", "-d", help="Target duration in seconds"),
):
    """Generate a content idea using AI"""
    from .research.ideas import IdeaGenerator

    generator = IdeaGenerator()
    console.print(f"[cyan]Generating idea for: {topic}[/cyan]\n")

    idea = generator.generate_idea(topic, style=style, duration=duration)

    if idea:
        generator.print_idea(idea)
    else:
        console.print("[red]Failed to generate idea. Make sure Ollama is running.[/red]")
        raise typer.Exit(1)


@app.command("ideas-batch")
def ideas_batch_cmd(
    topic: str = typer.Argument(..., help="Topic for idea generation"),
    count: int = typer.Option(5, "--count", "-c", help="Number of ideas to generate"),
    style: str = typer.Option("educational", "--style", "-s", help="Content style"),
):
    """Generate multiple content ideas"""
    from .research.ideas import IdeaGenerator

    generator = IdeaGenerator()
    console.print(f"[cyan]Generating {count} ideas for: {topic}[/cyan]\n")

    ideas = generator.generate_ideas_batch(topic, count=count, style=style)

    if ideas:
        generator.print_ideas(ideas)
        console.print(f"\n[green]Generated {len(ideas)} ideas[/green]")
    else:
        console.print("[red]Failed to generate ideas[/red]")
        raise typer.Exit(1)


@app.command("ideas-series")
def ideas_series_cmd(
    topic: str = typer.Argument(..., help="Series topic"),
    episodes: int = typer.Option(5, "--episodes", "-e", help="Number of episodes"),
    style: str = typer.Option("educational", "--style", "-s", help="Content style"),
):
    """Generate a video series concept"""
    from .research.ideas import IdeaGenerator

    generator = IdeaGenerator()
    console.print(f"[cyan]Generating {episodes}-part series for: {topic}[/cyan]\n")

    ideas = generator.generate_series(topic, episodes=episodes, style=style)

    if ideas:
        generator.print_ideas(ideas)
        console.print(f"\n[green]Series generated with {len(ideas)} episodes[/green]")
    else:
        console.print("[red]Failed to generate series[/red]")
        raise typer.Exit(1)


@app.command("ideas-from-trending")
def ideas_from_trending_cmd(
    count: int = typer.Option(5, "--count", "-c", help="Number of ideas"),
    niche: str = typer.Option(None, "--niche", "-n", help="Specific niche"),
    style: str = typer.Option("educational", "--style", "-s", help="Content style"),
):
    """Generate ideas based on current trending topics"""
    from .research.trending import TrendingTopics
    from .research.ideas import IdeaGenerator

    trending = TrendingTopics()
    generator = IdeaGenerator()

    console.print("[cyan]Fetching trending topics...[/cyan]")

    if niche:
        topics = trending.search_niche(niche)
    else:
        topics = trending.get_all_trending()

    if not topics:
        console.print("[yellow]No trending topics found[/yellow]")
        raise typer.Exit(1)

    console.print(f"[cyan]Generating ideas from {len(topics[:count])} trending topics...[/cyan]\n")

    ideas = generator.generate_from_trending(topics[:count], style=style)

    if ideas:
        generator.print_ideas(ideas)
        console.print(f"\n[green]Generated {len(ideas)} ideas from trending[/green]")
    else:
        console.print("[red]Failed to generate ideas[/red]")
        raise typer.Exit(1)


# ============== Thumbnail Commands ==============

@app.command("thumbnail")
def thumbnail_cmd(
    text: str = typer.Argument(..., help="Text for the thumbnail"),
    style: str = typer.Option("bold", "--style", "-s", help="Thumbnail style"),
    background: Path = typer.Option(None, "--background", "-b", help="Background image"),
    output: str = typer.Option(None, "--output", "-o", help="Output filename"),
):
    """Generate a thumbnail image"""
    from .images.thumbnail import ThumbnailGenerator

    gen = ThumbnailGenerator()
    console.print(f"[cyan]Generating thumbnail with style: {style}[/cyan]")

    result = gen.generate(
        text=text,
        style=style,
        background_image=background,
        output_name=output,
    )

    if result:
        console.print(f"[green]Thumbnail saved: {result}[/green]")
    else:
        console.print("[red]Failed to generate thumbnail[/red]")
        raise typer.Exit(1)


@app.command("thumbnail-variants")
def thumbnail_variants_cmd(
    text: str = typer.Argument(..., help="Text for the thumbnails"),
    styles: str = typer.Option("bold,minimal,viral", "--styles", "-s", help="Comma-separated styles"),
    background: Path = typer.Option(None, "--background", "-b", help="Background image"),
):
    """Generate multiple thumbnail variants for A/B testing"""
    from .images.thumbnail import ThumbnailGenerator

    gen = ThumbnailGenerator()
    style_list = [s.strip() for s in styles.split(",")]

    console.print(f"[cyan]Generating {len(style_list)} thumbnail variants...[/cyan]")

    results = gen.generate_variants(
        text=text,
        styles=style_list,
        background_image=background,
    )

    if results:
        console.print(f"\n[green]Generated {len(results)} variants:[/green]")
        for path in results:
            console.print(f"  • {path}")
    else:
        console.print("[red]Failed to generate thumbnails[/red]")
        raise typer.Exit(1)


@app.command("thumbnail-styles")
def thumbnail_styles_cmd():
    """List available thumbnail styles"""
    from .images.thumbnail import ThumbnailGenerator
    ThumbnailGenerator().print_styles()


# ============== A/B Testing Commands ==============

@app.command("ab-create")
def ab_create_cmd(
    video_id: str = typer.Argument(..., help="Video ID to test"),
    test_type: str = typer.Option("title", "--type", "-t", help="Test type: title, thumbnail, description"),
    variant_a: str = typer.Option(..., "--a", help="Variant A value"),
    variant_b: str = typer.Option(..., "--b", help="Variant B value"),
    min_impressions: int = typer.Option(1000, "--min", help="Minimum impressions before declaring winner"),
):
    """Create a new A/B test"""
    from .optimization.ab_testing import ABTestManager

    manager = ABTestManager()
    test = manager.create_test(
        video_id=video_id,
        test_type=test_type,
        variant_a_value=variant_a,
        variant_b_value=variant_b,
        min_impressions=min_impressions,
    )

    console.print(f"[green]Created A/B test: {test.id}[/green]")
    manager.print_test(test)


@app.command("ab-list")
def ab_list_cmd(
    status: str = typer.Option(None, "--status", "-s", help="Filter by status"),
):
    """List all A/B tests"""
    from .optimization.ab_testing import ABTestManager
    manager = ABTestManager()
    manager.print_all_tests()


@app.command("ab-status")
def ab_status_cmd(
    test_id: str = typer.Argument(..., help="Test ID"),
):
    """Show A/B test status and results"""
    from .optimization.ab_testing import ABTestManager

    manager = ABTestManager()
    test = manager.get_test(test_id)

    if test:
        manager.print_test(test)
    else:
        console.print(f"[red]Test {test_id} not found[/red]")
        raise typer.Exit(1)


@app.command("ab-end")
def ab_end_cmd(
    test_id: str = typer.Argument(..., help="Test ID to end"),
    winner: str = typer.Option(None, "--winner", "-w", help="Force winner (a or b)"),
):
    """End an A/B test"""
    from .optimization.ab_testing import ABTestManager

    manager = ABTestManager()
    winner_name = f"Variant {winner.upper()}" if winner else None
    manager.end_test(test_id, winner=winner_name)

    console.print(f"[green]Test {test_id} ended[/green]")


@app.command("ab-generate-titles")
def ab_generate_titles_cmd(
    title: str = typer.Argument(..., help="Original title"),
    count: int = typer.Option(3, "--count", "-c", help="Number of variants"),
):
    """Generate title variants for A/B testing"""
    from .optimization.ab_testing import ABTestManager

    manager = ABTestManager()
    variants = manager.generate_title_variants(title, count=count)

    console.print("[bold]Title Variants:[/bold]\n")
    console.print(f"  Original: {title}")
    for i, variant in enumerate(variants, 1):
        console.print(f"  Variant {i}: {variant}")


# ============== Channel Management Commands ==============

@app.command("channels")
def channels_cmd():
    """List all managed channels"""
    from .channels.manager import ChannelManager
    manager = ChannelManager()
    manager.print_all_channels()


@app.command("channel-create")
def channel_create_cmd(
    channel_id: str = typer.Argument(..., help="Unique channel ID"),
    name: str = typer.Option(..., "--name", "-n", help="Channel name"),
    niche: str = typer.Option(..., "--niche", help="Channel niche"),
    language: str = typer.Option("en", "--language", "-l", help="Language code"),
    voice: str = typer.Option("en_US-ryan-medium", "--voice", "-v", help="Piper TTS voice"),
    style: str = typer.Option("educational", "--style", "-s", help="Default content style"),
):
    """Create a new channel configuration"""
    from .channels.manager import ChannelManager

    manager = ChannelManager()
    channel = manager.create_channel(
        channel_id=channel_id,
        name=name,
        niche=niche,
        language=language,
        voice=voice,
        default_style=style,
    )

    console.print(f"[green]Created channel: {channel.id}[/green]")
    manager.print_channel(channel)


@app.command("channel-info")
def channel_info_cmd(
    channel_id: str = typer.Argument(..., help="Channel ID"),
):
    """Show channel details"""
    from .channels.manager import ChannelManager

    manager = ChannelManager()
    channel = manager.get_channel(channel_id)

    if channel:
        manager.print_channel(channel)
    else:
        console.print(f"[red]Channel {channel_id} not found[/red]")
        raise typer.Exit(1)


@app.command("channel-generate")
def channel_generate_cmd(
    channel_id: str = typer.Argument(..., help="Channel ID"),
    topic: str = typer.Argument(..., help="Video topic"),
    upload: bool = typer.Option(False, "--upload", "-u", help="Upload after generation"),
):
    """Generate a video for a specific channel"""
    from .channels.manager import ChannelManager

    manager = ChannelManager()
    result = manager.generate_for_channel(channel_id, topic, upload=upload)

    if result:
        console.print(f"\n[green]Video generated: {result.video_path}[/green]")
    else:
        console.print("[red]Generation failed[/red]")
        raise typer.Exit(1)


@app.command("channel-update")
def channel_update_cmd(
    channel_id: str = typer.Argument(..., help="Channel ID"),
    voice: str = typer.Option(None, "--voice", "-v", help="Update voice"),
    style: str = typer.Option(None, "--style", "-s", help="Update style"),
    auto_upload: bool = typer.Option(None, "--auto-upload", help="Enable/disable auto upload"),
    active: bool = typer.Option(None, "--active", help="Enable/disable channel"),
):
    """Update channel configuration"""
    from .channels.manager import ChannelManager

    manager = ChannelManager()
    updates = {}

    if voice is not None:
        updates["voice"] = voice
    if style is not None:
        updates["default_style"] = style
    if auto_upload is not None:
        updates["auto_upload"] = auto_upload

    if updates:
        manager.update_channel(channel_id, **updates)

    if active is not None:
        manager.set_active(channel_id, active)

    console.print(f"[green]Channel {channel_id} updated[/green]")


@app.command("channel-delete")
def channel_delete_cmd(
    channel_id: str = typer.Argument(..., help="Channel ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a channel configuration"""
    from .channels.manager import ChannelManager

    if not confirm:
        console.print(f"[yellow]Are you sure you want to delete channel '{channel_id}'?[/yellow]")
        console.print("Use --yes to confirm")
        raise typer.Exit(0)

    manager = ChannelManager()
    manager.delete_channel(channel_id)
    console.print(f"[green]Channel {channel_id} deleted[/green]")


@app.command("channel-samples")
def channel_samples_cmd():
    """Create sample channel configurations"""
    from .channels.manager import create_sample_channels
    create_sample_channels()


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
