"""Multi-channel management for YouTube automation"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class ChannelConfig:
    """Configuration for a YouTube channel"""
    # Branding
    name: str
    niche: str
    language: str = "en"
    voice: str = "en_US-ryan-medium"  # Piper voice

    # Content settings
    default_style: str = "educational"
    default_duration: int = 30
    default_template: str | None = None

    # Upload settings
    auto_upload: bool = False
    default_privacy: Literal["public", "private", "unlisted"] = "private"
    upload_schedule: list[str] = field(default_factory=list)  # Cron-like schedule

    # YouTube OAuth
    credentials_file: str | None = None  # Path to channel-specific credentials

    # TikTok settings
    tiktok_enabled: bool = False
    tiktok_session: str | None = None  # Session file path

    # Content modifiers
    intro_text: str = ""
    outro_text: str = ""
    watermark: str | None = None  # Path to watermark image
    default_tags: list[str] = field(default_factory=list)

    # Posting limits
    max_posts_per_day: int = 3
    min_hours_between_posts: int = 4


@dataclass
class Channel:
    """A managed YouTube channel"""
    id: str
    config: ChannelConfig
    created_at: datetime = field(default_factory=datetime.now)
    last_upload: datetime | None = None
    total_videos: int = 0
    active: bool = True


class ChannelManager:
    """
    Manage multiple YouTube channels.

    Features:
    - Configure multiple channels with different niches
    - Channel-specific settings (voice, style, templates)
    - Per-channel YouTube OAuth credentials
    - Content scheduling per channel
    - Cross-posting to TikTok per channel
    """

    def __init__(self, config_dir: Path | str = "config/channels"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.channels: dict[str, Channel] = {}
        self._load_channels()

    def _load_channels(self):
        """Load all channel configurations"""
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    data = json.load(f)

                channel_id = config_file.stem
                config = ChannelConfig(**data.get("config", {}))

                self.channels[channel_id] = Channel(
                    id=channel_id,
                    config=config,
                    created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                    last_upload=datetime.fromisoformat(data["last_upload"]) if data.get("last_upload") else None,
                    total_videos=data.get("total_videos", 0),
                    active=data.get("active", True),
                )
            except Exception as e:
                console.print(f"[yellow]Error loading channel {config_file}: {e}[/yellow]")

    def _save_channel(self, channel: Channel):
        """Save channel configuration"""
        config_file = self.config_dir / f"{channel.id}.json"

        data = {
            "config": {
                "name": channel.config.name,
                "niche": channel.config.niche,
                "language": channel.config.language,
                "voice": channel.config.voice,
                "default_style": channel.config.default_style,
                "default_duration": channel.config.default_duration,
                "default_template": channel.config.default_template,
                "auto_upload": channel.config.auto_upload,
                "default_privacy": channel.config.default_privacy,
                "upload_schedule": channel.config.upload_schedule,
                "credentials_file": channel.config.credentials_file,
                "tiktok_enabled": channel.config.tiktok_enabled,
                "tiktok_session": channel.config.tiktok_session,
                "intro_text": channel.config.intro_text,
                "outro_text": channel.config.outro_text,
                "watermark": channel.config.watermark,
                "default_tags": channel.config.default_tags,
                "max_posts_per_day": channel.config.max_posts_per_day,
                "min_hours_between_posts": channel.config.min_hours_between_posts,
            },
            "created_at": channel.created_at.isoformat(),
            "last_upload": channel.last_upload.isoformat() if channel.last_upload else None,
            "total_videos": channel.total_videos,
            "active": channel.active,
        }

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_channel(
        self,
        channel_id: str,
        name: str,
        niche: str,
        **kwargs,
    ) -> Channel:
        """Create a new channel configuration"""
        if channel_id in self.channels:
            raise ValueError(f"Channel {channel_id} already exists")

        config = ChannelConfig(name=name, niche=niche, **kwargs)
        channel = Channel(id=channel_id, config=config)

        self.channels[channel_id] = channel
        self._save_channel(channel)

        return channel

    def get_channel(self, channel_id: str) -> Channel | None:
        """Get a channel by ID"""
        return self.channels.get(channel_id)

    def list_channels(self, active_only: bool = False) -> list[Channel]:
        """List all channels"""
        channels = list(self.channels.values())
        if active_only:
            channels = [c for c in channels if c.active]
        return sorted(channels, key=lambda c: c.config.name)

    def update_channel(self, channel_id: str, **kwargs):
        """Update channel configuration"""
        channel = self.get_channel(channel_id)
        if not channel:
            raise ValueError(f"Channel {channel_id} not found")

        for key, value in kwargs.items():
            if hasattr(channel.config, key):
                setattr(channel.config, key, value)

        self._save_channel(channel)

    def delete_channel(self, channel_id: str):
        """Delete a channel configuration"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            config_file = self.config_dir / f"{channel_id}.json"
            if config_file.exists():
                config_file.unlink()

    def set_active(self, channel_id: str, active: bool):
        """Enable or disable a channel"""
        channel = self.get_channel(channel_id)
        if channel:
            channel.active = active
            self._save_channel(channel)

    def record_upload(self, channel_id: str):
        """Record that a video was uploaded"""
        channel = self.get_channel(channel_id)
        if channel:
            channel.last_upload = datetime.now()
            channel.total_videos += 1
            self._save_channel(channel)

    def can_upload(self, channel_id: str) -> tuple[bool, str]:
        """Check if a channel can upload (respecting limits)"""
        channel = self.get_channel(channel_id)
        if not channel:
            return False, "Channel not found"

        if not channel.active:
            return False, "Channel is inactive"

        if channel.last_upload:
            hours_since_last = (datetime.now() - channel.last_upload).total_seconds() / 3600
            if hours_since_last < channel.config.min_hours_between_posts:
                wait_time = channel.config.min_hours_between_posts - hours_since_last
                return False, f"Must wait {wait_time:.1f} more hours"

        return True, "OK"

    def get_pipeline_config(self, channel_id: str) -> dict:
        """Get pipeline configuration for a channel"""
        channel = self.get_channel(channel_id)
        if not channel:
            return {}

        return {
            "voice": channel.config.voice,
            "style": channel.config.default_style,
            "duration": channel.config.default_duration,
            "template": channel.config.default_template,
            "tags": channel.config.default_tags,
            "intro": channel.config.intro_text,
            "outro": channel.config.outro_text,
            "watermark": channel.config.watermark,
        }

    def generate_for_channel(
        self,
        channel_id: str,
        topic: str,
        upload: bool = False,
        **kwargs,
    ):
        """Generate a video for a specific channel"""
        channel = self.get_channel(channel_id)
        if not channel:
            console.print(f"[red]Channel {channel_id} not found[/red]")
            return None

        # Get channel-specific config
        config = self.get_pipeline_config(channel_id)
        config.update(kwargs)

        # Apply channel branding
        if channel.config.intro_text:
            topic = f"{channel.config.intro_text} {topic}"

        console.print(f"[cyan]Generating for channel: {channel.config.name}[/cyan]")
        console.print(f"[dim]Niche: {channel.config.niche}, Voice: {channel.config.voice}[/dim]")

        from ..pipeline import Pipeline
        from ..utils.config import load_config

        pipeline_config = load_config()
        # Override TTS voice
        pipeline_config.tts.voice = channel.config.voice

        pipeline = Pipeline(pipeline_config)
        result = pipeline.generate(
            topic=topic,
            style=config.get("style", channel.config.default_style),
            duration=config.get("duration", channel.config.default_duration),
        )

        if result and upload:
            can_upload, reason = self.can_upload(channel_id)
            if can_upload:
                self._upload_video(channel, result)
            else:
                console.print(f"[yellow]Cannot upload: {reason}[/yellow]")

        return result

    def _upload_video(self, channel: Channel, result):
        """Upload video for a channel"""
        from ..upload.youtube import YouTubeUploader

        # Use channel-specific credentials if available
        credentials_file = channel.config.credentials_file or "config/client_secrets.json"

        uploader = YouTubeUploader(credentials_file=credentials_file)

        if not uploader.is_authenticated():
            console.print("[yellow]YouTube not authenticated for this channel[/yellow]")
            return

        # Combine channel tags with video tags
        tags = list(set(channel.config.default_tags + result.metadata.get("tags", [])))

        upload_result = uploader.upload(
            video_path=result.video_path,
            title=result.metadata.get("title", "Untitled"),
            description=result.metadata.get("description", ""),
            tags=tags,
            privacy_status=channel.config.default_privacy,
        )

        if upload_result:
            self.record_upload(channel.id)
            console.print(f"[green]Uploaded to {channel.config.name}: {upload_result.url}[/green]")

            # TikTok cross-post
            if channel.config.tiktok_enabled:
                self._upload_tiktok(channel, result)

    def _upload_tiktok(self, channel: Channel, result):
        """Upload to TikTok for a channel"""
        try:
            from ..upload.tiktok import TikTokUploader

            uploader = TikTokUploader(
                session_file=channel.config.tiktok_session,
            )

            uploader.upload(
                video_path=result.video_path,
                description=result.metadata.get("title", ""),
            )
            console.print(f"[green]Cross-posted to TikTok[/green]")
        except Exception as e:
            console.print(f"[yellow]TikTok upload failed: {e}[/yellow]")

    def print_channel(self, channel: Channel):
        """Print channel details"""
        content = f"""[bold]{channel.config.name}[/bold]
[dim]ID: {channel.id}[/dim]

[cyan]Niche:[/cyan] {channel.config.niche}
[cyan]Language:[/cyan] {channel.config.language}
[cyan]Voice:[/cyan] {channel.config.voice}
[cyan]Style:[/cyan] {channel.config.default_style}
[cyan]Duration:[/cyan] {channel.config.default_duration}s

[cyan]Auto Upload:[/cyan] {'Yes' if channel.config.auto_upload else 'No'}
[cyan]Default Privacy:[/cyan] {channel.config.default_privacy}
[cyan]TikTok:[/cyan] {'Enabled' if channel.config.tiktok_enabled else 'Disabled'}

[cyan]Total Videos:[/cyan] {channel.total_videos}
[cyan]Last Upload:[/cyan] {channel.last_upload.strftime('%Y-%m-%d %H:%M') if channel.last_upload else 'Never'}
[cyan]Status:[/cyan] {'[green]Active[/green]' if channel.active else '[red]Inactive[/red]'}"""

        if channel.config.default_tags:
            content += f"\n[cyan]Tags:[/cyan] {', '.join(channel.config.default_tags)}"

        console.print(Panel(content, title="Channel Details"))

    def print_all_channels(self):
        """Print summary of all channels"""
        channels = self.list_channels()

        if not channels:
            console.print("[dim]No channels configured[/dim]")
            console.print("\nCreate a channel with:")
            console.print("  ytauto channel-create <id> --name 'My Channel' --niche 'tech'")
            return

        table = Table(title="Managed Channels")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Niche")
        table.add_column("Voice")
        table.add_column("Videos")
        table.add_column("Status")

        for channel in channels:
            status = "[green]Active[/green]" if channel.active else "[red]Inactive[/red]"
            table.add_row(
                channel.id,
                channel.config.name,
                channel.config.niche,
                channel.config.voice.split("-")[0] if "-" in channel.config.voice else channel.config.voice,
                str(channel.total_videos),
                status,
            )

        console.print(table)


def create_sample_channels():
    """Create sample channel configurations"""
    manager = ChannelManager()

    # Tech facts channel
    manager.create_channel(
        channel_id="tech_facts",
        name="Amazing Tech Facts",
        niche="tech",
        voice="en_US-ryan-medium",
        default_style="educational",
        default_duration=30,
        default_tags=["tech", "facts", "technology", "shorts"],
    )

    # Spanish motivation channel
    manager.create_channel(
        channel_id="motivacion_es",
        name="Motivación Diaria",
        niche="motivation",
        language="es",
        voice="es_ES-davefx-medium",
        default_style="motivational",
        default_duration=45,
        default_tags=["motivacion", "exito", "superacion", "shorts"],
    )

    # Gaming tips channel
    manager.create_channel(
        channel_id="gaming_tips",
        name="Pro Gaming Tips",
        niche="gaming",
        voice="en_US-lessac-medium",
        default_style="tutorial",
        default_duration=30,
        default_tags=["gaming", "tips", "tricks", "shorts"],
    )

    console.print("[green]Created 3 sample channels[/green]")
    manager.print_all_channels()
