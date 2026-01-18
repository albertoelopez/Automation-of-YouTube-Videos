"""Video template management system"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class SubtitleStyle:
    """Subtitle styling configuration"""
    enabled: bool = True
    font_size: int = 60
    color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    position: Literal["top", "center", "bottom"] = "bottom"
    animation: Literal["none", "fade", "typewriter", "highlight"] = "none"


@dataclass
class VideoStyle:
    """Video styling configuration"""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    background_color: str = "#1a1a2e"
    enable_ken_burns: bool = True
    ken_burns_zoom: float = 1.2
    transition: Literal["none", "fade", "slide", "zoom"] = "fade"
    transition_duration: float = 0.5


@dataclass
class AudioStyle:
    """Audio styling configuration"""
    voice: str = "en_US-lessac-medium"
    voice_speed: float = 1.0
    background_music: bool = True
    music_volume: float = 0.3
    silence_padding: float = 0.3


@dataclass
class ContentStyle:
    """Content generation configuration"""
    style: str = "informative"  # informative, entertaining, educational, dramatic
    tone: str = "professional"  # casual, professional, friendly, authoritative
    hook_style: str = "question"  # question, statistic, story, bold_claim
    cta_style: str = "subscribe"  # subscribe, like, comment, follow


@dataclass
class VideoTemplate:
    """Complete video template configuration"""
    name: str
    description: str = ""
    category: str = "general"

    video: VideoStyle = field(default_factory=VideoStyle)
    audio: AudioStyle = field(default_factory=AudioStyle)
    subtitles: SubtitleStyle = field(default_factory=SubtitleStyle)
    content: ContentStyle = field(default_factory=ContentStyle)

    # LLM prompt customization
    system_prompt: str = ""
    example_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoTemplate":
        """Create from dictionary"""
        # Handle nested dataclasses
        if "video" in data and isinstance(data["video"], dict):
            data["video"] = VideoStyle(**data["video"])
        if "audio" in data and isinstance(data["audio"], dict):
            data["audio"] = AudioStyle(**data["audio"])
        if "subtitles" in data and isinstance(data["subtitles"], dict):
            data["subtitles"] = SubtitleStyle(**data["subtitles"])
        if "content" in data and isinstance(data["content"], dict):
            data["content"] = ContentStyle(**data["content"])
        return cls(**data)


# Built-in templates
BUILTIN_TEMPLATES = {
    "facts": VideoTemplate(
        name="facts",
        description="Educational facts videos with engaging hooks",
        category="educational",
        content=ContentStyle(
            style="educational",
            tone="friendly",
            hook_style="statistic",
            cta_style="subscribe",
        ),
        audio=AudioStyle(
            voice="en_US-lessac-medium",
            voice_speed=1.0,
        ),
        subtitles=SubtitleStyle(
            font_size=65,
            animation="highlight",
        ),
        system_prompt="""You are creating a viral facts video.
Start with a mind-blowing statistic or fact as a hook.
Present 3-5 interesting facts, each building on the previous.
End with the most surprising fact and a call to action.""",
        example_topics=[
            "5 Mind-Blowing Facts About the Ocean",
            "Things You Didn't Know About Your Brain",
        ],
    ),

    "tutorial": VideoTemplate(
        name="tutorial",
        description="Step-by-step tutorial videos",
        category="educational",
        content=ContentStyle(
            style="educational",
            tone="professional",
            hook_style="bold_claim",
            cta_style="like",
        ),
        audio=AudioStyle(
            voice="en_US-ryan-medium",
            voice_speed=0.95,
        ),
        subtitles=SubtitleStyle(
            font_size=55,
            position="bottom",
        ),
        system_prompt="""You are creating a tutorial video.
Start by stating what viewers will learn.
Break down the process into clear, numbered steps.
Keep each step concise and actionable.
End with a summary and encouragement.""",
        example_topics=[
            "How to Start a YouTube Channel",
            "5 Productivity Hacks for Beginners",
        ],
    ),

    "motivation": VideoTemplate(
        name="motivation",
        description="Motivational and inspirational content",
        category="lifestyle",
        video=VideoStyle(
            background_color="#0f0f23",
            ken_burns_zoom=1.3,
        ),
        content=ContentStyle(
            style="dramatic",
            tone="authoritative",
            hook_style="story",
            cta_style="follow",
        ),
        audio=AudioStyle(
            voice="en_GB-alan-medium",
            voice_speed=0.9,
            music_volume=0.4,
        ),
        subtitles=SubtitleStyle(
            font_size=70,
            color="yellow",
            animation="fade",
        ),
        system_prompt="""You are creating a motivational video.
Start with a powerful statement or question.
Build emotional connection through relatable struggles.
Pivot to empowerment and actionable advice.
End with an inspiring call to action.""",
        example_topics=[
            "Why Most People Give Up (And How Not To)",
            "The One Thing Successful People Do Daily",
        ],
    ),

    "storytime": VideoTemplate(
        name="storytime",
        description="Narrative storytelling format",
        category="entertainment",
        content=ContentStyle(
            style="entertaining",
            tone="casual",
            hook_style="story",
            cta_style="comment",
        ),
        audio=AudioStyle(
            voice="en_US-amy-medium",
            voice_speed=1.0,
        ),
        subtitles=SubtitleStyle(
            font_size=60,
            animation="typewriter",
        ),
        system_prompt="""You are telling a compelling story.
Start with an intriguing hook that creates curiosity.
Build tension and engagement throughout.
Include vivid details and emotional moments.
End with a satisfying conclusion or cliffhanger.""",
        example_topics=[
            "The Craziest Thing That Happened at Work",
            "How I Almost Got Scammed",
        ],
    ),

    "news": VideoTemplate(
        name="news",
        description="News and current events coverage",
        category="news",
        video=VideoStyle(
            background_color="#1a1a2e",
            transition="slide",
        ),
        content=ContentStyle(
            style="informative",
            tone="professional",
            hook_style="bold_claim",
            cta_style="subscribe",
        ),
        audio=AudioStyle(
            voice="en_US-joe-medium",
            voice_speed=1.05,
            music_volume=0.2,
        ),
        subtitles=SubtitleStyle(
            font_size=55,
            color="white",
        ),
        system_prompt="""You are reporting news in a concise format.
Lead with the most important information.
Provide context and key details.
Present multiple perspectives if relevant.
End with implications or what to watch for.""",
        example_topics=[
            "Breaking: Major Tech Announcement",
            "This Week in AI News",
        ],
    ),

    "listicle": VideoTemplate(
        name="listicle",
        description="Numbered list format (Top 5, Top 10)",
        category="general",
        content=ContentStyle(
            style="entertaining",
            tone="friendly",
            hook_style="question",
            cta_style="like",
        ),
        audio=AudioStyle(
            voice="en_US-lessac-medium",
            voice_speed=1.0,
        ),
        subtitles=SubtitleStyle(
            font_size=65,
            animation="highlight",
        ),
        system_prompt="""You are creating a listicle video.
Start with a compelling hook about the list.
Number each item clearly.
Make each item punchy and memorable.
Save the best for last.
End with engagement call to action.""",
        example_topics=[
            "Top 5 Apps You Need in 2024",
            "10 Things Rich People Never Buy",
        ],
    ),

    "shorts_viral": VideoTemplate(
        name="shorts_viral",
        description="Optimized for YouTube Shorts virality",
        category="shorts",
        video=VideoStyle(
            width=1080,
            height=1920,
            fps=30,
            ken_burns_zoom=1.15,
            transition_duration=0.3,
        ),
        content=ContentStyle(
            style="entertaining",
            tone="casual",
            hook_style="bold_claim",
            cta_style="follow",
        ),
        audio=AudioStyle(
            voice="en_US-lessac-medium",
            voice_speed=1.1,  # Slightly faster for shorts
            music_volume=0.35,
        ),
        subtitles=SubtitleStyle(
            font_size=70,
            color="white",
            stroke_width=4,
            animation="highlight",
        ),
        system_prompt="""You are creating a viral YouTube Short.
HOOK: First 2 seconds must stop the scroll.
CONTENT: Get to the point immediately.
PACING: Keep it fast, no filler words.
CTA: End with engagement prompt.
Total: 30-45 seconds max.""",
        example_topics=[
            "The $1 Trick That Saves Thousands",
            "POV: You Finally Understand This",
        ],
    ),

    "spanish_facts": VideoTemplate(
        name="spanish_facts",
        description="Facts videos in Spanish",
        category="educational",
        content=ContentStyle(
            style="educational",
            tone="friendly",
            hook_style="statistic",
            cta_style="subscribe",
        ),
        audio=AudioStyle(
            voice="es_MX-ald-medium",
            voice_speed=1.0,
        ),
        subtitles=SubtitleStyle(
            font_size=60,
        ),
        system_prompt="""Estás creando un video de datos interesantes.
Comienza con un dato impactante.
Presenta 3-5 hechos interesantes.
Termina con el dato más sorprendente.
Incluye un llamado a la acción.""",
        example_topics=[
            "5 Datos Increíbles Sobre el Espacio",
            "Cosas que No Sabías de México",
        ],
    ),
}


class TemplateManager:
    """Manages video templates"""

    def __init__(self, templates_dir: Path | str = "config/templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, VideoTemplate] = {}
        self._load_templates()

    def _load_templates(self):
        """Load all templates (builtin + custom)"""
        # Load builtin templates
        self._templates.update(BUILTIN_TEMPLATES)

        # Load custom templates from directory
        for file_path in self.templates_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                template = VideoTemplate.from_dict(data)
                self._templates[template.name] = template
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load template {file_path}: {e}[/yellow]")

    def get(self, name: str) -> VideoTemplate | None:
        """Get a template by name"""
        return self._templates.get(name)

    def list(self) -> list[VideoTemplate]:
        """List all available templates"""
        return list(self._templates.values())

    def list_by_category(self, category: str) -> list[VideoTemplate]:
        """List templates by category"""
        return [t for t in self._templates.values() if t.category == category]

    def save(self, template: VideoTemplate) -> Path:
        """Save a custom template"""
        file_path = self.templates_dir / f"{template.name}.json"
        with open(file_path, "w") as f:
            json.dump(template.to_dict(), f, indent=2)
        self._templates[template.name] = template
        return file_path

    def delete(self, name: str) -> bool:
        """Delete a custom template"""
        if name in BUILTIN_TEMPLATES:
            console.print(f"[red]Cannot delete builtin template: {name}[/red]")
            return False

        file_path = self.templates_dir / f"{name}.json"
        if file_path.exists():
            file_path.unlink()

        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def print_list(self):
        """Print formatted list of templates"""
        table = Table(title="Available Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Category")
        table.add_column("Description")
        table.add_column("Voice")

        for template in sorted(self._templates.values(), key=lambda t: t.category):
            builtin = " (builtin)" if template.name in BUILTIN_TEMPLATES else ""
            table.add_row(
                template.name + builtin,
                template.category,
                template.description[:40] + "..." if len(template.description) > 40 else template.description,
                template.audio.voice,
            )

        console.print(table)


def load_template(name: str) -> VideoTemplate | None:
    """Convenience function to load a template"""
    manager = TemplateManager()
    return manager.get(name)
