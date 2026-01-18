"""AI-powered content idea generation"""
from dataclasses import dataclass, field
from typing import Literal
from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass
class ContentIdea:
    """A generated content idea"""
    title: str
    hook: str
    outline: list[str]
    target_audience: str
    estimated_duration: int  # seconds
    difficulty: Literal["easy", "medium", "hard"]
    viral_potential: Literal["low", "medium", "high"]
    tags: list[str] = field(default_factory=list)


class IdeaGenerator:
    """
    Generate content ideas using local LLM.

    Features:
    - Generate ideas from trending topics
    - Generate ideas from niche/category
    - Create variations of existing ideas
    - Generate series concepts
    """

    IDEA_PROMPT = """Generate a YouTube Shorts video idea based on the following:

Topic/Niche: {topic}
Style: {style}
Target Duration: {duration} seconds

Respond with a JSON object containing:
{{
    "title": "Catchy video title (max 60 chars)",
    "hook": "Opening line that grabs attention in first 2 seconds",
    "outline": ["Point 1", "Point 2", "Point 3"],
    "target_audience": "Who this video is for",
    "tags": ["tag1", "tag2", "tag3"],
    "viral_potential": "low/medium/high",
    "difficulty": "easy/medium/hard"
}}

Make it engaging, shareable, and optimized for the YouTube Shorts algorithm.
Output only valid JSON, no other text."""

    SERIES_PROMPT = """Generate a YouTube video series concept:

Main Topic: {topic}
Number of Episodes: {episodes}
Style: {style}

For each episode, provide:
{{
    "series_title": "Overall series name",
    "episodes": [
        {{
            "episode_number": 1,
            "title": "Episode title",
            "hook": "Opening hook",
            "key_points": ["Point 1", "Point 2"]
        }}
    ]
}}

Make each episode stand alone while contributing to the series.
Output only valid JSON."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _get_llm(self):
        """Get or create LLM client"""
        if self.llm_client is None:
            from ..llm.ollama import OllamaClient
            self.llm_client = OllamaClient()
        return self.llm_client

    def generate_idea(
        self,
        topic: str,
        style: str = "educational",
        duration: int = 30,
    ) -> ContentIdea | None:
        """Generate a single content idea"""
        llm = self._get_llm()

        if not llm.is_available():
            console.print("[red]LLM not available. Start Ollama first.[/red]")
            return None

        prompt = self.IDEA_PROMPT.format(
            topic=topic,
            style=style,
            duration=duration,
        )

        try:
            response = llm.generate(prompt, temperature=0.8)

            # Parse JSON from response
            import json
            response = response.strip()

            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)

            return ContentIdea(
                title=data.get("title", f"Video about {topic}"),
                hook=data.get("hook", ""),
                outline=data.get("outline", []),
                target_audience=data.get("target_audience", "General audience"),
                estimated_duration=duration,
                difficulty=data.get("difficulty", "medium"),
                viral_potential=data.get("viral_potential", "medium"),
                tags=data.get("tags", []),
            )

        except Exception as e:
            console.print(f"[yellow]Idea generation error: {e}[/yellow]")
            return None

    def generate_ideas_batch(
        self,
        topic: str,
        count: int = 5,
        **kwargs,
    ) -> list[ContentIdea]:
        """Generate multiple content ideas"""
        ideas = []
        variations = [
            f"{topic} - beginner friendly",
            f"{topic} - advanced tips",
            f"{topic} - common mistakes",
            f"{topic} - quick facts",
            f"{topic} - surprising truth",
        ]

        for i, variation in enumerate(variations[:count]):
            console.print(f"[dim]Generating idea {i+1}/{count}...[/dim]")
            idea = self.generate_idea(variation, **kwargs)
            if idea:
                ideas.append(idea)

        return ideas

    def generate_from_trending(
        self,
        topics: list,
        count: int = 5,
        **kwargs,
    ) -> list[ContentIdea]:
        """Generate ideas from trending topics"""
        ideas = []

        for topic in topics[:count]:
            title = topic.title if hasattr(topic, "title") else str(topic)
            console.print(f"[dim]Generating idea for: {title[:50]}...[/dim]")

            idea = self.generate_idea(title, **kwargs)
            if idea:
                ideas.append(idea)

        return ideas

    def generate_series(
        self,
        topic: str,
        episodes: int = 5,
        style: str = "educational",
    ) -> list[ContentIdea]:
        """Generate a series of related content ideas"""
        llm = self._get_llm()

        if not llm.is_available():
            console.print("[red]LLM not available[/red]")
            return []

        prompt = self.SERIES_PROMPT.format(
            topic=topic,
            episodes=episodes,
            style=style,
        )

        try:
            response = llm.generate(prompt, temperature=0.7)

            import json
            response = response.strip()

            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)

            ideas = []
            series_title = data.get("series_title", topic)

            for ep in data.get("episodes", []):
                ideas.append(ContentIdea(
                    title=f"{series_title} #{ep.get('episode_number', 1)}: {ep.get('title', '')}",
                    hook=ep.get("hook", ""),
                    outline=ep.get("key_points", []),
                    target_audience="Series followers",
                    estimated_duration=45,
                    difficulty="medium",
                    viral_potential="medium",
                    tags=[series_title.lower().replace(" ", "")],
                ))

            return ideas

        except Exception as e:
            console.print(f"[yellow]Series generation error: {e}[/yellow]")
            return []

    def print_idea(self, idea: ContentIdea):
        """Print a content idea nicely"""
        content = f"""[bold]{idea.title}[/bold]

[cyan]Hook:[/cyan] {idea.hook}

[cyan]Outline:[/cyan]
{chr(10).join(f"  • {point}" for point in idea.outline)}

[cyan]Target:[/cyan] {idea.target_audience}
[cyan]Duration:[/cyan] {idea.estimated_duration}s
[cyan]Difficulty:[/cyan] {idea.difficulty}
[cyan]Viral Potential:[/cyan] {idea.viral_potential}
[cyan]Tags:[/cyan] {', '.join(idea.tags)}"""

        console.print(Panel(content, title="Content Idea"))

    def print_ideas(self, ideas: list[ContentIdea]):
        """Print multiple ideas"""
        for i, idea in enumerate(ideas, 1):
            console.print(f"\n[bold]Idea #{i}[/bold]")
            self.print_idea(idea)
