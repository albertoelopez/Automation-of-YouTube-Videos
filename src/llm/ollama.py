"""Ollama client for local LLM inference"""
import httpx
from typing import Generator
from pydantic import BaseModel
from rich.console import Console

console = Console()


class ScriptSegment(BaseModel):
    """A segment of the video script"""
    text: str
    duration_hint: float | None = None  # Suggested duration in seconds


class VideoScript(BaseModel):
    """Complete video script"""
    title: str
    description: str
    segments: list[ScriptSegment]
    hashtags: list[str] = []


class OllamaClient:
    """Client for Ollama local LLM"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=120.0)

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.ConnectError:
            return False

    def list_models(self) -> list[str]:
        """List available models"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate text from prompt"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        response = self.client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        return response.json()["response"]

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Generate text with streaming"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }

        if system:
            payload["system"] = system

        with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
        ) as response:
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]


SCRIPT_SYSTEM_PROMPT = """You are a YouTube Shorts script writer. Create engaging, concise scripts for vertical short-form videos (30-60 seconds).

Rules:
- Write in a conversational, engaging tone
- Each segment should be 1-2 sentences max
- Start with a hook that grabs attention
- Include a call-to-action at the end
- Keep total script under 150 words for a 60-second video
- Output valid JSON only

Format your response as JSON:
{
    "title": "Video title (max 100 chars)",
    "description": "YouTube description (2-3 sentences)",
    "segments": [
        {"text": "Hook - grabbing opening line", "duration_hint": 3},
        {"text": "Main point 1", "duration_hint": 5},
        {"text": "Main point 2", "duration_hint": 5},
        {"text": "Call to action", "duration_hint": 3}
    ],
    "hashtags": ["relevant", "hashtags", "here"]
}"""


def generate_script(
    client: OllamaClient,
    topic: str,
    style: str = "informative",
    duration: int = 30,
) -> VideoScript:
    """Generate a video script for a given topic"""

    prompt = f"""Create a {duration}-second YouTube Shorts script about: {topic}

Style: {style}
Target duration: {duration} seconds

Remember to output valid JSON only, no other text."""

    response = client.generate(
        prompt=prompt,
        system=SCRIPT_SYSTEM_PROMPT,
        temperature=0.7,
    )

    # Parse JSON from response
    import json

    # Try to extract JSON from response
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]

    try:
        data = json.loads(response)
        return VideoScript(**data)
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Warning: Could not parse LLM response as JSON: {e}[/yellow]")
        console.print(f"[dim]Response was: {response[:500]}...[/dim]")

        # Return a fallback script
        return VideoScript(
            title=f"Video about {topic}",
            description=f"Learn about {topic} in this short video.",
            segments=[
                ScriptSegment(text=f"Let's talk about {topic}.", duration_hint=3),
                ScriptSegment(text="Here's what you need to know.", duration_hint=5),
                ScriptSegment(text="Follow for more content like this!", duration_hint=3),
            ],
            hashtags=[topic.replace(" ", "").lower(), "shorts", "viral"]
        )
