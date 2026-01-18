"""Local image generation using Stable Diffusion or Pollinations (fallback)"""
import httpx
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from rich.console import Console

console = Console()


@dataclass
class GeneratedImage:
    """A generated image"""
    path: Path
    prompt: str
    width: int
    height: int


class ImageGenerator:
    """
    Image generator with multiple backends:
    1. Local Stable Diffusion (via diffusers) - requires GPU
    2. Pollinations.ai (free, no API key) - fallback
    """

    def __init__(
        self,
        output_dir: Path | str = "assets/images/generated",
        backend: Literal["auto", "diffusers", "pollinations"] = "auto",
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.model = model
        self._diffusers_pipe = None
        self._backend_available = None

    def _check_diffusers(self) -> bool:
        """Check if diffusers with CUDA is available"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            import diffusers
            return True
        except ImportError:
            return False

    def _get_backend(self) -> str:
        """Determine which backend to use"""
        if self.backend != "auto":
            return self.backend

        if self._backend_available is None:
            self._backend_available = "diffusers" if self._check_diffusers() else "pollinations"

        return self._backend_available

    def _init_diffusers(self):
        """Initialize Stable Diffusion pipeline"""
        if self._diffusers_pipe is not None:
            return

        console.print("[cyan]Loading Stable Diffusion model (this may take a while)...[/cyan]")

        import torch
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

        self._diffusers_pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self._diffusers_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._diffusers_pipe.scheduler.config
        )
        self._diffusers_pipe = self._diffusers_pipe.to("cuda")

        # Enable memory optimizations
        self._diffusers_pipe.enable_attention_slicing()

        console.print("[green]Model loaded![/green]")

    def generate(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> GeneratedImage | None:
        """Generate an image from a text prompt"""
        backend = self._get_backend()

        if backend == "diffusers":
            return self._generate_diffusers(
                prompt, width, height, negative_prompt, steps, guidance_scale, seed
            )
        else:
            return self._generate_pollinations(prompt, width, height, seed)

    def _generate_diffusers(
        self,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedImage | None:
        """Generate using local Stable Diffusion"""
        try:
            import torch

            self._init_diffusers()

            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed)

            console.print(f"[dim]Generating: {prompt[:50]}...[/dim]")

            image = self._diffusers_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            # Save image
            filename = self._get_filename(prompt)
            output_path = self.output_dir / filename
            image.save(output_path)

            return GeneratedImage(
                path=output_path,
                prompt=prompt,
                width=width,
                height=height,
            )

        except Exception as e:
            console.print(f"[red]Diffusers error: {e}[/red]")
            console.print("[yellow]Falling back to Pollinations...[/yellow]")
            return self._generate_pollinations(prompt, width, height, seed)

    def _generate_pollinations(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int | None,
    ) -> GeneratedImage | None:
        """
        Generate using Pollinations.ai (free, no API key)
        https://pollinations.ai/
        """
        try:
            import urllib.parse

            # Build URL
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

            params = {
                "width": width,
                "height": height,
                "nologo": "true",
            }
            if seed is not None:
                params["seed"] = seed

            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{param_str}"

            console.print(f"[dim]Generating via Pollinations: {prompt[:50]}...[/dim]")

            # Download image
            with httpx.Client(timeout=120.0) as client:
                response = client.get(full_url, follow_redirects=True)
                response.raise_for_status()

                # Save image
                filename = self._get_filename(prompt)
                output_path = self.output_dir / filename

                with open(output_path, "wb") as f:
                    f.write(response.content)

                return GeneratedImage(
                    path=output_path,
                    prompt=prompt,
                    width=width,
                    height=height,
                )

        except Exception as e:
            console.print(f"[red]Pollinations error: {e}[/red]")
            return None

    def _get_filename(self, prompt: str) -> str:
        """Generate unique filename from prompt"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
        return f"{safe_prompt}_{prompt_hash}.png"

    def generate_batch(
        self,
        prompts: list[str],
        **kwargs,
    ) -> list[GeneratedImage]:
        """Generate multiple images"""
        results = []
        for i, prompt in enumerate(prompts):
            console.print(f"[cyan]Generating image {i+1}/{len(prompts)}[/cyan]")
            result = self.generate(prompt, **kwargs)
            if result:
                results.append(result)
        return results

    def generate_for_script(
        self,
        segments: list[dict],
        style: str = "cinematic, professional, high quality",
        **kwargs,
    ) -> list[GeneratedImage]:
        """Generate images for video script segments"""
        prompts = []
        for segment in segments:
            text = segment.get("text", str(segment))
            # Create visual prompt from segment text
            prompt = f"{text}, {style}"
            prompts.append(prompt)

        return self.generate_batch(prompts, **kwargs)


def generate_image(
    prompt: str,
    output_dir: Path | str = "assets/images/generated",
    **kwargs,
) -> GeneratedImage | None:
    """Convenience function to generate a single image"""
    generator = ImageGenerator(output_dir=output_dir)
    return generator.generate(prompt, **kwargs)
