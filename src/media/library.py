"""Local media library management"""
import random
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class MediaAsset:
    """A media asset from the library"""
    path: Path
    asset_type: str  # "image", "video", "music"
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MediaLibrary:
    """Manages local media assets for video generation"""

    SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    SUPPORTED_VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}

    def __init__(
        self,
        images_dir: Path | str = "assets/images",
        footage_dir: Path | str = "assets/footage",
        music_dir: Path | str = "assets/music",
    ):
        self.images_dir = Path(images_dir)
        self.footage_dir = Path(footage_dir)
        self.music_dir = Path(music_dir)

        # Create directories if they don't exist
        for dir_path in [self.images_dir, self.footage_dir, self.music_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self._image_cache: list[MediaAsset] = []
        self._footage_cache: list[MediaAsset] = []
        self._music_cache: list[MediaAsset] = []

        self.refresh()

    def refresh(self) -> None:
        """Refresh the media library cache"""
        self._image_cache = self._scan_directory(self.images_dir, "image", self.SUPPORTED_IMAGES)
        self._footage_cache = self._scan_directory(self.footage_dir, "video", self.SUPPORTED_VIDEOS)
        self._music_cache = self._scan_directory(self.music_dir, "music", self.SUPPORTED_AUDIO)

    def _scan_directory(
        self,
        directory: Path,
        asset_type: str,
        extensions: set[str],
    ) -> list[MediaAsset]:
        """Scan directory for media files"""
        assets = []

        if not directory.exists():
            return assets

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Extract tags from path (subdirectories become tags)
                relative = file_path.relative_to(directory)
                tags = [p for p in relative.parts[:-1]]  # Exclude filename
                tags.append(file_path.stem.lower())  # Add filename as tag

                assets.append(MediaAsset(
                    path=file_path,
                    asset_type=asset_type,
                    tags=tags,
                ))

        return assets

    @property
    def images(self) -> list[MediaAsset]:
        """Get all images"""
        return self._image_cache

    @property
    def footage(self) -> list[MediaAsset]:
        """Get all video footage"""
        return self._footage_cache

    @property
    def music(self) -> list[MediaAsset]:
        """Get all music tracks"""
        return self._music_cache

    def get_random_image(self, tags: list[str] | None = None) -> MediaAsset | None:
        """Get a random image, optionally filtered by tags"""
        return self._get_random_asset(self._image_cache, tags)

    def get_random_footage(self, tags: list[str] | None = None) -> MediaAsset | None:
        """Get random video footage, optionally filtered by tags"""
        return self._get_random_asset(self._footage_cache, tags)

    def get_random_music(self, tags: list[str] | None = None) -> MediaAsset | None:
        """Get random music track, optionally filtered by tags"""
        return self._get_random_asset(self._music_cache, tags)

    def _get_random_asset(
        self,
        assets: list[MediaAsset],
        tags: list[str] | None = None,
    ) -> MediaAsset | None:
        """Get random asset from list, optionally filtered by tags"""
        if not assets:
            return None

        if tags:
            # Filter by tags (any match)
            tags_lower = [t.lower() for t in tags]
            filtered = [
                a for a in assets
                if any(t.lower() in a.tags for t in tags_lower)
            ]
            if filtered:
                return random.choice(filtered)

        return random.choice(assets)

    def get_images_for_segments(
        self,
        num_segments: int,
        tags: list[str] | None = None,
    ) -> list[MediaAsset]:
        """Get images for multiple segments, avoiding duplicates if possible"""
        available = self._image_cache.copy()

        if tags:
            tags_lower = [t.lower() for t in tags]
            available = [
                a for a in available
                if any(t.lower() in a.tags for t in tags_lower)
            ]

        if not available:
            available = self._image_cache.copy()

        result = []
        for _ in range(num_segments):
            if available:
                asset = random.choice(available)
                result.append(asset)
                # Try to avoid duplicates
                if len(available) > 1:
                    available.remove(asset)
            elif self._image_cache:
                # Fall back to any image
                result.append(random.choice(self._image_cache))

        return result

    def stats(self) -> dict:
        """Get library statistics"""
        return {
            "images": len(self._image_cache),
            "footage": len(self._footage_cache),
            "music": len(self._music_cache),
            "total": len(self._image_cache) + len(self._footage_cache) + len(self._music_cache),
        }

    def print_stats(self) -> None:
        """Print library statistics"""
        stats = self.stats()
        console.print("\n[bold]Media Library Statistics[/bold]")
        console.print(f"  Images:  {stats['images']}")
        console.print(f"  Footage: {stats['footage']}")
        console.print(f"  Music:   {stats['music']}")
        console.print(f"  [bold]Total:   {stats['total']}[/bold]\n")


def download_sample_assets(library: MediaLibrary) -> None:
    """Download some sample placeholder images for testing"""
    import httpx

    console.print("[cyan]Creating sample placeholder images...[/cyan]")

    # Create simple colored placeholder images using PIL
    try:
        from PIL import Image, ImageDraw, ImageFont

        colors = [
            ((41, 128, 185), "Blue"),
            ((39, 174, 96), "Green"),
            ((192, 57, 43), "Red"),
            ((142, 68, 173), "Purple"),
            ((243, 156, 18), "Orange"),
        ]

        for color, name in colors:
            img = Image.new("RGB", (1920, 1080), color)
            draw = ImageDraw.Draw(img)

            # Add text
            text = f"Sample {name}"
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 72)
            except OSError:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (1920 - text_width) // 2
            y = (1080 - text_height) // 2

            draw.text((x, y), text, fill="white", font=font)

            output_path = library.images_dir / f"sample_{name.lower()}.png"
            img.save(output_path)
            console.print(f"  Created: {output_path.name}")

        library.refresh()
        console.print("[green]Sample images created![/green]")

    except ImportError:
        console.print("[yellow]PIL not available - skipping sample images[/yellow]")
