"""Thumbnail generator for YouTube videos"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from rich.console import Console
import math

console = Console()

# YouTube thumbnail dimensions
THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720


@dataclass
class ThumbnailStyle:
    """Thumbnail style configuration"""
    name: str
    background_color: tuple[int, int, int] = (20, 20, 20)
    text_color: tuple[int, int, int] = (255, 255, 255)
    accent_color: tuple[int, int, int] = (255, 0, 0)
    font_size: int = 120
    font_family: str = "DejaVuSans-Bold"
    text_position: Literal["center", "bottom", "top"] = "center"
    overlay_opacity: float = 0.6
    add_border: bool = True
    border_width: int = 8
    text_shadow: bool = True
    gradient_overlay: bool = True


# Predefined styles
THUMBNAIL_STYLES = {
    "bold": ThumbnailStyle(
        name="bold",
        background_color=(20, 20, 20),
        text_color=(255, 255, 255),
        accent_color=(255, 0, 0),
        font_size=140,
        text_position="center",
    ),
    "minimal": ThumbnailStyle(
        name="minimal",
        background_color=(255, 255, 255),
        text_color=(20, 20, 20),
        accent_color=(0, 120, 255),
        font_size=100,
        add_border=False,
        text_shadow=False,
        gradient_overlay=False,
    ),
    "gaming": ThumbnailStyle(
        name="gaming",
        background_color=(15, 15, 35),
        text_color=(0, 255, 200),
        accent_color=(255, 0, 100),
        font_size=130,
        text_position="bottom",
    ),
    "news": ThumbnailStyle(
        name="news",
        background_color=(30, 30, 30),
        text_color=(255, 255, 255),
        accent_color=(255, 200, 0),
        font_size=110,
        text_position="bottom",
        border_width=12,
    ),
    "tutorial": ThumbnailStyle(
        name="tutorial",
        background_color=(0, 50, 100),
        text_color=(255, 255, 255),
        accent_color=(100, 200, 255),
        font_size=120,
        text_position="center",
    ),
    "viral": ThumbnailStyle(
        name="viral",
        background_color=(0, 0, 0),
        text_color=(255, 255, 0),
        accent_color=(255, 0, 0),
        font_size=150,
        text_position="center",
        border_width=15,
    ),
}


class ThumbnailGenerator:
    """
    Generate eye-catching thumbnails for YouTube videos.

    Features:
    - Multiple predefined styles
    - Custom backgrounds (solid color, gradient, image)
    - Text with shadows and outlines
    - Emoji support
    - Face detection placement (if image provided)
    - A/B testing variants
    """

    def __init__(self, output_dir: Path | str = "output/thumbnails"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        text: str,
        style: str | ThumbnailStyle = "bold",
        background_image: Path | str | None = None,
        output_name: str | None = None,
        emoji: str | None = None,
    ) -> Path | None:
        """Generate a thumbnail"""
        if isinstance(style, str):
            style = THUMBNAIL_STYLES.get(style, THUMBNAIL_STYLES["bold"])

        # Create base image
        if background_image:
            img = self._load_background(background_image)
        else:
            img = self._create_gradient_background(style)

        # Apply overlay if using background image
        if background_image and style.gradient_overlay:
            img = self._apply_gradient_overlay(img, style.overlay_opacity)

        # Add border
        if style.add_border:
            img = self._add_border(img, style.accent_color, style.border_width)

        # Add text
        img = self._add_text(img, text, style)

        # Add emoji if provided
        if emoji:
            img = self._add_emoji(img, emoji)

        # Save
        if output_name is None:
            import uuid
            output_name = f"thumb_{uuid.uuid4().hex[:8]}"

        output_path = self.output_dir / f"{output_name}.png"
        img.save(output_path, "PNG", quality=95)

        return output_path

    def _load_background(self, image_path: Path | str) -> Image.Image:
        """Load and resize background image"""
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to cover thumbnail dimensions
        img_ratio = img.width / img.height
        thumb_ratio = THUMBNAIL_WIDTH / THUMBNAIL_HEIGHT

        if img_ratio > thumb_ratio:
            # Image is wider, fit by height
            new_height = THUMBNAIL_HEIGHT
            new_width = int(new_height * img_ratio)
        else:
            # Image is taller, fit by width
            new_width = THUMBNAIL_WIDTH
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - THUMBNAIL_WIDTH) // 2
        top = (new_height - THUMBNAIL_HEIGHT) // 2
        img = img.crop((left, top, left + THUMBNAIL_WIDTH, top + THUMBNAIL_HEIGHT))

        return img

    def _create_gradient_background(self, style: ThumbnailStyle) -> Image.Image:
        """Create a gradient background"""
        img = Image.new("RGB", (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
        draw = ImageDraw.Draw(img)

        # Create vertical gradient
        r1, g1, b1 = style.background_color
        r2, g2, b2 = style.accent_color

        for y in range(THUMBNAIL_HEIGHT):
            ratio = y / THUMBNAIL_HEIGHT
            r = int(r1 + (r2 - r1) * ratio * 0.3)
            g = int(g1 + (g2 - g1) * ratio * 0.3)
            b = int(b1 + (b2 - b1) * ratio * 0.3)
            draw.line([(0, y), (THUMBNAIL_WIDTH, y)], fill=(r, g, b))

        return img

    def _apply_gradient_overlay(
        self,
        img: Image.Image,
        opacity: float = 0.6,
    ) -> Image.Image:
        """Apply a dark gradient overlay for text readability"""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Bottom-heavy gradient
        for y in range(THUMBNAIL_HEIGHT):
            ratio = y / THUMBNAIL_HEIGHT
            alpha = int(255 * opacity * (0.3 + ratio * 0.7))
            draw.line([(0, y), (THUMBNAIL_WIDTH, y)], fill=(0, 0, 0, alpha))

        # Convert image to RGBA and composite
        img = img.convert("RGBA")
        return Image.alpha_composite(img, overlay).convert("RGB")

    def _add_border(
        self,
        img: Image.Image,
        color: tuple[int, int, int],
        width: int,
    ) -> Image.Image:
        """Add a colored border"""
        draw = ImageDraw.Draw(img)

        # Draw border rectangles
        draw.rectangle(
            [0, 0, THUMBNAIL_WIDTH - 1, width - 1],
            fill=color
        )
        draw.rectangle(
            [0, THUMBNAIL_HEIGHT - width, THUMBNAIL_WIDTH - 1, THUMBNAIL_HEIGHT - 1],
            fill=color
        )
        draw.rectangle(
            [0, 0, width - 1, THUMBNAIL_HEIGHT - 1],
            fill=color
        )
        draw.rectangle(
            [THUMBNAIL_WIDTH - width, 0, THUMBNAIL_WIDTH - 1, THUMBNAIL_HEIGHT - 1],
            fill=color
        )

        return img

    def _get_font(self, size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
        """Get a font, falling back to default if not found"""
        font_names = [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "arial.ttf",
            "Arial.ttf",
            "Helvetica.ttf",
        ]

        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                continue

        # Fallback to default font
        return ImageFont.load_default()

    def _add_text(
        self,
        img: Image.Image,
        text: str,
        style: ThumbnailStyle,
    ) -> Image.Image:
        """Add text to thumbnail"""
        draw = ImageDraw.Draw(img)
        font = self._get_font(style.font_size)

        # Word wrap text
        lines = self._wrap_text(text, font, THUMBNAIL_WIDTH - 100)

        # Calculate total text height
        line_height = style.font_size + 10
        total_height = len(lines) * line_height

        # Determine Y position based on style
        if style.text_position == "center":
            start_y = (THUMBNAIL_HEIGHT - total_height) // 2
        elif style.text_position == "bottom":
            start_y = THUMBNAIL_HEIGHT - total_height - 60
        else:  # top
            start_y = 60

        # Draw each line
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (THUMBNAIL_WIDTH - text_width) // 2
            y = start_y + i * line_height

            # Draw shadow
            if style.text_shadow:
                shadow_offset = max(4, style.font_size // 20)
                draw.text(
                    (x + shadow_offset, y + shadow_offset),
                    line,
                    font=font,
                    fill=(0, 0, 0),
                )

            # Draw text outline
            outline_width = max(2, style.font_size // 30)
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))

            # Draw main text
            draw.text((x, y), line, font=font, fill=style.text_color)

        return img

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines[:3]  # Max 3 lines

    def _add_emoji(self, img: Image.Image, emoji: str) -> Image.Image:
        """Add an emoji to the corner"""
        # This is a simplified version - full emoji support requires
        # additional font handling
        draw = ImageDraw.Draw(img)
        font = self._get_font(100)

        # Position in top-right
        x = THUMBNAIL_WIDTH - 150
        y = 30

        draw.text((x, y), emoji, font=font, fill=(255, 255, 255))

        return img

    def generate_variants(
        self,
        text: str,
        styles: list[str] | None = None,
        background_image: Path | str | None = None,
    ) -> list[Path]:
        """Generate multiple thumbnail variants for A/B testing"""
        if styles is None:
            styles = ["bold", "minimal", "viral"]

        variants = []
        for i, style in enumerate(styles):
            path = self.generate(
                text=text,
                style=style,
                background_image=background_image,
                output_name=f"variant_{i+1}_{style}",
            )
            if path:
                variants.append(path)

        return variants

    def generate_from_template(
        self,
        template: str,
        variables: dict,
        style: str = "bold",
    ) -> Path | None:
        """Generate thumbnail from a template with variables"""
        # Simple template replacement
        text = template
        for key, value in variables.items():
            text = text.replace(f"{{{key}}}", str(value))

        return self.generate(text=text, style=style)

    def enhance_image(
        self,
        image_path: Path | str,
        brightness: float = 1.1,
        contrast: float = 1.2,
        saturation: float = 1.3,
    ) -> Image.Image:
        """Enhance an image for thumbnail use"""
        img = Image.open(image_path)

        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

        # Saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

        return img

    def add_face_frame(
        self,
        img: Image.Image,
        face_image: Path | str,
        position: Literal["left", "right"] = "right",
        size: float = 0.4,
    ) -> Image.Image:
        """Add a face/person cutout to the thumbnail"""
        face = Image.open(face_image)

        # Convert to RGBA for transparency
        if face.mode != "RGBA":
            face = face.convert("RGBA")

        # Resize face
        face_width = int(THUMBNAIL_WIDTH * size)
        face_height = int(face.height * (face_width / face.width))
        face = face.resize((face_width, face_height), Image.Resampling.LANCZOS)

        # Position
        if position == "right":
            x = THUMBNAIL_WIDTH - face_width + 50
        else:
            x = -50

        y = THUMBNAIL_HEIGHT - face_height

        # Composite
        img = img.convert("RGBA")
        img.paste(face, (x, y), face)

        return img.convert("RGB")

    @staticmethod
    def list_styles() -> list[dict]:
        """List available thumbnail styles"""
        return [
            {
                "name": style.name,
                "text_color": style.text_color,
                "accent_color": style.accent_color,
                "position": style.text_position,
            }
            for style in THUMBNAIL_STYLES.values()
        ]

    def print_styles(self):
        """Print available styles"""
        from rich.table import Table

        table = Table(title="Thumbnail Styles")
        table.add_column("Name")
        table.add_column("Text Position")
        table.add_column("Border")
        table.add_column("Shadow")

        for name, style in THUMBNAIL_STYLES.items():
            table.add_row(
                name,
                style.text_position,
                "Yes" if style.add_border else "No",
                "Yes" if style.text_shadow else "No",
            )

        console.print(table)
