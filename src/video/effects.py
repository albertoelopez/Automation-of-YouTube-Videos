"""Video effects and transformations"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def ken_burns_effect(
    image_path: Path | str,
    duration: float,
    target_size: tuple[int, int],
    zoom_ratio: float = 1.2,
    fps: int = 30,
    direction: str = "in",  # "in" or "out"
) -> list[np.ndarray]:
    """
    Apply Ken Burns (pan and zoom) effect to a static image.
    Returns list of frames as numpy arrays.
    """
    img = Image.open(image_path)

    # Ensure image is large enough for effect
    min_size = (int(target_size[0] * zoom_ratio), int(target_size[1] * zoom_ratio))

    # Resize if needed while maintaining aspect ratio
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        # Image is wider - fit by height
        new_height = min_size[1]
        new_width = int(new_height * img_ratio)
    else:
        # Image is taller - fit by width
        new_width = min_size[0]
        new_height = int(new_width / img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    total_frames = int(duration * fps)
    frames = []

    for frame_idx in range(total_frames):
        progress = frame_idx / max(total_frames - 1, 1)

        if direction == "in":
            # Zoom in: start wide, end tight
            current_zoom = 1.0 + (zoom_ratio - 1.0) * progress
        else:
            # Zoom out: start tight, end wide
            current_zoom = zoom_ratio - (zoom_ratio - 1.0) * progress

        # Calculate crop size
        crop_width = int(target_size[0] * (zoom_ratio / current_zoom))
        crop_height = int(target_size[1] * (zoom_ratio / current_zoom))

        # Center crop with slight movement
        center_x = img.width // 2 + int((progress - 0.5) * 20)  # Slight pan
        center_y = img.height // 2 + int((progress - 0.5) * 10)

        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        right = min(img.width, left + crop_width)
        bottom = min(img.height, top + crop_height)

        # Adjust if crop goes out of bounds
        if right - left < crop_width:
            left = max(0, right - crop_width)
        if bottom - top < crop_height:
            top = max(0, bottom - crop_height)

        # Crop and resize to target
        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize(target_size, Image.Resampling.LANCZOS)

        frames.append(np.array(resized))

    return frames


def create_text_clip(
    text: str,
    size: tuple[int, int],
    font_path: str | None = None,
    font_size: int = 60,
    color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 3,
    position: str = "bottom",
    padding: int = 50,
) -> np.ndarray:
    """Create a transparent text overlay"""
    # Create transparent image
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try system fonts
            for font_name in ["DejaVuSans.ttf", "Arial.ttf", "FreeSans.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except OSError:
                    continue
            else:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Word wrap text
    max_width = size[0] - (padding * 2)
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    # Calculate text position
    line_height = font_size + 10
    total_height = len(lines) * line_height

    if position == "top":
        y_start = padding
    elif position == "center":
        y_start = (size[1] - total_height) // 2
    else:  # bottom
        y_start = size[1] - total_height - padding

    # Draw text with stroke
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = y_start + i * line_height

        # Draw stroke
        if stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)

        # Draw text
        draw.text((x, y), line, font=font, fill=color)

    return np.array(img)


def add_subtitles(
    frames: list[np.ndarray],
    text: str,
    **kwargs,
) -> list[np.ndarray]:
    """Add subtitles to frames"""
    if not frames:
        return frames

    height, width = frames[0].shape[:2]
    text_overlay = create_text_clip(text, (width, height), **kwargs)

    # Composite text onto each frame
    result = []
    for frame in frames:
        # Convert frame to RGBA if needed
        if frame.shape[2] == 3:
            frame_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            frame_rgba[:, :, :3] = frame
            frame_rgba[:, :, 3] = 255
        else:
            frame_rgba = frame.copy()

        # Alpha composite
        alpha = text_overlay[:, :, 3:4] / 255.0
        composite = frame_rgba[:, :, :3] * (1 - alpha) + text_overlay[:, :, :3] * alpha
        result.append(composite.astype(np.uint8))

    return result


def crossfade_frames(
    frames1: list[np.ndarray],
    frames2: list[np.ndarray],
    transition_frames: int = 15,
) -> list[np.ndarray]:
    """Create crossfade transition between two frame sequences"""
    if not frames1 or not frames2:
        return frames1 + frames2

    result = frames1[:-transition_frames] if len(frames1) > transition_frames else []

    # Create transition
    for i in range(transition_frames):
        alpha = i / transition_frames
        idx1 = len(frames1) - transition_frames + i
        idx2 = i

        if idx1 < len(frames1) and idx2 < len(frames2):
            blended = (
                frames1[idx1].astype(float) * (1 - alpha) +
                frames2[idx2].astype(float) * alpha
            ).astype(np.uint8)
            result.append(blended)

    # Add remaining frames from second sequence
    if len(frames2) > transition_frames:
        result.extend(frames2[transition_frames:])
    elif frames2:
        result.extend(frames2)

    return result
