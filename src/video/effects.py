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


def slide_transition(
    frames1: list[np.ndarray],
    frames2: list[np.ndarray],
    transition_frames: int = 15,
    direction: str = "left",  # left, right, up, down
) -> list[np.ndarray]:
    """Create slide transition between two frame sequences"""
    if not frames1 or not frames2:
        return frames1 + frames2

    height, width = frames1[0].shape[:2]
    result = frames1[:-transition_frames] if len(frames1) > transition_frames else []

    for i in range(transition_frames):
        progress = i / transition_frames
        # Ease-out curve for smoother animation
        progress = 1 - (1 - progress) ** 2

        idx1 = len(frames1) - transition_frames + i
        idx2 = min(i, len(frames2) - 1)

        frame1 = frames1[idx1] if idx1 < len(frames1) else frames1[-1]
        frame2 = frames2[idx2] if idx2 < len(frames2) else frames2[0]

        # Create composite frame
        composite = np.zeros_like(frame1)

        if direction == "left":
            offset = int(width * progress)
            composite[:, :width-offset] = frame1[:, offset:]
            composite[:, width-offset:] = frame2[:, :offset]
        elif direction == "right":
            offset = int(width * progress)
            composite[:, offset:] = frame1[:, :width-offset]
            composite[:, :offset] = frame2[:, width-offset:]
        elif direction == "up":
            offset = int(height * progress)
            composite[:height-offset, :] = frame1[offset:, :]
            composite[height-offset:, :] = frame2[:offset, :]
        elif direction == "down":
            offset = int(height * progress)
            composite[offset:, :] = frame1[:height-offset, :]
            composite[:offset, :] = frame2[height-offset:, :]

        result.append(composite)

    # Add remaining frames
    if len(frames2) > transition_frames:
        result.extend(frames2[transition_frames:])
    elif frames2:
        result.extend(frames2)

    return result


def zoom_transition(
    frames1: list[np.ndarray],
    frames2: list[np.ndarray],
    transition_frames: int = 15,
    zoom_out: bool = True,
) -> list[np.ndarray]:
    """Create zoom in/out transition between sequences"""
    if not frames1 or not frames2:
        return frames1 + frames2

    height, width = frames1[0].shape[:2]
    result = frames1[:-transition_frames] if len(frames1) > transition_frames else []

    for i in range(transition_frames):
        progress = i / transition_frames
        # Ease-in-out curve
        progress = progress * progress * (3 - 2 * progress)

        idx1 = len(frames1) - transition_frames + i
        idx2 = min(i, len(frames2) - 1)

        frame1 = frames1[idx1] if idx1 < len(frames1) else frames1[-1]
        frame2 = frames2[idx2] if idx2 < len(frames2) else frames2[0]

        if zoom_out:
            # Frame 1 zooms out, frame 2 appears
            scale1 = 1.0 + progress * 0.5  # 1.0 to 1.5
            alpha = progress
        else:
            # Frame 2 zooms in from small
            scale1 = 1.0
            alpha = progress

        # Apply zoom to frame1
        if scale1 != 1.0:
            from PIL import Image
            img1 = Image.fromarray(frame1)
            new_size = (int(width * scale1), int(height * scale1))
            img1 = img1.resize(new_size, Image.Resampling.LANCZOS)
            # Center crop
            left = (new_size[0] - width) // 2
            top = (new_size[1] - height) // 2
            img1 = img1.crop((left, top, left + width, top + height))
            frame1 = np.array(img1)

        # Blend frames
        composite = (
            frame1.astype(float) * (1 - alpha) +
            frame2.astype(float) * alpha
        ).astype(np.uint8)

        result.append(composite)

    if len(frames2) > transition_frames:
        result.extend(frames2[transition_frames:])
    elif frames2:
        result.extend(frames2)

    return result


def apply_vignette(
    frames: list[np.ndarray],
    strength: float = 0.5,
) -> list[np.ndarray]:
    """Apply vignette (dark edges) effect to frames"""
    if not frames:
        return frames

    height, width = frames[0].shape[:2]

    # Create vignette mask
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2

    # Distance from center, normalized
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist = dist / max_dist

    # Vignette falloff
    vignette = 1 - (dist ** 2) * strength
    vignette = np.clip(vignette, 0, 1)
    vignette = vignette[:, :, np.newaxis]

    result = []
    for frame in frames:
        vignetted = (frame.astype(float) * vignette).astype(np.uint8)
        result.append(vignetted)

    return result


def apply_color_grade(
    frames: list[np.ndarray],
    preset: str = "cinematic",
) -> list[np.ndarray]:
    """Apply color grading preset to frames"""
    if not frames:
        return frames

    presets = {
        "cinematic": {
            "contrast": 1.1,
            "saturation": 0.9,
            "shadows_tint": (0, 0, 20),  # Blue shadows
            "highlights_tint": (20, 10, 0),  # Warm highlights
        },
        "vintage": {
            "contrast": 0.95,
            "saturation": 0.7,
            "shadows_tint": (20, 10, 0),
            "highlights_tint": (30, 20, 0),
        },
        "vibrant": {
            "contrast": 1.15,
            "saturation": 1.3,
            "shadows_tint": (0, 0, 0),
            "highlights_tint": (0, 0, 0),
        },
        "moody": {
            "contrast": 1.2,
            "saturation": 0.8,
            "shadows_tint": (0, 10, 30),
            "highlights_tint": (0, 0, 0),
        },
        "warm": {
            "contrast": 1.05,
            "saturation": 1.1,
            "shadows_tint": (10, 5, 0),
            "highlights_tint": (30, 15, 0),
        },
        "cool": {
            "contrast": 1.05,
            "saturation": 0.95,
            "shadows_tint": (0, 5, 20),
            "highlights_tint": (0, 10, 30),
        },
    }

    if preset not in presets:
        return frames

    settings = presets[preset]
    result = []

    for frame in frames:
        graded = frame.astype(float)

        # Apply contrast
        graded = (graded - 128) * settings["contrast"] + 128

        # Apply saturation (convert to HSV-like adjustment)
        gray = np.mean(graded, axis=2, keepdims=True)
        graded = gray + (graded - gray) * settings["saturation"]

        # Apply shadow/highlight tints
        luminance = np.mean(graded, axis=2)
        shadow_mask = (luminance < 80)[:, :, np.newaxis]
        highlight_mask = (luminance > 180)[:, :, np.newaxis]

        shadows = np.array(settings["shadows_tint"])
        highlights = np.array(settings["highlights_tint"])

        graded = graded + shadow_mask * shadows * 0.5
        graded = graded + highlight_mask * highlights * 0.5

        graded = np.clip(graded, 0, 255).astype(np.uint8)
        result.append(graded)

    return result


def create_animated_text(
    text: str,
    size: tuple[int, int],
    duration: float,
    fps: int = 30,
    animation: str = "fade",  # fade, typewriter, scale, slide
    **text_kwargs,
) -> list[np.ndarray]:
    """Create animated text overlay frames"""
    total_frames = int(duration * fps)
    frames = []

    for i in range(total_frames):
        progress = i / max(total_frames - 1, 1)

        if animation == "fade":
            # Fade in during first 20%, stay, fade out during last 20%
            if progress < 0.2:
                alpha = progress / 0.2
            elif progress > 0.8:
                alpha = (1 - progress) / 0.2
            else:
                alpha = 1.0

            text_img = create_text_clip(text, size, **text_kwargs)
            text_img[:, :, 3] = (text_img[:, :, 3] * alpha).astype(np.uint8)
            frames.append(text_img)

        elif animation == "typewriter":
            # Reveal text character by character
            chars_to_show = int(len(text) * min(progress * 1.5, 1.0))
            partial_text = text[:chars_to_show]
            if partial_text:
                text_img = create_text_clip(partial_text, size, **text_kwargs)
            else:
                text_img = np.zeros((*size[::-1], 4), dtype=np.uint8)
            frames.append(text_img)

        elif animation == "scale":
            # Scale up and settle
            if progress < 0.3:
                scale = 0.5 + progress / 0.3 * 0.5
            else:
                scale = 1.0

            text_img = create_text_clip(text, size, **text_kwargs)

            if scale != 1.0:
                from PIL import Image
                img = Image.fromarray(text_img)
                new_size = (int(size[0] * scale), int(size[1] * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                # Center
                result = np.zeros((*size[::-1], 4), dtype=np.uint8)
                x = (size[0] - new_size[0]) // 2
                y = (size[1] - new_size[1]) // 2
                result[y:y+new_size[1], x:x+new_size[0]] = np.array(img)
                text_img = result

            frames.append(text_img)

        elif animation == "slide":
            # Slide in from bottom
            text_img = create_text_clip(text, size, **text_kwargs)

            if progress < 0.3:
                offset = int(size[1] * 0.3 * (1 - progress / 0.3))
                result = np.zeros_like(text_img)
                if offset < size[1]:
                    result[:-offset or None, :] = text_img[offset:, :]
                text_img = result

            frames.append(text_img)

        else:
            # No animation
            frames.append(create_text_clip(text, size, **text_kwargs))

    return frames


def add_progress_bar(
    frames: list[np.ndarray],
    color: tuple = (255, 0, 0),
    height: int = 5,
    position: str = "bottom",
) -> list[np.ndarray]:
    """Add a progress bar to the video"""
    if not frames:
        return frames

    total_frames = len(frames)
    frame_height, frame_width = frames[0].shape[:2]

    result = []
    for i, frame in enumerate(frames):
        progress = (i + 1) / total_frames
        bar_width = int(frame_width * progress)

        frame_copy = frame.copy()

        if position == "bottom":
            y = frame_height - height
        else:
            y = 0

        frame_copy[y:y+height, :bar_width] = color
        result.append(frame_copy)

    return result


def apply_shake_effect(
    frames: list[np.ndarray],
    intensity: float = 5.0,
    frequency: float = 2.0,
) -> list[np.ndarray]:
    """Apply camera shake effect"""
    if not frames:
        return frames

    import math

    height, width = frames[0].shape[:2]
    result = []

    for i, frame in enumerate(frames):
        t = i / 30.0  # Assume 30 fps

        # Random-ish shake using sine waves
        offset_x = int(intensity * math.sin(t * frequency * 10) * math.cos(t * frequency * 7))
        offset_y = int(intensity * math.sin(t * frequency * 8) * math.cos(t * frequency * 11))

        # Create padded frame
        padded = np.zeros((height + abs(offset_y) * 2, width + abs(offset_x) * 2, 3), dtype=np.uint8)
        py = abs(offset_y)
        px = abs(offset_x)
        padded[py:py+height, px:px+width] = frame[:, :, :3] if frame.shape[2] >= 3 else frame

        # Crop with offset
        crop_y = py + offset_y
        crop_x = px + offset_x
        cropped = padded[crop_y:crop_y+height, crop_x:crop_x+width]

        result.append(cropped)

    return result


def create_countdown(
    duration: float,
    size: tuple[int, int],
    fps: int = 30,
    font_size: int = 200,
    color: str = "white",
    background_color: tuple = (0, 0, 0),
) -> list[np.ndarray]:
    """Create countdown animation frames"""
    total_frames = int(duration * fps)
    frames = []

    for i in range(total_frames):
        remaining = duration - (i / fps)
        seconds = max(0, int(remaining) + 1)

        # Create frame
        img = Image.new("RGB", size, background_color)
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        text = str(seconds)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        # Scale effect within each second
        sub_progress = remaining - int(remaining)
        scale = 1.0 + sub_progress * 0.2

        draw.text((x, y), text, font=font, fill=color)

        frames.append(np.array(img))

    return frames
