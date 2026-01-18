"""Unit tests for video effects module"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from src.video.effects import (
    ken_burns_effect,
    create_text_clip,
    add_subtitles,
    crossfade_frames,
    slide_transition,
    zoom_transition,
    apply_vignette,
    apply_color_grade,
    create_animated_text,
    add_progress_bar,
    apply_shake_effect,
    create_countdown,
)


class TestKenBurns:
    """Tests for Ken Burns effect"""

    def test_ken_burns_returns_frames(self, temp_dir):
        """Test that Ken Burns returns valid frames"""
        # Create a test image
        img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
        img_path = temp_dir / "test_image.png"
        img.save(img_path)

        frames = ken_burns_effect(
            img_path,
            duration=1.0,
            target_size=(1080, 1920),
            fps=10,
        )

        assert frames is not None
        assert len(frames) == 10  # 1 second * 10 fps
        assert isinstance(frames[0], np.ndarray)

    def test_ken_burns_zoom_in(self, temp_dir):
        """Test zoom in direction"""
        img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
        img_path = temp_dir / "test_image.png"
        img.save(img_path)

        frames = ken_burns_effect(
            img_path,
            duration=0.5,
            target_size=(1080, 1920),
            direction="in",
            fps=10,
        )

        assert len(frames) == 5

    def test_ken_burns_zoom_out(self, temp_dir):
        """Test zoom out direction"""
        img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
        img_path = temp_dir / "test_image.png"
        img.save(img_path)

        frames = ken_burns_effect(
            img_path,
            duration=0.5,
            target_size=(1080, 1920),
            direction="out",
            fps=10,
        )

        assert len(frames) == 5


class TestTextClip:
    """Tests for text clip creation"""

    def test_create_text_clip_basic(self):
        """Test basic text clip creation"""
        result = create_text_clip(
            "Test text",
            size=(1920, 1080),
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (1080, 1920, 4)  # RGBA

    def test_create_text_clip_with_options(self):
        """Test text clip with custom options"""
        result = create_text_clip(
            "Custom text",
            size=(1920, 1080),
            font_size=80,
            color="red",
            position="top",
        )

        assert result is not None
        assert result.shape == (1080, 1920, 4)

    def test_create_text_clip_positions(self):
        """Test different text positions"""
        for position in ["top", "center", "bottom"]:
            result = create_text_clip(
                "Position test",
                size=(1920, 1080),
                position=position,
            )
            assert result is not None


class TestSubtitles:
    """Tests for subtitle functionality"""

    def test_add_subtitles_to_frames(self):
        """Test adding subtitles to frames"""
        # Create sample frames
        frames = [np.ones((1080, 1920, 3), dtype=np.uint8) * 128 for _ in range(10)]

        result = add_subtitles(frames, "Test subtitle")

        assert result is not None
        assert len(result) == 10
        assert result[0].shape[:2] == (1080, 1920)

    def test_add_subtitles_empty_frames(self):
        """Test with empty frames list"""
        result = add_subtitles([], "Test")

        assert result == []


class TestTransitions:
    """Tests for video transitions"""

    def test_crossfade_frames(self):
        """Test crossfade between frame sequences"""
        frames1 = [np.ones((100, 100, 3), dtype=np.uint8) * 50 for _ in range(20)]
        frames2 = [np.ones((100, 100, 3), dtype=np.uint8) * 200 for _ in range(20)]

        result = crossfade_frames(frames1, frames2, transition_frames=10)

        assert result is not None
        assert len(result) > 0

    def test_crossfade_empty_frames(self):
        """Test crossfade with empty inputs"""
        result = crossfade_frames([], [])
        assert result == []

    def test_slide_transition_directions(self):
        """Test slide transition in different directions"""
        frames1 = [np.ones((100, 100, 3), dtype=np.uint8) * 50 for _ in range(20)]
        frames2 = [np.ones((100, 100, 3), dtype=np.uint8) * 200 for _ in range(20)]

        for direction in ["left", "right", "up", "down"]:
            result = slide_transition(
                frames1, frames2,
                transition_frames=10,
                direction=direction,
            )

            assert result is not None
            assert len(result) > 0

    def test_zoom_transition(self):
        """Test zoom transition"""
        frames1 = [np.ones((100, 100, 3), dtype=np.uint8) * 50 for _ in range(20)]
        frames2 = [np.ones((100, 100, 3), dtype=np.uint8) * 200 for _ in range(20)]

        result = zoom_transition(frames1, frames2, transition_frames=10)

        assert result is not None
        assert len(result) > 0


class TestColorGrading:
    """Tests for color grading"""

    def test_apply_color_grade_presets(self):
        """Test applying color grade presets"""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]

        presets = ["cinematic", "vintage", "vibrant", "moody", "warm", "cool"]

        for preset in presets:
            result = apply_color_grade(frames, preset=preset)

            assert result is not None
            assert len(result) == 5

    def test_apply_color_grade_invalid_preset(self):
        """Test with invalid preset"""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 128]

        result = apply_color_grade(frames, preset="nonexistent")

        # Should return original frames
        assert len(result) == 1

    def test_apply_vignette(self):
        """Test vignette effect"""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 255 for _ in range(3)]

        result = apply_vignette(frames, strength=0.5)

        assert result is not None
        assert len(result) == 3

        # Edges should be darker than center
        frame = result[0]
        center_val = np.mean(frame[50, 50, :])
        corner_val = np.mean(frame[0, 0, :])
        assert corner_val < center_val

    def test_apply_vignette_empty(self):
        """Test vignette with empty frames"""
        result = apply_vignette([])
        assert result == []


class TestAnimatedText:
    """Tests for animated text"""

    def test_create_animated_text_fade(self):
        """Test fade animation"""
        result = create_animated_text(
            "Test",
            size=(640, 480),
            duration=1.0,
            fps=10,
            animation="fade",
        )

        assert result is not None
        assert len(result) == 10

    def test_create_animated_text_typewriter(self):
        """Test typewriter animation"""
        result = create_animated_text(
            "Hello World",
            size=(640, 480),
            duration=1.0,
            fps=10,
            animation="typewriter",
        )

        assert result is not None
        assert len(result) == 10

    def test_create_animated_text_scale(self):
        """Test scale animation"""
        result = create_animated_text(
            "Scale Test",
            size=(640, 480),
            duration=1.0,
            fps=10,
            animation="scale",
        )

        assert result is not None
        assert len(result) == 10

    def test_create_animated_text_slide(self):
        """Test slide animation"""
        result = create_animated_text(
            "Slide Test",
            size=(640, 480),
            duration=1.0,
            fps=10,
            animation="slide",
        )

        assert result is not None
        assert len(result) == 10


class TestProgressBar:
    """Tests for progress bar"""

    def test_add_progress_bar(self):
        """Test adding progress bar"""
        frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(10)]

        result = add_progress_bar(
            frames,
            color=(255, 0, 0),
            height=5,
            position="bottom",
        )

        assert result is not None
        assert len(result) == 10

        # Check last frame has full bar
        last_frame = result[-1]
        # Bottom row should have red pixels
        assert last_frame[95, 100, 0] == 255  # Red channel

    def test_add_progress_bar_top(self):
        """Test progress bar at top"""
        frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(5)]

        result = add_progress_bar(frames, position="top")

        assert len(result) == 5

    def test_add_progress_bar_empty(self):
        """Test with empty frames"""
        result = add_progress_bar([])
        assert result == []


class TestShakeEffect:
    """Tests for camera shake effect"""

    def test_apply_shake_effect(self):
        """Test shake effect"""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(10)]

        result = apply_shake_effect(frames, intensity=5.0)

        assert result is not None
        assert len(result) == 10

    def test_apply_shake_effect_zero_intensity(self):
        """Test with zero intensity"""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]

        result = apply_shake_effect(frames, intensity=0.0)

        assert len(result) == 5

    def test_apply_shake_effect_empty(self):
        """Test with empty frames"""
        result = apply_shake_effect([])
        assert result == []


class TestCountdown:
    """Tests for countdown animation"""

    def test_create_countdown(self):
        """Test countdown creation"""
        result = create_countdown(
            duration=3.0,
            size=(640, 480),
            fps=10,
        )

        assert result is not None
        assert len(result) == 30  # 3 seconds * 10 fps

    def test_create_countdown_custom_colors(self):
        """Test countdown with custom colors"""
        result = create_countdown(
            duration=2.0,
            size=(640, 480),
            fps=10,
            color="red",
            background_color=(0, 0, 255),
        )

        assert result is not None
        assert len(result) == 20

    def test_countdown_frame_dimensions(self):
        """Test countdown frame dimensions"""
        result = create_countdown(
            duration=1.0,
            size=(1920, 1080),
            fps=5,
        )

        assert result[0].shape == (1080, 1920, 3)
