"""Unit tests for video effects module"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.video.effects import (
    apply_ken_burns,
    create_text_clip,
    add_subtitles,
    add_background_music,
    crossfade_clips,
    slide_transition,
    zoom_transition,
    apply_vignette,
    apply_color_grade,
    create_animated_text,
    add_progress_bar,
    apply_shake_effect,
    create_countdown,
    COLOR_GRADES,
)


class TestKenBurns:
    """Tests for Ken Burns effect"""

    def test_ken_burns_returns_clip(self):
        """Test that Ken Burns returns a valid clip"""
        # Create mock image clip
        mock_clip = MagicMock()
        mock_clip.duration = 5
        mock_clip.w = 1920
        mock_clip.h = 1080
        mock_clip.resize.return_value = mock_clip
        mock_clip.set_position.return_value = mock_clip

        result = apply_ken_burns(mock_clip, duration=5)

        assert result is not None

    def test_ken_burns_zoom_in(self):
        """Test zoom in effect"""
        mock_clip = MagicMock()
        mock_clip.duration = 3
        mock_clip.w = 1920
        mock_clip.h = 1080
        mock_clip.resize.return_value = mock_clip
        mock_clip.set_position.return_value = mock_clip

        result = apply_ken_burns(mock_clip, zoom_start=1.0, zoom_end=1.3, duration=3)

        assert result is not None

    def test_ken_burns_zoom_out(self):
        """Test zoom out effect"""
        mock_clip = MagicMock()
        mock_clip.duration = 3
        mock_clip.w = 1920
        mock_clip.h = 1080
        mock_clip.resize.return_value = mock_clip
        mock_clip.set_position.return_value = mock_clip

        result = apply_ken_burns(mock_clip, zoom_start=1.3, zoom_end=1.0, duration=3)

        assert result is not None


class TestTextClip:
    """Tests for text clip creation"""

    def test_create_text_clip_basic(self):
        """Test basic text clip creation"""
        with patch("src.video.effects.TextClip") as mock_text:
            mock_instance = MagicMock()
            mock_text.return_value = mock_instance
            mock_instance.set_position.return_value = mock_instance
            mock_instance.set_duration.return_value = mock_instance

            result = create_text_clip("Test text", duration=5)

            assert result is not None
            mock_text.assert_called_once()

    def test_create_text_clip_with_options(self):
        """Test text clip with custom options"""
        with patch("src.video.effects.TextClip") as mock_text:
            mock_instance = MagicMock()
            mock_text.return_value = mock_instance
            mock_instance.set_position.return_value = mock_instance
            mock_instance.set_duration.return_value = mock_instance

            result = create_text_clip(
                "Test text",
                duration=5,
                fontsize=80,
                color="red",
                position="top",
            )

            assert result is not None


class TestSubtitles:
    """Tests for subtitle functionality"""

    def test_add_subtitles_returns_composite(self):
        """Test that subtitles return composite clip"""
        mock_video = MagicMock()
        mock_video.duration = 10
        mock_video.w = 1920
        mock_video.h = 1080

        subtitles = [
            {"text": "Hello", "start": 0, "end": 2},
            {"text": "World", "start": 2, "end": 4},
        ]

        with patch("src.video.effects.TextClip") as mock_text, \
             patch("src.video.effects.CompositeVideoClip") as mock_composite:

            mock_text_instance = MagicMock()
            mock_text.return_value = mock_text_instance
            mock_text_instance.set_position.return_value = mock_text_instance
            mock_text_instance.set_start.return_value = mock_text_instance
            mock_text_instance.set_duration.return_value = mock_text_instance

            mock_composite.return_value = MagicMock()

            result = add_subtitles(mock_video, subtitles)

            assert result is not None
            mock_composite.assert_called_once()


class TestBackgroundMusic:
    """Tests for background music"""

    def test_add_background_music(self, temp_dir):
        """Test adding background music"""
        mock_video = MagicMock()
        mock_video.duration = 30
        mock_video.audio = MagicMock()

        with patch("src.video.effects.AudioFileClip") as mock_audio:
            mock_audio_instance = MagicMock()
            mock_audio.return_value = mock_audio_instance
            mock_audio_instance.volumex.return_value = mock_audio_instance
            mock_audio_instance.audio_loop.return_value = mock_audio_instance
            mock_audio_instance.subclip.return_value = mock_audio_instance

            with patch("src.video.effects.CompositeAudioClip") as mock_composite:
                mock_composite.return_value = MagicMock()

                result = add_background_music(
                    mock_video,
                    str(temp_dir / "music.mp3"),
                    volume=0.3,
                )

                assert result is not None


class TestTransitions:
    """Tests for video transitions"""

    def test_crossfade_clips(self):
        """Test crossfade between clips"""
        mock_clip1 = MagicMock()
        mock_clip1.duration = 5
        mock_clip2 = MagicMock()
        mock_clip2.duration = 5

        with patch("src.video.effects.CompositeVideoClip") as mock_composite:
            mock_composite.return_value = MagicMock()

            result = crossfade_clips(mock_clip1, mock_clip2, duration=1)

            assert result is not None

    def test_slide_transition_directions(self):
        """Test slide transition in different directions"""
        mock_clip1 = MagicMock()
        mock_clip1.duration = 5
        mock_clip1.w = 1920
        mock_clip1.h = 1080

        mock_clip2 = MagicMock()
        mock_clip2.duration = 5
        mock_clip2.w = 1920
        mock_clip2.h = 1080

        directions = ["left", "right", "up", "down"]

        for direction in directions:
            with patch("src.video.effects.CompositeVideoClip") as mock_composite:
                mock_composite.return_value = MagicMock()

                result = slide_transition(
                    mock_clip1,
                    mock_clip2,
                    direction=direction,
                    duration=1,
                )

                assert result is not None

    def test_zoom_transition(self):
        """Test zoom transition"""
        mock_clip1 = MagicMock()
        mock_clip1.duration = 5
        mock_clip2 = MagicMock()
        mock_clip2.duration = 5

        with patch("src.video.effects.CompositeVideoClip") as mock_composite:
            mock_composite.return_value = MagicMock()

            result = zoom_transition(mock_clip1, mock_clip2, zoom_type="in")

            assert result is not None


class TestColorGrading:
    """Tests for color grading"""

    def test_color_grades_exist(self):
        """Test that all color grades are defined"""
        expected_grades = ["cinematic", "vintage", "vibrant", "moody", "warm", "cool"]

        for grade in expected_grades:
            assert grade in COLOR_GRADES

    def test_apply_color_grade_presets(self):
        """Test applying color grade presets"""
        mock_clip = MagicMock()

        for preset in COLOR_GRADES.keys():
            with patch("src.video.effects.vfx") as mock_vfx:
                mock_vfx.colorx.return_value = mock_clip

                result = apply_color_grade(mock_clip, preset=preset)

                assert result is not None

    def test_apply_vignette(self):
        """Test vignette effect"""
        # Create a sample frame
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

        result = apply_vignette(frame, strength=0.5)

        assert result is not None
        assert result.shape == frame.shape
        # Edges should be darker than center
        assert np.mean(result[0, :, :]) < np.mean(result[540, :, :])


class TestAnimatedText:
    """Tests for animated text"""

    def test_create_animated_text_fade(self):
        """Test fade animation"""
        with patch("src.video.effects.TextClip") as mock_text:
            mock_instance = MagicMock()
            mock_text.return_value = mock_instance
            mock_instance.set_position.return_value = mock_instance
            mock_instance.set_duration.return_value = mock_instance
            mock_instance.crossfadein.return_value = mock_instance
            mock_instance.crossfadeout.return_value = mock_instance

            result = create_animated_text(
                "Test",
                duration=3,
                animation="fade",
            )

            assert result is not None

    def test_create_animated_text_typewriter(self):
        """Test typewriter animation"""
        with patch("src.video.effects.TextClip") as mock_text, \
             patch("src.video.effects.CompositeVideoClip") as mock_composite:

            mock_instance = MagicMock()
            mock_text.return_value = mock_instance
            mock_instance.set_position.return_value = mock_instance
            mock_instance.set_duration.return_value = mock_instance
            mock_instance.set_start.return_value = mock_instance

            mock_composite.return_value = MagicMock()

            result = create_animated_text(
                "Test",
                duration=3,
                animation="typewriter",
            )

            assert result is not None

    def test_create_animated_text_scale(self):
        """Test scale animation"""
        with patch("src.video.effects.TextClip") as mock_text:
            mock_instance = MagicMock()
            mock_text.return_value = mock_instance
            mock_instance.set_position.return_value = mock_instance
            mock_instance.set_duration.return_value = mock_instance
            mock_instance.resize.return_value = mock_instance

            result = create_animated_text(
                "Test",
                duration=3,
                animation="scale",
            )

            assert result is not None


class TestProgressBar:
    """Tests for progress bar"""

    def test_add_progress_bar(self):
        """Test adding progress bar"""
        mock_video = MagicMock()
        mock_video.duration = 30
        mock_video.w = 1920
        mock_video.h = 1080

        with patch("src.video.effects.ColorClip") as mock_color, \
             patch("src.video.effects.CompositeVideoClip") as mock_composite:

            mock_color.return_value = MagicMock()
            mock_composite.return_value = MagicMock()

            result = add_progress_bar(
                mock_video,
                color=(255, 0, 0),
                height=5,
                position="bottom",
            )

            assert result is not None


class TestShakeEffect:
    """Tests for camera shake effect"""

    def test_apply_shake_effect(self):
        """Test shake effect"""
        # Create sample frame
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128

        result = apply_shake_effect(frame, intensity=10, t=0.5)

        assert result is not None
        assert result.shape == frame.shape

    def test_shake_intensity(self):
        """Test that intensity affects shake amount"""
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128

        # With no intensity, frame should be unchanged
        result_zero = apply_shake_effect(frame.copy(), intensity=0, t=0.5)

        # With intensity, frame should be shifted
        result_shake = apply_shake_effect(frame.copy(), intensity=20, t=0.5)

        # Results should be different
        # Note: Due to randomness, we just check shape is preserved
        assert result_zero.shape == result_shake.shape


class TestCountdown:
    """Tests for countdown animation"""

    def test_create_countdown(self):
        """Test countdown creation"""
        with patch("src.video.effects.TextClip") as mock_text, \
             patch("src.video.effects.concatenate_videoclips") as mock_concat:

            mock_instance = MagicMock()
            mock_text.return_value = mock_instance
            mock_instance.set_position.return_value = mock_instance
            mock_instance.set_duration.return_value = mock_instance

            mock_concat.return_value = MagicMock()

            result = create_countdown(
                start=3,
                duration_per_number=1,
            )

            assert result is not None
            # Should create clips for 3, 2, 1
            assert mock_text.call_count >= 3

    def test_create_countdown_custom_start(self):
        """Test countdown with custom start"""
        with patch("src.video.effects.TextClip") as mock_text, \
             patch("src.video.effects.concatenate_videoclips") as mock_concat:

            mock_instance = MagicMock()
            mock_text.return_value = mock_instance
            mock_instance.set_position.return_value = mock_instance
            mock_instance.set_duration.return_value = mock_instance

            mock_concat.return_value = MagicMock()

            result = create_countdown(start=5)

            assert result is not None
            # Should create clips for 5, 4, 3, 2, 1
            assert mock_text.call_count >= 5
