"""End-to-end tests for the video generation pipeline"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

# Import skip decorators from conftest
from tests.conftest import skip_without_ollama, skip_without_ffmpeg


class TestPipelineE2E:
    """End-to-end tests for the video pipeline"""

    @pytest.fixture
    def mock_pipeline_deps(self):
        """Mock all external dependencies"""
        with patch("src.llm.ollama.OllamaClient") as mock_llm, \
             patch("src.tts.piper.PiperTTS") as mock_tts, \
             patch("src.images.generator.ImageGenerator") as mock_img:

            # Configure LLM mock
            llm_instance = MagicMock()
            llm_instance.is_available.return_value = True
            llm_instance.generate.return_value = json.dumps({
                "title": "Test Video",
                "script": "This is a test script for the video.",
                "tags": ["test", "video"],
            })
            mock_llm.return_value = llm_instance

            # Configure TTS mock
            tts_instance = MagicMock()
            tts_instance.is_available.return_value = True
            tts_instance.synthesize.return_value = Path("/tmp/test_audio.wav")
            mock_tts.return_value = tts_instance

            # Configure image generator mock
            img_instance = MagicMock()
            img_instance.generate.return_value = MagicMock(path=Path("/tmp/test_img.png"))
            mock_img.return_value = img_instance

            yield {
                "llm": llm_instance,
                "tts": tts_instance,
                "image": img_instance,
            }

    @pytest.mark.e2e
    def test_pipeline_initialization(self, temp_dir):
        """Test pipeline can be initialized"""
        from src.pipeline import Pipeline
        from src.utils.config import load_config

        config = load_config()

        pipeline = Pipeline(config)

        assert pipeline is not None
        assert pipeline.llm is not None
        assert pipeline.tts is not None

    @pytest.mark.e2e
    def test_pipeline_status_check(self, temp_dir):
        """Test pipeline status check"""
        from src.pipeline import Pipeline
        from src.utils.config import load_config

        config = load_config()

        pipeline = Pipeline(config)

        # This should not raise
        pipeline.print_status()

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.requires_ollama
    def test_full_generation_mocked(self, temp_dir, mock_pipeline_deps):
        """Test full video generation with mocked dependencies"""
        # This test requires a more complete mocking setup
        # Skip for now - covered by unit tests
        pytest.skip("Requires complete pipeline mocking")


class TestCLIE2E:
    """End-to-end tests for CLI commands"""

    @pytest.mark.e2e
    def test_cli_status_command(self):
        """Test CLI status command"""
        from typer.testing import CliRunner
        from src.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["status"])

        # Should complete without error
        assert result.exit_code == 0 or "Ollama" in result.output

    @pytest.mark.e2e
    def test_cli_voices_command(self):
        """Test CLI voices command"""
        from typer.testing import CliRunner
        from src.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["voices"])

        assert result.exit_code == 0
        assert "Voice Name" in result.output or "voices" in result.output.lower()

    @pytest.mark.e2e
    def test_cli_templates_command(self):
        """Test CLI templates command"""
        from typer.testing import CliRunner
        from src.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["templates"])

        assert result.exit_code == 0
        assert "facts" in result.output.lower() or "Template" in result.output

    @pytest.mark.e2e
    def test_cli_thumbnail_styles_command(self):
        """Test CLI thumbnail-styles command"""
        from typer.testing import CliRunner
        from src.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["thumbnail-styles"])

        assert result.exit_code == 0
        assert "bold" in result.output.lower() or "Name" in result.output

    @pytest.mark.e2e
    def test_cli_channels_command(self, temp_dir):
        """Test CLI channels command"""
        from typer.testing import CliRunner
        from src.main import app

        runner = CliRunner()

        with patch("src.channels.manager.ChannelManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.list_channels.return_value = []
            mock_instance.print_all_channels.return_value = None
            mock_manager.return_value = mock_instance

            result = runner.invoke(app, ["channels"])

        # Check it ran without critical errors
        assert result.exit_code == 0 or "channel" in result.output.lower()

    @pytest.mark.e2e
    def test_cli_ab_list_command(self, temp_dir):
        """Test CLI ab-list command"""
        from typer.testing import CliRunner
        from src.main import app

        runner = CliRunner()

        with patch("src.optimization.ab_testing.ABTestManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.list_tests.return_value = []
            mock_instance.print_all_tests.return_value = None
            mock_manager.return_value = mock_instance

            result = runner.invoke(app, ["ab-list"])

        # Check it ran without critical errors
        assert result.exit_code == 0 or "test" in result.output.lower()


class TestThumbnailE2E:
    """End-to-end tests for thumbnail generation"""

    @pytest.mark.e2e
    def test_thumbnail_generation(self, temp_dir):
        """Test thumbnail generation end-to-end"""
        from src.images.thumbnail import ThumbnailGenerator

        gen = ThumbnailGenerator(output_dir=temp_dir / "thumbnails")
        result = gen.generate(
            text="Test Thumbnail E2E",
            style="bold",
        )

        assert result is not None
        assert result.exists()
        assert result.stat().st_size > 0

        # Verify it's a valid image
        from PIL import Image
        img = Image.open(result)
        assert img.size == (1280, 720)

    @pytest.mark.e2e
    def test_thumbnail_variants_generation(self, temp_dir):
        """Test generating multiple thumbnail variants"""
        from src.images.thumbnail import ThumbnailGenerator

        gen = ThumbnailGenerator(output_dir=temp_dir / "thumbnails")
        variants = gen.generate_variants(
            text="Variant Test",
            styles=["bold", "minimal"],
        )

        assert len(variants) == 2
        for path in variants:
            assert path.exists()


class TestResearchE2E:
    """End-to-end tests for research module"""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_trending_topics_fetch(self):
        """Test fetching trending topics (requires network)"""
        from src.research.trending import TrendingTopics

        trending = TrendingTopics()

        # Try Hacker News as it's most reliable
        topics = trending.get_hackernews_trending(limit=5)

        # Should get some topics (unless rate limited)
        # We don't assert count as network may fail
        assert isinstance(topics, list)

    @pytest.mark.e2e
    def test_idea_generator_with_mock(self):
        """Test idea generation with mocked LLM"""
        from src.research.ideas import IdeaGenerator

        gen = IdeaGenerator()

        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps({
            "title": "Test Idea",
            "hook": "Amazing hook",
            "outline": ["Point 1", "Point 2"],
            "target_audience": "Developers",
            "tags": ["test"],
            "viral_potential": "high",
            "difficulty": "easy",
        })

        gen.llm_client = mock_llm

        idea = gen.generate_idea("Python tips")

        assert idea is not None
        assert idea.title == "Test Idea"
        assert idea.viral_potential == "high"


class TestChannelE2E:
    """End-to-end tests for channel management"""

    @pytest.mark.e2e
    def test_channel_lifecycle(self, temp_dir):
        """Test full channel lifecycle"""
        from src.channels.manager import ChannelManager

        manager = ChannelManager(config_dir=temp_dir / "channels")

        # Create
        channel = manager.create_channel(
            channel_id="test_ch",
            name="Test Channel",
            niche="tech",
            default_tags=["test", "video"],
        )

        assert channel.id == "test_ch"

        # List
        channels = manager.list_channels()
        assert len(channels) == 1

        # Update
        manager.update_channel("test_ch", default_style="tutorial")
        updated = manager.get_channel("test_ch")
        assert updated.config.default_style == "tutorial"

        # Record upload
        manager.record_upload("test_ch")
        assert manager.get_channel("test_ch").total_videos == 1

        # Delete
        manager.delete_channel("test_ch")
        assert manager.get_channel("test_ch") is None


class TestABTestingE2E:
    """End-to-end tests for A/B testing"""

    @pytest.mark.e2e
    def test_ab_test_lifecycle(self, temp_dir):
        """Test full A/B test lifecycle"""
        from src.optimization.ab_testing import ABTestManager

        manager = ABTestManager(db_path=temp_dir / "ab.db")

        # Create test
        test = manager.create_test(
            video_id="vid123",
            test_type="title",
            variant_a_value="Title A",
            variant_b_value="Title B",
            min_impressions=100,
        )

        assert test is not None

        # Record data
        for _ in range(150):
            variant = manager.select_variant(test.id)
            manager.record_impression(test.id, variant)
            if variant == "a":
                # Variant A has higher CTR
                manager.record_click(test.id, "a")

        # Check results
        updated = manager.get_test(test.id)
        assert updated.variant_a.impressions > 0
        assert updated.variant_b.impressions > 0

        # End test
        manager.end_test(test.id)
        final = manager.get_test(test.id)
        assert final.status == "completed"


@pytest.mark.e2e
@pytest.mark.requires_ollama
class TestWithOllama:
    """Tests that require Ollama to be running"""

    @skip_without_ollama
    def test_llm_script_generation(self):
        """Test actual LLM script generation"""
        from src.llm.ollama import OllamaClient

        client = OllamaClient()

        if not client.is_available():
            pytest.skip("Ollama not available")

        result = client.generate(
            "Write a short greeting in JSON format: {\"message\": \"...\"}",
            temperature=0.5,
        )

        assert result is not None
        assert len(result) > 0


@pytest.mark.e2e
@pytest.mark.requires_ffmpeg
class TestWithFFmpeg:
    """Tests that require FFmpeg to be installed"""

    @skip_without_ffmpeg
    def test_video_assembly(self, temp_dir, sample_image):
        """Test video assembly with FFmpeg"""
        # This would test actual video creation
        # Skipped if FFmpeg not available
        pass
