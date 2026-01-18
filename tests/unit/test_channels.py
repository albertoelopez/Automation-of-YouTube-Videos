"""Unit tests for channel management module"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.channels.manager import (
    ChannelManager,
    Channel,
    ChannelConfig,
    create_sample_channels,
)


class TestChannelConfig:
    """Tests for ChannelConfig dataclass"""

    def test_create_config(self):
        """Test creating a channel config"""
        config = ChannelConfig(
            name="Test Channel",
            niche="tech",
        )

        assert config.name == "Test Channel"
        assert config.niche == "tech"
        assert config.language == "en"
        assert config.voice == "en_US-ryan-medium"

    def test_config_defaults(self):
        """Test default values"""
        config = ChannelConfig(name="Test", niche="test")

        assert config.default_style == "educational"
        assert config.default_duration == 30
        assert config.auto_upload is False
        assert config.default_privacy == "private"
        assert config.max_posts_per_day == 3

    def test_config_custom_values(self):
        """Test custom values"""
        config = ChannelConfig(
            name="Gaming Channel",
            niche="gaming",
            language="es",
            voice="es_ES-davefx-medium",
            default_style="entertaining",
            default_duration=45,
            auto_upload=True,
            default_tags=["gaming", "tips"],
        )

        assert config.language == "es"
        assert config.voice == "es_ES-davefx-medium"
        assert config.default_tags == ["gaming", "tips"]


class TestChannel:
    """Tests for Channel dataclass"""

    def test_create_channel(self):
        """Test creating a channel"""
        config = ChannelConfig(name="Test", niche="test")
        channel = Channel(id="test_channel", config=config)

        assert channel.id == "test_channel"
        assert channel.config.name == "Test"
        assert channel.total_videos == 0
        assert channel.active is True
        assert channel.last_upload is None

    def test_channel_created_at(self):
        """Test that created_at is set"""
        config = ChannelConfig(name="Test", niche="test")
        channel = Channel(id="test", config=config)

        assert channel.created_at is not None
        assert isinstance(channel.created_at, datetime)


class TestChannelManager:
    """Tests for ChannelManager class"""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create manager with temp config dir"""
        return ChannelManager(config_dir=temp_dir / "channels")

    def test_init_creates_config_dir(self, temp_dir):
        """Test that init creates config directory"""
        config_dir = temp_dir / "channel_configs"
        manager = ChannelManager(config_dir=config_dir)

        assert config_dir.exists()

    def test_create_channel(self, manager):
        """Test creating a channel"""
        channel = manager.create_channel(
            channel_id="test_ch",
            name="Test Channel",
            niche="tech",
        )

        assert channel is not None
        assert channel.id == "test_ch"
        assert channel.config.name == "Test Channel"

    def test_create_channel_with_options(self, manager):
        """Test creating channel with all options"""
        channel = manager.create_channel(
            channel_id="gaming_ch",
            name="Gaming Channel",
            niche="gaming",
            language="es",
            voice="es_ES-davefx-medium",
            default_style="entertaining",
            default_tags=["gaming", "tips"],
        )

        assert channel.config.language == "es"
        assert channel.config.voice == "es_ES-davefx-medium"
        assert "gaming" in channel.config.default_tags

    def test_create_duplicate_channel_raises(self, manager):
        """Test that creating duplicate channel raises error"""
        manager.create_channel("test", "Test", "test")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_channel("test", "Test 2", "test")

    def test_get_channel(self, manager):
        """Test retrieving a channel"""
        created = manager.create_channel("ch1", "Channel 1", "tech")

        retrieved = manager.get_channel("ch1")

        assert retrieved is not None
        assert retrieved.id == "ch1"
        assert retrieved.config.name == "Channel 1"

    def test_get_nonexistent_channel(self, manager):
        """Test retrieving non-existent channel"""
        result = manager.get_channel("nonexistent")

        assert result is None

    def test_list_channels(self, manager):
        """Test listing channels"""
        manager.create_channel("ch1", "Channel 1", "tech")
        manager.create_channel("ch2", "Channel 2", "gaming")
        manager.create_channel("ch3", "Channel 3", "finance")

        channels = manager.list_channels()

        assert len(channels) == 3

    def test_list_channels_active_only(self, manager):
        """Test listing only active channels"""
        manager.create_channel("ch1", "Channel 1", "tech")
        manager.create_channel("ch2", "Channel 2", "gaming")
        manager.set_active("ch2", False)

        active = manager.list_channels(active_only=True)

        assert len(active) == 1
        assert active[0].id == "ch1"

    def test_update_channel(self, manager):
        """Test updating channel config"""
        manager.create_channel("ch1", "Channel 1", "tech")

        manager.update_channel(
            "ch1",
            voice="en_US-lessac-medium",
            default_style="tutorial",
        )

        updated = manager.get_channel("ch1")

        assert updated.config.voice == "en_US-lessac-medium"
        assert updated.config.default_style == "tutorial"

    def test_update_nonexistent_channel_raises(self, manager):
        """Test updating non-existent channel raises error"""
        with pytest.raises(ValueError, match="not found"):
            manager.update_channel("nonexistent", voice="test")

    def test_delete_channel(self, manager):
        """Test deleting a channel"""
        manager.create_channel("ch1", "Channel 1", "tech")

        manager.delete_channel("ch1")

        assert manager.get_channel("ch1") is None

    def test_delete_removes_config_file(self, manager, temp_dir):
        """Test that delete removes config file"""
        manager.create_channel("ch1", "Channel 1", "tech")
        config_file = temp_dir / "channels" / "ch1.json"

        assert config_file.exists()

        manager.delete_channel("ch1")

        assert not config_file.exists()

    def test_set_active(self, manager):
        """Test enabling/disabling channel"""
        manager.create_channel("ch1", "Channel 1", "tech")

        manager.set_active("ch1", False)
        assert manager.get_channel("ch1").active is False

        manager.set_active("ch1", True)
        assert manager.get_channel("ch1").active is True

    def test_record_upload(self, manager):
        """Test recording an upload"""
        manager.create_channel("ch1", "Channel 1", "tech")

        manager.record_upload("ch1")

        channel = manager.get_channel("ch1")
        assert channel.total_videos == 1
        assert channel.last_upload is not None

    def test_can_upload_active_channel(self, manager):
        """Test upload eligibility for active channel"""
        manager.create_channel("ch1", "Channel 1", "tech")

        can_upload, reason = manager.can_upload("ch1")

        assert can_upload is True
        assert reason == "OK"

    def test_can_upload_inactive_channel(self, manager):
        """Test upload eligibility for inactive channel"""
        manager.create_channel("ch1", "Channel 1", "tech")
        manager.set_active("ch1", False)

        can_upload, reason = manager.can_upload("ch1")

        assert can_upload is False
        assert "inactive" in reason.lower()

    def test_can_upload_respects_cooldown(self, manager):
        """Test that upload cooldown is respected"""
        channel = manager.create_channel("ch1", "Channel 1", "tech")
        channel.config.min_hours_between_posts = 4

        # Simulate recent upload
        manager.record_upload("ch1")

        can_upload, reason = manager.can_upload("ch1")

        assert can_upload is False
        assert "wait" in reason.lower()

    def test_get_pipeline_config(self, manager):
        """Test getting pipeline config for channel"""
        manager.create_channel(
            "ch1",
            "Channel 1",
            "tech",
            voice="en_US-lessac-medium",
            default_style="tutorial",
            default_tags=["tech", "tips"],
        )

        config = manager.get_pipeline_config("ch1")

        assert config["voice"] == "en_US-lessac-medium"
        assert config["style"] == "tutorial"
        assert "tech" in config["tags"]

    def test_get_pipeline_config_nonexistent(self, manager):
        """Test pipeline config for non-existent channel"""
        config = manager.get_pipeline_config("nonexistent")

        assert config == {}


class TestChannelManagerPersistence:
    """Tests for channel persistence"""

    def test_channels_persist_across_instances(self, temp_dir):
        """Test that channels persist when creating new manager"""
        config_dir = temp_dir / "persist_channels"

        # Create channel with first instance
        manager1 = ChannelManager(config_dir=config_dir)
        manager1.create_channel("ch1", "Channel 1", "tech")
        manager1.record_upload("ch1")

        # Create new instance and verify data persisted
        manager2 = ChannelManager(config_dir=config_dir)
        channel = manager2.get_channel("ch1")

        assert channel is not None
        assert channel.config.name == "Channel 1"
        assert channel.total_videos == 1

    def test_channel_config_file_format(self, temp_dir):
        """Test that config file is valid JSON"""
        config_dir = temp_dir / "json_test"
        manager = ChannelManager(config_dir=config_dir)

        manager.create_channel(
            "ch1",
            "Test Channel",
            "tech",
            default_tags=["test", "video"],
        )

        config_file = config_dir / "ch1.json"
        assert config_file.exists()

        with open(config_file) as f:
            data = json.load(f)

        assert "config" in data
        assert data["config"]["name"] == "Test Channel"
        assert data["config"]["default_tags"] == ["test", "video"]


class TestChannelGenerateForChannel:
    """Tests for channel-specific video generation"""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create manager with temp config dir"""
        return ChannelManager(config_dir=temp_dir / "channels")

    def test_generate_for_nonexistent_channel(self, manager):
        """Test generation for non-existent channel"""
        result = manager.generate_for_channel("nonexistent", "Test topic")

        assert result is None

    def test_generate_for_channel_with_mock_pipeline(self, manager):
        """Test generation with mocked pipeline"""
        manager.create_channel("ch1", "Test Channel", "tech")

        with patch("src.pipeline.Pipeline") as mock_pipeline_class, \
             patch("src.utils.config.load_config") as mock_config:

            mock_config.return_value = MagicMock()
            mock_config.return_value.tts = MagicMock()
            mock_pipeline = MagicMock()
            mock_pipeline.generate.return_value = MagicMock(video_path="/path/to/video.mp4")
            mock_pipeline_class.return_value = mock_pipeline

            result = manager.generate_for_channel("ch1", "Test topic")

        assert result is not None
        mock_pipeline.generate.assert_called_once()


class TestCreateSampleChannels:
    """Tests for sample channel creation"""

    def test_create_sample_channels(self, temp_dir):
        """Test creating sample channels"""
        with patch("src.channels.manager.ChannelManager") as mock_class:
            mock_manager = MagicMock()
            mock_class.return_value = mock_manager

            create_sample_channels()

        # Should create 3 sample channels
        assert mock_manager.create_channel.call_count == 3
