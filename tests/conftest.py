"""Pytest configuration and shared fixtures"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import json


# ============== Path Fixtures ==============

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create output directory structure"""
    output = temp_dir / "output"
    output.mkdir()
    (output / "videos").mkdir()
    (output / "thumbnails").mkdir()
    (output / "audio").mkdir()
    return output


@pytest.fixture
def temp_config_dir(temp_dir):
    """Create config directory with default config"""
    config = temp_dir / "config"
    config.mkdir()

    # Create default config
    default_config = {
        "llm": {"model": "llama3.1:8b", "base_url": "http://localhost:11434"},
        "tts": {"voice": "en_US-ryan-medium"},
        "video": {"width": 1080, "height": 1920, "fps": 30},
    }

    with open(config / "default.toml", "w") as f:
        import toml
        toml.dump(default_config, f)

    return config


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image"""
    from PIL import Image

    img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
    path = temp_dir / "sample_image.png"
    img.save(path)
    return path


@pytest.fixture
def sample_audio(temp_dir):
    """Create a sample audio file (silence)"""
    import numpy as np
    from scipy.io import wavfile

    sample_rate = 44100
    duration = 2  # seconds
    samples = np.zeros(int(sample_rate * duration), dtype=np.int16)

    path = temp_dir / "sample_audio.wav"
    wavfile.write(path, sample_rate, samples)
    return path


# ============== Mock Fixtures ==============

@pytest.fixture
def mock_ollama():
    """Mock Ollama client"""
    with patch("src.llm.ollama.OllamaClient") as mock:
        instance = MagicMock()
        instance.is_available.return_value = True
        instance.generate.return_value = json.dumps({
            "title": "Test Video Title",
            "script": "This is a test script.",
            "hook": "Amazing opening hook!",
            "outline": ["Point 1", "Point 2", "Point 3"],
            "tags": ["test", "video"],
        })
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_httpx():
    """Mock httpx client for network requests"""
    with patch("httpx.Client") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_piper():
    """Mock Piper TTS"""
    with patch("src.tts.piper.PiperTTS") as mock:
        instance = MagicMock()
        instance.is_available.return_value = True
        instance.synthesize.return_value = Path("/tmp/test_audio.wav")
        mock.return_value = instance
        yield instance


# ============== Data Fixtures ==============

@pytest.fixture
def sample_trending_data():
    """Sample trending topics data"""
    return [
        {
            "title": "AI Takes Over Programming",
            "source": "reddit",
            "category": "technology",
            "score": 5000,
            "url": "https://reddit.com/r/technology/123",
        },
        {
            "title": "New Python 3.13 Features",
            "source": "hackernews",
            "category": "tech",
            "score": 300,
            "url": "https://news.ycombinator.com/item?id=456",
        },
        {
            "title": "SpaceX Launch Success",
            "source": "google_trends",
            "category": "trending",
            "score": 100,
            "url": "",
        },
    ]


@pytest.fixture
def sample_content_idea():
    """Sample content idea"""
    return {
        "title": "5 Python Tips You Didn't Know",
        "hook": "Did you know Python can do THIS?",
        "outline": [
            "Walrus operator tricks",
            "F-string debugging",
            "Match statement patterns",
        ],
        "target_audience": "Python developers",
        "tags": ["python", "programming", "tips"],
        "viral_potential": "high",
        "difficulty": "easy",
    }


@pytest.fixture
def sample_channel_config():
    """Sample channel configuration"""
    return {
        "name": "Test Tech Channel",
        "niche": "tech",
        "language": "en",
        "voice": "en_US-ryan-medium",
        "default_style": "educational",
        "default_duration": 30,
        "default_tags": ["tech", "tutorial"],
        "auto_upload": False,
    }


# ============== Helper Functions ==============

def assert_valid_video(path: Path):
    """Assert that a file is a valid video"""
    assert path.exists(), f"Video file does not exist: {path}"
    assert path.suffix in [".mp4", ".webm", ".mov"], f"Invalid video format: {path.suffix}"
    assert path.stat().st_size > 0, "Video file is empty"


def assert_valid_image(path: Path):
    """Assert that a file is a valid image"""
    from PIL import Image

    assert path.exists(), f"Image file does not exist: {path}"
    img = Image.open(path)
    assert img.size[0] > 0 and img.size[1] > 0, "Image has invalid dimensions"


def assert_valid_audio(path: Path):
    """Assert that a file is a valid audio file"""
    assert path.exists(), f"Audio file does not exist: {path}"
    assert path.suffix in [".wav", ".mp3", ".ogg"], f"Invalid audio format: {path.suffix}"
    assert path.stat().st_size > 0, "Audio file is empty"


# ============== Markers ==============

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_ollama: Requires Ollama running")
    config.addinivalue_line("markers", "requires_ffmpeg: Requires FFmpeg installed")


# ============== Skip Conditions ==============

def is_ollama_available():
    """Check if Ollama is running"""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def is_ffmpeg_available():
    """Check if FFmpeg is installed"""
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


# Skip decorators
skip_without_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available"
)

skip_without_ffmpeg = pytest.mark.skipif(
    not is_ffmpeg_available(),
    reason="FFmpeg not available"
)
