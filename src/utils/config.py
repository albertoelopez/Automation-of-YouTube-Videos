"""Configuration management"""
import tomllib
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class VideoConfig(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    format: str = "mp4"
    codec: str = "libx264"
    bitrate: str = "8M"
    min_clip_duration: float = 3.0
    max_clip_duration: float = 6.0
    transition_duration: float = 0.5
    enable_ken_burns: bool = True
    ken_burns_zoom: float = 1.2


class AudioConfig(BaseModel):
    sample_rate: int = 22050
    voice_speed: float = 1.0
    silence_padding: float = 0.3


class LLMConfig(BaseModel):
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1000


class TTSConfig(BaseModel):
    model: str = "en_US-lessac-medium"
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8


class SubtitleConfig(BaseModel):
    enabled: bool = True
    font: str = "Arial"
    font_size: int = 60
    color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    position: str = "bottom"


class MediaConfig(BaseModel):
    footage_dir: str = "assets/footage"
    music_dir: str = "assets/music"
    images_dir: str = "assets/images"
    fonts_dir: str = "assets/fonts"


class GeneralConfig(BaseModel):
    output_dir: str = "output"
    temp_dir: str = "temp"
    log_level: str = "INFO"


class Config(BaseSettings):
    general: GeneralConfig = GeneralConfig()
    video: VideoConfig = VideoConfig()
    audio: AudioConfig = AudioConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    subtitles: SubtitleConfig = SubtitleConfig()
    media: MediaConfig = MediaConfig()

    class Config:
        env_prefix = "YTAUTO_"


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration from TOML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.toml"

    config_path = Path(config_path)

    if not config_path.exists():
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return Config(**data)
