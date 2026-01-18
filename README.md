# Local YouTube Automation

**Fully local YouTube Shorts automation - No API keys required.**

Generate YouTube Shorts videos entirely on your machine using:
- **Ollama** for script generation (local LLM)
- **Piper TTS** for voice synthesis (local)
- **FFmpeg + MoviePy** for video assembly (local)
- **Stable Diffusion / Pollinations** for image generation
- **Your own media library** (local images/music)

## Features

- 🔒 **100% Local** - No API keys required for core features
- 🎬 **Full Pipeline** - Script → Voice → Video in one command
- 🎨 **Ken Burns Effect** - Professional pan/zoom on images
- 📝 **Auto Subtitles** - Burned-in captions
- 🎵 **Background Music** - Mix voice with music
- 🖼️ **AI Image Generation** - Stable Diffusion or Pollinations fallback
- 📤 **YouTube Upload** - Direct upload with OAuth (optional)
- 📦 **Batch Processing** - Generate multiple videos from a list
- ⚙️ **Configurable** - TOML-based settings

## Quick Start

### 1. Install Dependencies

```bash
# System dependencies
sudo apt install ffmpeg  # Ubuntu/Debian
# or: brew install ffmpeg  # macOS

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install Piper TTS
pip install piper-tts

# Install Python dependencies
cd youtube_automation
pip install -e .

# Optional: YouTube upload support
pip install -e ".[youtube]"

# Optional: Local image generation (requires NVIDIA GPU)
pip install -e ".[diffusers]"
```

### 2. Start Ollama & Pull a Model

```bash
# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull llama3.1:8b
```

### 3. Initialize Project

```bash
python -m src.main init
```

### 4. Add Your Media

Add images to `assets/images/` and music to `assets/music/`.

### 5. Generate a Video

```bash
python -m src.main generate "5 Amazing Facts About Space"
```

## Usage

### CLI Commands

```bash
# Generate a video
python -m src.main generate "Your Topic" --duration 30 --style informative

# Generate with AI images
python -m src.main generate "Your Topic" --generate-images

# Generate and upload to YouTube
python -m src.main generate "Your Topic" --upload --privacy private

# Batch process multiple videos
python -m src.main batch batch_jobs.json

# Create sample batch file
python -m src.main batch-sample

# Check status
python -m src.main status

# List available models
python -m src.main models

# List available voices
python -m src.main voices

# Generate a single image
python -m src.main generate-image "A beautiful sunset over mountains"

# Authenticate with YouTube
python -m src.main youtube-auth

# Upload existing video
python -m src.main upload video.mp4 --title "My Video" --privacy unlisted
```

### Python API

```python
from src.pipeline import Pipeline

pipeline = Pipeline()
result = pipeline.generate(
    topic="10 Python Tips",
    style="educational",
    duration=45,
)

print(f"Video saved to: {result.video_path}")
print(f"Title: {result.script.title}")
```

### Batch Processing

Create a `batch_jobs.json`:

```json
{
    "jobs": [
        {
            "topic": "5 Facts About the Ocean",
            "style": "educational",
            "duration": 30
        },
        {
            "topic": "How to Stay Productive",
            "style": "informative",
            "duration": 45
        }
    ]
}
```

Then run:

```bash
python -m src.main batch batch_jobs.json
```

## Configuration

Edit `config/default.toml`:

```toml
[llm]
model = "llama3.1:8b"  # Ollama model
temperature = 0.7

[tts]
model = "en_US-lessac-medium"  # Piper voice

[video]
width = 1080
height = 1920  # 9:16 for Shorts
fps = 30
enable_ken_burns = true

[subtitles]
enabled = true
font_size = 60
color = "white"
```

## Project Structure

```
youtube_automation/
├── src/
│   ├── llm/          # Ollama LLM integration
│   ├── tts/          # Piper TTS integration
│   ├── video/        # Video assembly (FFmpeg/MoviePy)
│   ├── images/       # AI image generation
│   ├── upload/       # YouTube upload
│   ├── media/        # Local media library
│   ├── utils/        # Config management
│   ├── batch.py      # Batch processing
│   ├── pipeline.py   # Main orchestrator
│   └── main.py       # CLI interface
├── config/
│   └── default.toml  # Configuration
├── assets/
│   ├── images/       # Your images
│   ├── music/        # Background music
│   └── footage/      # Video clips
├── output/           # Generated videos
└── models/           # Downloaded TTS models
```

## API Keys - What's Optional

| Feature | API Key Required? | Notes |
|---------|-------------------|-------|
| Script Generation | ❌ No | Ollama runs locally |
| Voice Synthesis | ❌ No | Piper TTS runs locally |
| Video Assembly | ❌ No | FFmpeg runs locally |
| Image Generation | ❌ No | Pollinations.ai (free, no key) or local SD |
| YouTube Upload | ⚠️ OAuth only | Free, one-time browser auth |

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| Storage | 10GB | 20GB |
| GPU | None | NVIDIA 8GB+ (for local SD) |
| Python | 3.10+ | 3.11+ |

## Available Voices

| Voice | Language | Quality |
|-------|----------|---------|
| en_US-lessac-medium | English (US) | Good |
| en_US-amy-medium | English (US) | Good |
| en_GB-alan-medium | English (UK) | Good |
| es_ES-davefx-medium | Spanish | Good |

More voices: https://rhasspy.github.io/piper-samples/

## YouTube Upload Setup (Optional)

1. Go to https://console.cloud.google.com/
2. Create a project
3. Enable YouTube Data API v3
4. Create OAuth 2.0 credentials (Desktop app)
5. Download `client_secrets.json` to `config/`
6. Run `python -m src.main youtube-auth`

## Troubleshooting

### Ollama not running
```bash
ollama serve
```

### No models available
```bash
ollama pull llama3.1:8b
```

### Piper not found
```bash
pip install piper-tts
```

### FFmpeg not found
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

### Image generation slow
Use Pollinations (default) for fast generation without GPU, or install `diffusers` with NVIDIA GPU for local Stable Diffusion.

## License

MIT License - Feel free to use, modify, and distribute.

## Contributing

Contributions welcome! Please open an issue or PR.

## Roadmap

- [ ] Whisper integration for better subtitle timing
- [ ] More TTS voices and languages
- [ ] TikTok/Instagram Reels upload
- [ ] Video templates
- [ ] Analytics dashboard
