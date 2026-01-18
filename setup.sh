#!/bin/bash
# Setup script for Local YouTube Automation

set -e

echo "🎬 Local YouTube Automation Setup"
echo "=================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is installed"
else
    echo "✗ FFmpeg not found"
    echo "  Install with: sudo apt install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)"
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
else
    echo "✗ Ollama not found"
    echo "  Install with: curl -fsSL https://ollama.com/install.sh | sh"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e .

# Install Piper TTS
echo "Installing Piper TTS..."
pip install piper-tts

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate venv:     source venv/bin/activate"
echo "  2. Start Ollama:      ollama serve"
echo "  3. Pull a model:      ollama pull llama3.1:8b"
echo "  4. Initialize:        python -m src.main init"
echo "  5. Add images to:     assets/images/"
echo "  6. Generate video:    python -m src.main generate 'Your Topic'"
echo ""
