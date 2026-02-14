#!/bin/bash
# Hızlı başlangıç script'i - günlük kullanım için

set -e

echo "🚀 D2L Quick Start"

# Conda ortamını kontrol et ve aktif et
if [ "$CONDA_DEFAULT_ENV" != "d2l" ]; then
    echo "📦 Activating d2l environment..."
    conda activate d2l 2>/dev/null || {
        echo "❌ d2l environment not found. Run ./scripts/dev.sh first"
        exit 1
    }
fi

# GPU durumunu kontrol et
echo "🔍 Checking GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "⚠️  NVIDIA GPU not detected"

# Jupyter Lab'ı başlat
echo "📓 Starting Jupyter Lab..."
echo "📍 Notebook directory: notebooks/d2l/"
echo "🌐 Access at: http://localhost:8888"
echo ""
echo "💡 Quick commands:"
echo "   make test        - Run tests"
echo "   make lint        - Check code quality"
echo "   make help        - See all commands"
echo ""

jupyter lab --no-browser --notebook-dir=notebooks/d2l/
