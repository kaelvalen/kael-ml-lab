#!/bin/bash
# Development environment setup script

set -e

echo "🚀 D2L Development Environment Setup"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create and activate conda environment
echo "📦 Creating conda environment..."
conda create -n d2l python=3.11 -y
conda activate d2l

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements-core.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in editable mode
echo "🔧 Installing d2l-custom package..."
pip install -e ".[dev]"

# Build CUDA modules if available
if command -v nvcc &> /dev/null; then
    echo "⚡ Building CUDA modules..."
    make cuda-build
else
    echo "⚠️  CUDA compiler not found. Skipping CUDA build."
fi

# Run tests to verify installation
echo "🧪 Running tests..."
make test

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   conda activate d2l"
echo "   make notebook    # Start Jupyter Lab"
echo "   make test        # Run tests"
echo "   make lint        # Check code quality"
