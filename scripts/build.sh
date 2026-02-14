#!/bin/bash
# Build script for D2L project

set -e

echo "🔨 Building D2L Project"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
make clean

# Format and lint code
echo "✨ Formatting code..."
make format

echo "🔍 Running linter..."
make lint

# Run tests
echo "🧪 Running tests..."
make test

# Build CUDA modules if available
if command -v nvcc &> /dev/null; then
    echo "⚡ Building CUDA modules..."
    make cuda-build
    make cuda-test
else
    echo "⚠️  CUDA compiler not found. Skipping CUDA build."
fi

# Build package
echo "📦 Building Python package..."
python -m build

echo "✅ Build completed successfully!"
echo ""
echo "📦 Package artifacts:"
ls -la dist/
