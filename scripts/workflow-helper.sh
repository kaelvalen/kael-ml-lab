#!/bin/bash
# Workflow helper script - sık kullanılan işlemler için kısayollar

set -e

case "${1:-help}" in
    "test")
        echo "🧪 Running tests..."
        make test
        ;;
    "lint")
        echo "🔍 Running lint checks..."
        make lint
        ;;
    "format")
        echo "✨ Formatting code..."
        make format
        ;;
    "gpu")
        echo "🔥 GPU Info:"
        make gpu-info
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        make clean
        ;;
    "build")
        echo "🔨 Building CUDA modules..."
        make cuda-build
        ;;
    "notebook")
        echo "📓 Starting notebook..."
        make notebook
        ;;
    "status")
        echo "📊 Project Status:"
        echo "=================="
        echo "Environment: $CONDA_DEFAULT_ENV"
        echo "Python: $(python --version)"
        echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
        echo "CUDA: $(python -c 'import torch; print("Available" if torch.cuda.is_available() else "Not available")' 2>/dev/null || echo 'Unknown')"
        ;;
    "help"|*)
        echo "🛠️  D2L Workflow Helper"
        echo "======================"
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  test     - Run all tests"
        echo "  lint     - Check code quality"
        echo "  format   - Format code"
        echo "  gpu      - Show GPU information"
        echo "  clean    - Clean build files"
        echo "  build    - Build CUDA modules"
        echo "  notebook - Start Jupyter Lab"
        echo "  status   - Show project status"
        echo "  help     - Show this help"
        ;;
esac
