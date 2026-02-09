#!/usr/bin/env bash
# ============================================================
# D2L Project - Cleanup Script
# ============================================================
# Bu script, geçici ve gereksiz dosyaları temizler.

set -e

echo "🧹 D2L Proje Temizleme"
echo "====================="
echo ""

# Python cache
echo "→ Python cache temizleniyor..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Jupyter checkpoints
echo "→ Jupyter checkpoints temizleniyor..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# Build artifacts
echo "→ Build dosyaları temizleniyor..."
rm -rf build/ dist/ 2>/dev/null || true
rm -rf cuda/build/ 2>/dev/null || true
rm -rf .pytest_cache/ .ruff_cache/ 2>/dev/null || true

# Logs
echo "→ Log dosyaları temizleniyor..."
find . -type f -name "*.log" -delete 2>/dev/null || true

# Eski placeholder dosyalar (eğer varsa)
echo "→ Initialize placeholder dosyaları temizleniyor..."
find . -name "initialize_folder_for_github" -type f -delete 2>/dev/null || true

# Mac DS_Store
echo "→ .DS_Store dosyaları temizleniyor..."
find . -name ".DS_Store" -delete 2>/dev/null || true

echo ""
echo "✅ Temizlik tamamlandı!"
echo ""
echo "Kalan büyük dosyalar:"
du -sh data/ checkpoints/ 2>/dev/null || echo "  (veri dizinleri boş)"
