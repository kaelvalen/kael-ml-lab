# ============================================================
# D2L Project - Makefile
# ============================================================

.PHONY: help setup install cuda-build cuda-test test lint notebook clean

PYTHON ?= python
CONDA_ENV ?= d2l

help: ## Bu yardım mesajını göster
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Kurulum ────────────────────────────────────────────────
setup: ## Tam ortam kurulumu (conda + pip + cuda)
	chmod +x setup.sh && ./setup.sh

install: ## Python paketini editable modda kur
	pip install -e .

install-dev: ## Geliştirme bağımlılıklarını kur
	pip install -r requirements.txt
	pip install -e ".[dev]"

# ── CUDA ───────────────────────────────────────────────────
cuda-build: ## CUDA modüllerini derle
	@mkdir -p cuda/build
	cd cuda/build && cmake ../.. -DCMAKE_BUILD_TYPE=Release && make -j$$(nproc)
	@echo "\033[32m✓ CUDA modülleri derlendi\033[0m"

cuda-test: cuda-build ## CUDA testlerini çalıştır
	@echo "=== CUDA Testleri ==="
	cd cuda/build && ./cuda/test_matmul
	cd cuda/build && ./cuda/test_activations
	@echo "\033[32m✓ Tüm CUDA testleri tamamlandı\033[0m"

cuda-clean: ## CUDA build dosyalarını temizle
	rm -rf cuda/build

# ── Python ─────────────────────────────────────────────────
test: ## Python testlerini çalıştır
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Kod kalitesi kontrolü
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

format: ## Kodu formatla
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

# ── Notebook ───────────────────────────────────────────────
notebook: ## Jupyter Lab'ı başlat
	jupyter lab --no-browser

# ── Temizlik ───────────────────────────────────────────────
clean: cuda-clean ## Tüm build dosyalarını temizle
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
	@echo "\033[32m✓ Temizlendi\033[0m"

# ── Hızlı Başlangıç ───────────────────────────────────────────
quick: ## Hızlı başlangıç (notebook + ortam kontrolü)
	@./scripts/quick-start.sh

workflow: ## Workflow helper script'i
	@./scripts/workflow-helper.sh help

# ── GPU Bilgisi ────────────────────────────────────────────
gpu-info: ## GPU bilgisini göster
	@nvidia-smi 2>/dev/null || echo "nvidia-smi bulunamadı"
	@echo ""
	@$(PYTHON) -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}'); \
		[print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" \
		2>/dev/null || echo "PyTorch kurulu değil"
