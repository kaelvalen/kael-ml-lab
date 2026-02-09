#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# D2L Project Setup Script - Linux
# ============================================================

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENV_NAME="d2l"
PYTHON_VERSION="3.11"

echo -e "${CYAN}=======================================${NC}"
echo -e "${CYAN}  D2L Project Setup - Linux            ${NC}"
echo -e "${CYAN}=======================================${NC}"
echo ""

# ── 1. Conda kontrolü ──────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo -e "${RED}[HATA] Conda bulunamadı. Lütfen önce Miniconda/Anaconda kurun.${NC}"
    echo "  → https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ── 2. Conda ortamı oluştur ────────────────────────────────
echo -e "${YELLOW}[1/6]${NC} Conda ortamı oluşturuluyor: ${ENV_NAME} (Python ${PYTHON_VERSION})"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "  → Ortam zaten mevcut, güncelleniyor..."
    conda install -n ${ENV_NAME} python=${PYTHON_VERSION} -y -q
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y -q
fi

# ── 3. Ortamı aktifle ──────────────────────────────────────
echo -e "${YELLOW}[2/6]${NC} Ortam aktifleştiriliyor..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# ── 4. PyTorch + CUDA kur ──────────────────────────────────
echo -e "${YELLOW}[3/6]${NC} PyTorch + CUDA kuruluyor..."
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d'.' -f1,2)
    echo -e "  → CUDA ${CUDA_VER} tespit edildi"
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y -q
else
    echo -e "  → CUDA bulunamadı, CPU sürümü kuruluyor..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y -q
fi

# ── 5. Python bağımlılıkları ───────────────────────────────
echo -e "${YELLOW}[4/6]${NC} Python bağımlılıkları kuruluyor..."
pip install -r requirements.txt -q

# ── 6. Proje paketini kur ──────────────────────────────────
echo -e "${YELLOW}[5/6]${NC} Proje paketi kuruluyor (editable mode)..."
pip install -e . -q

# ── 7. CUDA build ──────────────────────────────────────────
echo -e "${YELLOW}[6/6]${NC} CUDA modülleri kontrol ediliyor..."
if command -v nvcc &> /dev/null && command -v cmake &> /dev/null; then
    echo -e "  → CUDA modülleri derleniyor..."
    mkdir -p cuda/build
    cd cuda/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd ../..
    echo -e "  ${GREEN}✓ CUDA modülleri derlendi${NC}"
else
    echo -e "  → nvcc veya cmake bulunamadı, CUDA build atlanıyor"
fi

# ── Doğrulama ──────────────────────────────────────────────
echo ""
echo -e "${CYAN}=======================================${NC}"
echo -e "${CYAN}  Kurulum Kontrolü                     ${NC}"
echo -e "${CYAN}=======================================${NC}"

python -c "
import torch
print(f'  PyTorch      : {torch.__version__}')
print(f'  CUDA Mevcut  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Sürümü  : {torch.version.cuda}')
    print(f'  GPU          : {torch.cuda.get_device_name(0)}')
try:
    import d2l
    print(f'  d2l          : {d2l.__version__}')
except:
    print('  d2l          : kurulu değil')
print()
print('  ✓ Kurulum tamamlandı!')
"

echo ""
echo -e "${GREEN}Kullanım:${NC}"
echo -e "  conda activate ${ENV_NAME}"
echo -e "  jupyter lab"
echo ""
