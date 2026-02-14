# 🚀 D2L — Dive into Deep Learning

> **Dive into Deep Learning** kitabının kapsamlı çalışma reposu.  
> Python, C++ ve CUDA entegrasyonları ile derin öğrenme kavramlarını hem teorik hem pratik olarak keşfedin.

---

## 📁 Proje Yapısı

```
d2l/
├── 📓 notebooks/               # D2L kitap bölümleri (Jupyter Notebooks)
│   ├── README.md             # Notebook rehberi
│   └── d2l/                  # Bölümlere göre düzenlenmiş notebook'lar
│       ├── 1_introduction/   # Giriş
│       ├── 2_preliminaries/  # Matematiksel ön bilgiler
│       ├── 3_linear-neural-networks-for-regression/
│       └── ...
│
├── 🐍 src/                    # Python kaynak kodu
│   └── d2l_custom/           # Özel D2L yardımcı modülleri
│       ├── models/           # Model implementasyonları
│       │   ├── base.py       # Temel modeller
│       │   └── neural_networks.py
│       ├── utils/            # Yardımcı fonksiyonlar
│       │   ├── timing.py     # Zamanlama araçları
│       │   ├── device.py     # GPU/CPU yönetimi
│       │   └── model_utils.py
│       ├── training/         # Eğitim yardımcıları
│       │   ├── trainer.py    # Epoch bazlı eğitim
│       │   └── loop.py       # Tam eğitim döngüsü
│       ├── data/             # Veri işleme
│       │   ├── synthetic.py  # Sentetik veri
│       │   └── loaders.py    # Veri yükleyiciler
│       └── visualization/    # Görselleştirme
│
├── ⚡ cuda/                   # C++ / CUDA implementasyonları
│   ├── include/              # Header dosyaları
│   ├── src/                  # CUDA kernel ve C++ kaynak dosyaları
│   ├── bindings/             # pybind11 Python bağlantıları
│   └── tests/                # CUDA testleri
│
├── 🧪 tests/                 # Python testleri
│   ├── conftest.py           # pytest konfigürasyonu
│   ├── test_models.py        # Model testleri
│   ├── test_utils.py         # Yardımcı fonksiyon testleri
│   ├── test_training.py      # Eğitim testleri
│   └── README.md             # Test dokümantasyonu
│
├── 📊 data/                  # Veri setleri
│   ├── raw/                  # Ham veriler
│   ├── processed/            # İşlenmiş veriler
│   ├── cache/                # Önbellek
│   └── README.md             # Veri yönetimi rehberi
│
├── 🛠️ scripts/               # Geliştirme script'leri
│   ├── dev.sh               # Geliştirme ortamı kurulumu
│   └── build.sh             # Build script'i
│
├── 📚 docs/                  # Ek dokümantasyon
├── 💾 checkpoints/           # Model checkpoint'leri
│
├── ⚙️ pyproject.toml         # Python proje konfigürasyonu
├── 📋 requirements.txt       # Python bağımlılıkları
├── 📋 requirements-dev.txt   # Geliştirme bağımlılıkları
├── 📋 requirements-core.txt  # Temel bağımlılıklar
├── 🔨 Makefile              # Kısayol komutları
├── 🐧 setup.sh              # Linux kurulum script'i
└── 🪟 setup.bat             # Windows kurulum script'i
```

---

## ⚙️ Kurulum

### 📋 Gereksinimler

- **Python** 3.11+
- **CUDA Toolkit** 12.x (CUDA için)
- **CMake** 3.24+ (CUDA derleme için)
- **Conda** (önerilen)

### 🚀 Hızlı Kurulum

#### Linux/macOS
```bash
# Repo'yu klonlayın
git clone https://github.com/<username>/d2l.git && cd d2l

# Otomatik kurulum
chmod +x scripts/dev.sh && ./scripts/dev.sh

# Ortamı aktifleyin
conda activate d2l
```

#### Windows
```bat
setup.bat
```

### 🔧 Manuel Kurulum
```bash
# Conda ortamı oluştur
conda create -n d2l python=3.11 -y
conda activate d2l

# PyTorch ile CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Paketi kur
pip install -e ".[dev]"

# CUDA modüllerini derle (isteğe bağlı)
make cuda-build
```

---

## 🐍 Python Kullanımı

### 📦 Modül Import'ları
```python
# Model implementasyonları
from d2l_custom.models import LinearRegression, MLP, ResidualBlock

# Yardımcı fonksiyonlar
from d2l_custom.utils import Timer, Accumulator, try_gpu

# Eğitim yardımcıları
from d2l_custom.training import train, evaluate

# Veri işleme
from d2l_custom.data import synthetic_data, get_fashion_mnist
```

### 🧪 Hızlı Başlangıç
```python
import torch
from d2l_custom.models import LinearRegression
from d2l_custom.training import train
from d2l_custom.data import synthetic_data, load_data

# Sentetik veri oluştur
true_w = torch.tensor([2.0, -3.4])
true_b = 1.2
X, y = synthetic_data(true_w, true_b, 1000)

# Veri yükleyiciler oluştur
train_loader, test_loader = load_data(X, y, batch_size=32)

# Modeli eğit
model = LinearRegression(in_features=2, out_features=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

history = train(
    model, train_loader, test_loader, loss_fn, optimizer,
    num_epochs=10, verbose=True
)
```

---

## ⚡ CUDA Kullanımı

### 🔨 Derleme
```bash
# Tüm CUDA modüllerini derle
make cuda-build

# Testleri çalıştır
make cuda-test

# Temizle
make cuda-clean
```

### 🐍 Python'dan CUDA
```python
import torch
from d2l_custom.cuda_ops import custom_matmul  # CUDA implementasyonu

# GPU'da matris çarpımı
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
C = custom_matmul(A, B)  # Hızlı CUDA implementasyonu
```

---

## 📓 Notebook'lar

### 🗂️ Bölümler
| Bölüm | Konu | Durum |
|-------|------|-------|
| 01 | Introduction | � Mevcut |
| 02 | Preliminaries | � Mevcut |
| 03 | Linear Neural Networks (Regression) | � Mevcut |
| 04 | Linear Neural Networks (Classification) | � Mevcut |
| 05 | Multilayer Perceptrons | � Mevcut |
| 06 | Builder's Guide | � Mevcut |
| 07 | Convolutional Neural Networks | � Mevcut |
| 08 | Modern CNNs | � Mevcut |
| 09 | Recurrent Neural Networks | � Mevcut |
| 10 | Modern RNNs | � Mevcut |
| 11 | Attention & Transformers | � Mevcut |
| 12 | Optimization Algorithms | � Mevcut |
| 13 | Computational Performance | � Mevcut |
| 14 | Computer Vision | � Mevcut |
| 15 | NLP: Pretraining | � Mevcut |
| 16 | NLP: Applications | � Mevcut |
| 17 | Reinforcement Learning | � Mevcut |
| 18 | Gaussian Processes | � Mevcut |
| 19 | Hyperparameter Optimization | � Mevcut |
| 20 | GANs | � Mevcut |
| 21 | Recommender Systems | � Mevcut |

### 🚀 Notebook Çalıştırma
```bash
# Jupyter Lab başlat
make notebook

# Belirli bir bölüm
jupyter lab notebooks/d2l/3_linear-neural-networks-for-regression/
```

---

## 🛠️ Geliştirme

### 🧪 Testler
```bash
# Tüm testleri çalıştır
make test

# Coverage ile
pytest tests/ --cov=src --cov-report=html

# Belirli bir test
pytest tests/test_models.py -v
```

### 🔍 Code Quality
```bash
# Linting
make lint

# Formatlama
make format

# Type checking
mypy src/
```

### 📦 Build
```bash
# Build script'i
./scripts/build.sh

# Manuel build
python -m build

# Yükleme
pip install -e ".[dev]"
```

---

## 📊 Veri Yönetimi

### 📥 Veri Setleri
```python
from d2l_custom.data import get_fashion_mnist, synthetic_data

# Fashion-MNIST
train_loader, test_loader = get_fashion_mnist(batch_size=256)

# Sentetik veri
X, y = synthetic_data(w, b, num_examples=1000)
```

### 📁 Veri Dizini
```
data/
├── raw/           # Ham veriler (indirilen)
├── processed/     # İşlenmiş veriler
├── cache/         # Önbellek
└── external/      # Harici kaynaklar
```

---

## 🎯 Özellikler

### ✨ Özel Modüller
- **🧠 Models**: Temel ve ileri neural ağ implementasyonları
- **🛠️ Utils**: GPU yönetimi, zamanlama, metrikler
- **🏋️ Training**: Eğitim döngüleri, değerlendirme
- **📊 Data**: Veri yükleme, sentetik veri üretimi
- **📈 Visualization**: Eğitim görselleştirme

### ⚡ Performans
- **CUDA Entegrasyonu**: Özel CUDA kernelleri
- **Memory Efficient**: Optimize edilmiş veri işleme
- **Parallel Processing**: Çoklu GPU desteği
- **Caching**: Akıllı önbellekleme

### 🔧 Geliştirme Araçları
- **Type Hints**: Tam tip desteği
- **Testing**: Kapsamlı test paketi
- **Documentation**: Detaylı dokümantasyon
- **CI/CD**: Otomatik test ve build

---

## 📖 Kaynaklar

### 📚 Ana Kaynaklar
- [Dive into Deep Learning](https://d2l.ai/) — Ana kitap
- [D2L PyTorch](https://d2l.ai/chapter_installation/index.html) — PyTorch kurulumu
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — NVIDIA CUDA
- [pybind11](https://pybind11.readthedocs.io/) — C++/Python bağlantısı

### 🛠️ Teknolojiler
- **PyTorch** — Deep learning framework
- **CUDA** — GPU computing
- **C++** — Yüksek performanslı implementasyon
- **Jupyter** — Interactive development
- **pytest** — Testing framework

---

## 🤝 Katkı

### � Katkı Rehberi
1. Fork this repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### 🧪 Geliştirme Akışı
```bash
# Geliştirme ortamı kur
./scripts/dev.sh

# Yeni özellik geliştir
# ... kod değişiklikleri ...

# Test et
make test
make lint

# Build et
./scripts/build.sh
```

---

## �📄 Lisans

Bu proje eğitim amaçlıdır. D2L kitabının içeriği [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) lisansı altındadır.

---

## 🙏 Teşekkürler

- D2L yazarlarına ve topluluğuna
- PyTorch geliştiricilerine  
- NVIDIA CUDA ekibine
- Açık kaynak katkıcılarına

---

<div align="center">

**🚀 Happy Learning! 🧠**

Made with ❤️ for Deep Learning community

</div>
