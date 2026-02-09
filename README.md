# 🚀 D2L — Dive into Deep Learning

> **Dive into Deep Learning** kitabının kapsamlı çalışma reposu.  
> Python, C++ ve CUDA entegrasyonları ile derin öğrenme kavramlarını hem teorik hem pratik olarak keşfedin.

---

## 📁 Proje Yapısı

```
d2l/
├── notebooks/                  # 📓 D2L kitap bölümleri (Jupyter Notebooks)
│   ├── 01_introduction/
│   ├── 02_preliminaries/
│   ├── 03_linear-neural-networks-for-regression/
│   ├── ...
│   └── 21_recommender-systems/
│
├── src/                        # 🐍 Python kaynak kodu
│   ├── d2l_custom/             #    Özel D2L yardımcı modülleri
│   │   ├── models/             #    Model implementasyonları
│   │   ├── data/               #    Veri yükleme ve işleme
│   │   ├── utils/              #    Yardımcı fonksiyonlar
│   │   └── visualization/      #    Grafik ve görselleştirme
│   └── experiments/            #    Deney scriptleri
│
├── cuda/                       # ⚡ C++ / CUDA implementasyonları
│   ├── include/                #    Header dosyaları
│   ├── src/                    #    CUDA kernel ve C++ kaynak dosyaları
│   │   ├── matmul/             #    Matris çarpımı kernelleri
│   │   ├── convolution/        #    Konvolüsyon kernelleri
│   │   ├── attention/          #    Attention mekanizması
│   │   ├── activations/        #    Aktivasyon fonksiyonları
│   │   └── loss/               #    Loss fonksiyonları
│   ├── bindings/               #    pybind11 Python bağlantıları
│   └── tests/                  #    CUDA testleri
│
├── tests/                      # 🧪 Python testleri
├── data/                       # 📊 Veri setleri
├── checkpoints/                # 💾 Model checkpoint'leri
├── docs/                       # 📚 Ek dokümantasyon
│
├── CMakeLists.txt              # C++/CUDA build sistemi
├── Makefile                    # Kısayol komutları
├── pyproject.toml              # Python proje konfigürasyonu
├── requirements.txt            # Python bağımlılıkları
├── setup.sh                    # Linux kurulum scripti
└── setup.bat                   # Windows kurulum scripti
```

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.11+
- CUDA Toolkit 12.x
- CMake 3.24+
- GCC/G++ 11+ veya Clang 14+
- Conda (önerilen)

### Hızlı Kurulum (Linux)

```bash
# Repo'yu klonlayın
git clone https://github.com/<username>/d2l.git && cd d2l

# Ortamı kurun
chmod +x setup.sh && ./setup.sh

# Conda ortamını aktifleyin
conda activate d2l

# CUDA modüllerini derleyin
make cuda-build
```

### Windows

```bat
setup.bat
```

---

## 🐍 Python Kullanımı

```python
# Özel modüllerden import
from d2l_custom.models import LinearRegression
from d2l_custom.utils import Timer, Accumulator
from d2l_custom.visualization import plot_training_curve

# CUDA kernellerini Python'dan kullanma
from d2l_custom.cuda_ops import cuda_matmul, cuda_conv2d
```

---

## ⚡ CUDA Kullanımı

### Derleme

```bash
# Tüm CUDA modüllerini derle
make cuda-build

# Sadece belirli bir modülü derle
cd cuda/build && cmake --build . --target matmul_kernel

# Testleri çalıştır
make cuda-test
```

### Python'dan CUDA Çağırma

```python
import torch
from d2l_custom.cuda_ops import custom_matmul

# GPU'da matris çarpımı
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
C = custom_matmul(A, B)
```

---

## 📓 Notebook'lar

D2L kitabının tüm bölümleri Jupyter Notebook olarak hazırlanmıştır:

| Bölüm | Konu | Durum |
|-------|------|-------|
| 01 | Introduction | 🔲 |
| 02 | Preliminaries | 🔲 |
| 03 | Linear Neural Networks (Regression) | 🔲 |
| 04 | Linear Neural Networks (Classification) | 🔲 |
| 05 | Multilayer Perceptrons | 🔲 |
| 06 | Builder's Guide | 🔲 |
| 07 | Convolutional Neural Networks | 🔲 |
| 08 | Modern CNNs | 🔲 |
| 09 | Recurrent Neural Networks | 🔲 |
| 10 | Modern RNNs | 🔲 |
| 11 | Attention & Transformers | 🔲 |
| 12 | Optimization Algorithms | 🔲 |
| 13 | Computational Performance | 🔲 |
| 14 | Computer Vision | 🔲 |
| 15 | NLP: Pretraining | 🔲 |
| 16 | NLP: Applications | 🔲 |
| 17 | Reinforcement Learning | 🔲 |
| 18 | Gaussian Processes | 🔲 |
| 19 | Hyperparameter Optimization | 🔲 |
| 20 | GANs | 🔲 |
| 21 | Recommender Systems | 🔲 |

---

## 🛠️ Geliştirme

```bash
# Linting
make lint

# Testler
make test

# Jupyter Lab'ı başlat
make notebook
```

---

## 📖 Kaynaklar

- [Dive into Deep Learning](https://d2l.ai/) — Ana kitap
- [D2L PyTorch](https://d2l.ai/chapter_installation/index.html) — PyTorch kurulumu
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — NVIDIA CUDA
- [pybind11](https://pybind11.readthedocs.io/) — C++/Python bağlantısı

---

## 📄 Lisans

Bu proje eğitim amaçlıdır. D2L kitabının içeriği [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) lisansı altındadır.
