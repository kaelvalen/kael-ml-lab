# Katkıda Bulunma Rehberi

D2L projesine katkıda bulunmak için bu rehberi takip edin.

## 🚀 Hızlı Başlangıç

```bash
# Repo'yu fork edin ve klonlayın
git clone https://github.com/<username>/d2l.git
cd d2l

# Geliştirme ortamını kurun
./setup.sh

# Conda ortamını aktifleyin
conda activate d2l

# Development bağımlılıklarını kurun
pip install -e ".[dev]"
```

## 📁 Proje Yapısı

```
d2l/
├── src/d2l_custom/           # Python kaynak kodu
│   ├── models/               # Model implementasyonları
│   ├── data/                 # Veri yükleme
│   ├── utils/                # Yardımcı fonksiyonlar
│   ├── visualization/        # Görselleştirme
│   ├── training/             # Eğitim döngüsü
│   └── cuda_ops/             # CUDA Python arayüzü
│
├── cuda/                     # CUDA/C++ implementasyonları
│   ├── include/              # Header dosyaları
│   ├── src/                  # CUDA kerneller
│   ├── bindings/             # pybind11 bindings
│   └── tests/                # CUDA testleri
│
├── notebooks/                # Jupyter notebooks (D2L bölümleri)
└── tests/                    # Python testleri
```

## 🔧 Geliştirme İş Akışı

### 1. Branch Oluşturma

```bash
git checkout -b feature/new-feature
# veya
git checkout -b fix/bug-fix
```

### 2. Kod Yazma

#### Python Kodu

```python
# src/d2l_custom/models/my_model.py
import torch
from torch import nn

class MyModel(nn.Module):
    """Model açıklaması."""
    
    def __init__(self, ...):
        super().__init__()
        # ...
    
    def forward(self, X):
        return ...
```

#### CUDA Kodu

```cpp
// cuda/src/my_kernel/my_kernel.cu
#include "my_kernel.cuh"

namespace d2l {
namespace cuda {

__global__ void my_kernel(...) {
    // Kernel implementasyonu
}

void my_function(...) {
    // Host fonksiyonu
    my_kernel<<<grid, block>>>(...);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace d2l
```

### 3. Test Yazma

#### Python Test

```python
# tests/test_my_model.py
import torch
from d2l_custom.models import MyModel

def test_my_model():
    model = MyModel(...)
    X = torch.randn(10, 20)
    y = model(X)
    assert y.shape == (10, 5)
```

#### CUDA Test

```cpp
// cuda/tests/test_my_kernel.cu
#include "my_kernel.cuh"
#include <cassert>

int main() {
    // Test implementasyonu
    // ...
    return 0;
}
```

### 4. Testleri Çalıştırma

```bash
# Python testleri
make test

# CUDA testleri
make cuda-test

# Linting
make lint

# Format
make format
```

### 5. Commit ve Push

```bash
git add .
git commit -m "feat: yeni özellik eklendi"
git push origin feature/new-feature
```

## 📝 Commit Mesaj Formatı

[Conventional Commits](https://www.conventionalcommits.org/) standardını kullanıyoruz:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Tipler:**
- `feat`: Yeni özellik
- `fix`: Bug düzeltmesi
- `docs`: Dokümantasyon
- `style`: Formatlama, noktalı virgül eksikliği vb.
- `refactor`: Refactoring
- `perf`: Performans iyileştirmesi
- `test`: Test ekleme/düzeltme
- `chore`: Build, auxiliary araçlar

**Örnekler:**
```bash
git commit -m "feat(models): ResNet implementasyonu eklendi"
git commit -m "fix(cuda): matmul kernel'inde race condition düzeltildi"
git commit -m "docs: README'ye kurulum adımları eklendi"
```

## 🎨 Kod Stili

### Python

- **PEP 8** standardı
- **Type hints** kullan
- **Docstrings** yaz (Google style)
- Max satır uzunluğu: 100

```python
def my_function(x: torch.Tensor, y: int = 5) -> torch.Tensor:
    """Fonksiyon açıklaması.

    Args:
        x: Giriş tensörü
        y: Parametre açıklaması

    Returns:
        Çıkış tensörü
    """
    return x * y
```

### C++/CUDA

- **Camel case** fonksiyon isimleri
- **ALL_CAPS** constant isimleri
- **snake_case** değişken isimleri
- Namespace kullan: `d2l::cuda`

```cpp
namespace d2l {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

void matmul_kernel(const float* A, float* B, int N) {
    // ...
}

} // namespace cuda
} // namespace d2l
```

## 📖 Dokümantasyon

### Python Docstrings

```python
def train_model(model: nn.Module, data_loader: DataLoader) -> dict:
    """Modeli eğit.

    Args:
        model: Eğitilecek PyTorch modeli
        data_loader: Eğitim veri yükleyici

    Returns:
        Eğitim geçmişi (loss, accuracy vb.)

    Example:
        >>> model = MyModel()
        >>> loader = DataLoader(...)
        >>> history = train_model(model, loader)
    """
    pass
```

### CUDA Fonksiyon Yorumları

```cpp
// ── Matrix Multiplication Kernel ──────────────────────────
// Computes C = A * B where A is MxK, B is KxN, C is MxN
// Uses tiled approach with shared memory
// 
// Args:
//   A: Input matrix A (device pointer)
//   B: Input matrix B (device pointer)
//   C: Output matrix C (device pointer)
//   M, N, K: Matrix dimensions
void matmul_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K);
```

## 🐛 Bug Raporlama

Issue açarken şu bilgileri ekleyin:

1. **Açıklama:** Ne oluyor?
2. **Beklenen davranış:** Ne olmalıydı?
3. **Adımlar:** Nasıl tekrarlanır?
4. **Sistem:**
   - OS: Linux/Windows/Mac
   - Python versiyonu
   - PyTorch versiyonu
   - CUDA versiyonu (varsa)
5. **Hata mesajı:** Tam stack trace

## ✅ Pull Request Checklist

PR göndermeden önce:

- [ ] Testler yazıldı ve geçiyor
- [ ] Dokümantasyon güncellendi
- [ ] Code lint/format kontrolü yapıldı
- [ ] CHANGELOG güncellendi (eğer gerekiyorsa)
- [ ] Commit mesajları standart formatında

## 📚 Kaynak Kodu İnceleyin

Mevcut kodu inceleyerek stil ve yapıyı öğrenin:

```bash
# Python örnek model
cat src/d2l_custom/models/__init__.py

# CUDA örnek kernel
cat cuda/src/matmul/matmul.cu

# Örnek notebook
jupyter lab notebooks/0_roadmap/demo.ipynb
```

## 💬 Sorularınız mı var?

- **GitHub Issues:** Sorunlar ve öneriler için
- **GitHub Discussions:** Genel tartışmalar için
- **Email:** [yintsukuyomi@proton.me](mailto:yintsukuyomi@proton.me)

## 🎉 Teşekkürler!

Katkılarınız için teşekkür ederiz! Her katkı projeyi daha iyi hale getirir.
