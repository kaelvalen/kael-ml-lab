# 🧪 Test Suite

D2L Custom projesi için kapsamlı test paketi.

## 📁 Test Yapısı

```
tests/
├── conftest.py          # pytest konfigürasyonu ve ortak fixture'lar
├── test_models.py       # Model testleri
├── test_utils.py        # Yardımcı fonksiyon testleri
├── test_training.py     # Eğitim fonksiyonları testleri
└── README.md           # Bu dosya
```

## 🚀 Testleri Çalıştırma

### Tüm Testler
```bash
make test
# veya
pytest tests/ -v
```

### Belirli Bir Test Dosyası
```bash
pytest tests/test_models.py -v
```

### Belirli Bir Test Sınıfı
```bash
pytest tests/test_models.py::TestLinearRegression -v
```

### Coverage Raporu ile
```bash
pytest tests/ --cov=src --cov-report=html
```

### CUDA Testleri
```bash
pytest tests/ -k "cuda" -v
```

## 📊 Test Kategorileri

### 🧠 Model Testleri (`test_models.py`)
- **LinearRegression**: Doğrusal regresyon modeli
- **SoftmaxRegression**: Sınıflandırma modeli  
- **MLP**: Çok katmanlı algılayıcı
- **ResidualBlock**: Residual blok implementasyonu

### 🛠️ Util Testleri (`test_utils.py`)
- **Timer**: Zamanlama araçları
- **Accumulator**: Metrik biriktirici
- **Device Utils**: GPU/CPU yönetimi
- **Model Utils**: Model yardımcı fonksiyonları

### 🏋️ Training Testleri (`test_training.py`)
- **train_epoch**: Tek epoch eğitim
- **evaluate**: Model değerlendirme
- **accuracy_count**: Doğruluk hesaplama
- **train**: Tam eğitim döngüsü
- **Data Functions**: Veri yükleme ve işleme

## 🔧 Test Konfigürasyonu

### Fixture'lar
- `torch_seed`: Tekrarlanabilir testler için seed
- `cpu_device`: CPU cihazı
- `gpu_device`: GPU cihazı (mevcutsa)
- `set_default_dtype`: Varsayılan veri tipi

### Test Ortamı
- Python 3.11+
- PyTorch 2.1.0+
- pytest 7.4.0+

## 📝 Test Yazma İpuçları

### Yeni Test Ekleme
1. Test fonksiyonlarını `test_` ön eki ile adlandırın
2. Test sınıflarını `Test` ile başlatın
3. Anlaşılır test adları kullanın
4. Assertion mesajları ekleyin

### Örnek Test
```python
def test_model_forward_shape():
    """Test model forward pass shape."""
    model = LinearRegression(in_features=10, out_features=1)
    X = torch.randn(32, 10)
    y_hat = model(X)
    assert y_hat.shape == (32, 1), f"Expected (32, 1), got {y_hat.shape}"
```

### CUDA Testleri
```python
def test_model_on_cuda():
    """Test model on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = LinearRegression(10, 1).cuda()
    X = torch.randn(16, 10).cuda()
    y = model(X)
    assert y.is_cuda
```

## 🐛 Hata Ayıklama

### Test Hatalarını Görme
```bash
pytest tests/ -v --tb=long
```

### Belirli Bir Testi Çalıştırma
```bash
pytest tests/test_models.py::test_linear_regression -v -s
```

### Debug Mode
```bash
pytest tests/ --pdb
```

## 📈 Coverage

Coverage raporu oluşturmak için:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

HTML raporu için:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## 🔄 CI/CD

Testler GitHub Actions'da otomatik çalıştırılır:
- Python 3.11 ve 3.12
- CPU ve CUDA ortamları
- Code quality checks
- Coverage reporting
