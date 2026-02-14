# 📊 Data Directory

D2L projesi için veri setleri ve veri yönetimi.

## 📁 Dizin Yapısı

```
data/
├── raw/                 # Ham, işlenmemiş veri setleri
│   ├── fashion_mnist/   # Fashion-MNIST orijinal verisi
│   ├── cifar10/        # CIFAR-10 veri seti
│   └── custom/         # Özel veri setleri
├── processed/           # Temizlenmiş ve işlenmiş veriler
│   ├── features/       # Özellik matrisleri
│   ├── labels/         # Etiketler
│   └── splits/         # Train/val/test bölümleri
├── external/           # Harici kaynaklardan indirilenler
├── cache/              # Önbelleğe alınmış veriler
└── README.md          # Bu dosya
```

## 🗂️ Veri Setleri

### 📦 Standart Veri Setleri
- **Fashion-MNIST**: Giysi sınıflandırma (60k train, 10k test)
- **CIFAR-10**: Nesne sınıflandırma (50k train, 10k test)
- **MNIST**: El yazısı rakamlar (60k train, 10k test)

### 🔬 Sentetik Veriler
- **Linear Regression**: `synthetic_data()` fonksiyonu ile üretilir
- **Classification**: Yapay sınıflandırma verileri
- **Time Series**: Zaman serisi simülasyonları

### 📝 Özel Veriler
- **Custom Datasets**: Kullanıcı tanımlı veri setleri
- **Research Data**: Araştırma projesi verileri

## 🔄 Veri İşleme Akışı

### 1. Ham Veri Yükleme
```python
from d2l_custom.data import get_fashion_mnist

train_loader, test_loader = get_fashion_mnist(batch_size=256)
```

### 2. Sentetik Veri Üretme
```python
from d2l_custom.data import synthetic_data

w = torch.tensor([2.0, -3.4])
b = 1.2
X, y = synthetic_data(w, b, 1000)
```

### 3. Veri Bölme
```python
from d2l_custom.data import load_data

train_loader, test_loader = load_data(X, y, batch_size=32)
```

## 📋 Veri Formatları

### 🗃️ Desteklenen Formatlar
- **PyTorch Tensors**: `.pt`, `.pth`
- **NumPy Arrays**: `.npy`, `.npz`
- **CSV**: `.csv`, `.tsv`
- **Images**: `.jpg`, `.png`, `.bmp`
- **HDF5**: `.h5`, `.hdf5`

### 📊 Veri Standartları
- **Features**: `(n_samples, n_features)` shape
- **Labels**: `(n_samples,)` veya `(n_samples, 1)` shape
- **Images**: `(n_samples, channels, height, width)`
- **Sequences**: `(n_samples, seq_len, features)`

## 🛠️ Veri Yönetimi

### 📥 İndirme Script'leri
```bash
# Fashion-MNIST indirme
python scripts/download_fashion_mnist.py

# CIFAR-10 indirme  
python scripts/download_cifar10.py
```

### 🔧 Veri İşleme
```bash
# Veri temizleme
python scripts/clean_data.py --input raw/ --output processed/

# Veri normalizasyonu
python scripts/normalize_data.py --data processed/features/
```

### 📊 İstatistikler
```bash
# Veri seti özeti
python scripts/data_stats.py --data processed/

# Görselleştirme
python scripts/visualize_data.py --data processed/
```

## 💾 Depolama Politikası

### 🚫 Git'e Eklenmeyenler
- Büyük veri dosyaları (>10MB)
- Binary veri setleri
- Önbellek dosyaları
- Model checkpoint'leri

### ✅ Git'e Eklenenler
- `.gitkeep` dosyaları (dizin yapısı için)
- Küçük metadata dosyaları
- Veri işleme script'leri
- README ve dokümantasyon

### 📦 Veri Sürümleme
- **Raw Data**: Sürümlenmez, yeniden indirilir
- **Processed Data**: Sürümlenir (checksum ile)
- **Splits**: Deterministik bölünmeler
- **Metadata**: Tamamen sürümlenir

## 🔍 Veri Kalitesi

### ✅ Kalite Kontrolleri
- **Missing Values**: Eksiz veri kontrolü
- **Data Types**: Veri tipi doğrulaması
- **Range Checks**: Değer aralığı kontrolü
- **Duplicates**: Tekrarlayan veri tespiti

### 📊 İstatistiksel Özetler
- **Mean/Std**: Ortalama ve standart sapma
- **Min/Max**: Minimum ve maksimum değerler
- **Distribution**: Veri dağılımı
- **Correlations**: Özellik korelasyonları

## 🚀 Optimizasyon

### ⚡ Performans İpuçları
- **Memory Mapping**: Büyük dosyalar için `mmap`
- **Lazy Loading**: Gerekli olmadıkça yükleme
- **Caching**: Sık kullanılan verileri önbelleğe al
- **Compression**: Disk alanından tasarruf

### 🗄️ Veri Sıkıştırma
```python
# NumPy sıkıştırma
np.savez_compressed('data.npz', X=X, y=y)

# PyTorch sıkıştırma  
torch.save({'X': X, 'y': y}, 'data.pt', _use_new_zipfile_serialization=True)
```

## 🔐 Güvenlik

### 🛡️ Gizlilik
- **PII Data**: Kişisel bilgiler kaldırılır
- **Sensitive Data**: Hassas veriler şifrelenir
- **Access Control**: Erişim izinleri kontrol edilir

### 📝 Lisanslar
- **Open Data**: Açık veri setleri
- **Academic**: Akademik kullanım için
- **Commercial**: Ticari kullanım kısıtlamaları
