# 🚀 D2L Hızlı Başlangıç Cheat Sheet

## 🎯 Günlük Kullanım

### Ortam Başlatma
```bash
# Hızlı başlangıç (notebook + ortam kontrolü)
make quick

# Manuel başlangıç
conda activate d2l
make notebook
```

### Geliştirme Akışı
```bash
# Test çalıştır
make test

# Kod kalitesi kontrolü
make lint

# Kod formatlama
make format

# Temizlik
make clean
```

### GPU ve CUDA
```bash
# GPU bilgisi
make gpu-info

# CUDA modüllerini derle
make cuda-build

# CUDA testleri
make cuda-test
```

## 🛠️ Workflow Helper

```bash
# Tüm komutları gör
./scripts/workflow-helper.sh help

# Sık kullanılanlar
./scripts/workflow-helper.sh test
./scripts/workflow-helper.sh lint
./scripts/workflow-helper.sh gpu
./scripts/workflow-helper.sh status
```

## 📓 Notebook Çalışması

```bash
# Belirli bölümde çalış
jupyter lab notebooks/d2l/3_linear-neural-networks-for-regression/

# Tüm notebook'lar
make notebook
```

## 🔧 Alias'lar (isteğe bağlı)

```bash
# Alias'ları yükle
./scripts/aliases.sh

# Kullanım
d2l              # Proje dizinine git
d2l-notebook     # Notebook başlat
d2l-test         # Test çalıştır
d2l-lint         # Lint kontrolü
d2l-quick        # Hızlı başlangıç
d2l-gpu          # GPU bilgisi
```

## 📁 Önemli Dizinler

- `notebooks/d2l/` - D2L notebook'ları
- `src/d2l_custom/` - Python modülleri
- `cuda/` - CUDA implementasyonları
- `data/` - Veri setleri
- `tests/` - Test dosyaları
- `checkpoints/` - Model checkpoint'leri

## 🎯 Workflow Komutları

- `/dev-setup` - Geliştirme ortamı kurulumu
- `/development` - Günlük geliştirme akışı
- `/notebook-workflow` - Notebook çalışma rehberi

## 💡 İpuçları

1. **Her gün başlarken**: `make quick`
2. **Kod değişikliği sonrası**: `make test && make lint`
3. **GPU kontrolü**: `make gpu-info`
4. **Proje durumu**: `./scripts/workflow-helper.sh status`
