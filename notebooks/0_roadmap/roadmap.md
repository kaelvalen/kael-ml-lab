# 🗺️ D2L Öğrenim Yol Haritası

> **Dive into Deep Learning** kitabının kapsamlı çalışma rehberi.  
> Python, C++ ve CUDA entegrasyonları ile teoriden pratiğe derin öğrenme.

---

## 📋 Genel Bakış

Bu repo, D2L kitabının sistematik bir çalışmasını içerir. Her bölüm için:
- ✍️ **Teorik notlar** (Markdown + LaTeX)
- 💻 **Pratik kodlar** (Python + PyTorch)
- ⚡ **CUDA implementasyonları** (performans karşılaştırmaları)
- 📊 **Görselleştirmeler**
- 🔬 **Deneyler ve sonuçlar**

---

## 📚 Bölüm Durumu

| # | Bölüm | Notebook | Python | CUDA | Durum |
|---|-------|----------|--------|------|-------|
| **01** | Introduction | 🔲 | 🔲 | ➖ | Başlanmadı |
| **02** | Preliminaries | 🔲 | 🔲 | ✅ | Başlanmadı |
| | 2.1 Data Manipulation | 🔲 | 🔲 | ➖ | |
| | 2.2 Data Preprocessing | 🔲 | 🔲 | ➖ | |
| | 2.3 Linear Algebra | 🔲 | 🔲 | ✅ | matmul kernels |
| | 2.4 Calculus | 🔲 | 🔲 | ➖ | |
| | 2.5 Automatic Differentiation | 🔲 | 🔲 | ➖ | |
| | 2.6 Probability & Statistics | 🔲 | 🔲 | ➖ | |
| **03** | Linear Regression | 🔲 | ✅ | ✅ | Başlanmadı |
| **04** | Linear Classification | 🔲 | ✅ | ✅ | Başlanmadı |
| **05** | Multilayer Perceptrons | 🔲 | ✅ | ✅ | Başlanmadı |
| **06** | Builder's Guide | 🔲 | 🔲 | ➖ | Başlanmadı |
| **07** | Convolutional Neural Networks | 🔲 | ✅ | ✅ | Başlanmadı |
| **08** | Modern CNNs | 🔲 | ✅ | ✅ | Başlanmadı |
| | 8.1 AlexNet | 🔲 | ✅ | ➖ | |
| | 8.2 VGG | 🔲 | ✅ | ➖ | |
| | 8.3 NiN | 🔲 | ✅ | ➖ | |
| | 8.4 GoogLeNet | 🔲 | ✅ | ➖ | |
| | 8.5 Batch Normalization | 🔲 | ✅ | ✅ | |
| | 8.6 ResNet | 🔲 | ✅ | ➖ | |
| | 8.7 DenseNet | 🔲 | ✅ | ➖ | |
| **09** | Recurrent Neural Networks | 🔲 | ✅ | ➖ | Başlanmadı |
| **10** | Modern RNNs | 🔲 | ✅ | ➖ | Başlanmadı |
| | 10.1 LSTM | 🔲 | ✅ | ➖ | |
| | 10.2 GRU | 🔲 | ✅ | ➖ | |
| | 10.5 Seq2Seq | 🔲 | ✅ | ➖ | |
| **11** | Attention & Transformers | 🔲 | ✅ | ✅ | Başlanmadı |
| | 11.3 Attention Mechanisms | 🔲 | ✅ | ✅ | scaled dot-product |
| | 11.5 Multi-Head Attention | 🔲 | ✅ | ✅ | |
| | 11.7 Transformer | 🔲 | ✅ | ➖ | |
| **12** | Optimization | 🔲 | ✅ | ➖ | Başlanmadı |
| **13** | Computational Performance | 🔲 | 🔲 | ✅ | Başlanmadı |
| **14** | Computer Vision | 🔲 | ✅ | ✅ | Başlanmadı |
| **15** | NLP: Pretraining | 🔲 | 🔲 | ➖ | Başlanmadı |
| **16** | NLP: Applications | 🔲 | 🔲 | ➖ | Başlanmadı |
| **17** | Reinforcement Learning | 🔲 | 🔲 | ➖ | Başlanmadı |
| **18** | Gaussian Processes | 🔲 | 🔲 | ➖ | Başlanmadı |
| **19** | Hyperparameter Optimization | 🔲 | 🔲 | ➖ | Başlanmadı |
| **20** | GANs | 🔲 | 🔲 | ➖ | Başlanmadı |
| **21** | Recommender Systems | 🔲 | 🔲 | ➖ | Başlanmadı |

**Durum İkonları:**
- 🔲 Başlanmadı
- 🔄 Devam ediyor
- ✅ Tamamlandı
- ➖ Uygulanamaz

---

## ⚡ CUDA İmplementasyonları

### Tamamlanan Kerneller
- ✅ **Matrix Multiplication** (Naive + Tiled)
- ✅ **Activation Functions** (ReLU, Sigmoid, Tanh, Softmax)
- ✅ **Convolution 2D** (Naive)
- ✅ **Scaled Dot-Product Attention**
- ✅ **Loss Functions** (MSE, Cross-Entropy)

### Planlanan Kerneller
- 🔲 Batch Normalization
- 🔲 Layer Normalization
- 🔲 Dropout
- 🔲 Max/Avg Pooling
- 🔲 Embedding Lookup
- 🔲 LSTM Cell
- 🔲 Optimizers (SGD, Adam)

---

## 🎯 Öğrenim Hedefleri

### Faz 1: Temel Kavramlar (Bölüm 1-6)
**Hedef:** Derin öğrenmenin temel taşlarını anlamak
- [ ] Tensor işlemleri ve PyTorch temelleri
- [ ] Otomatik türev alma (autograd)
- [ ] Doğrusal modeller (regresyon + sınıflandırma)
- [ ] MLP ve backpropagation
- [ ] Overfitting, underfitting, regularization

### Faz 2: Konvolüsyonel Ağlar (Bölüm 7-8)
**Hedef:** Görüntü işleme için özelleşmiş mimariler
- [ ] CNN temelleri (conv, pooling, padding)
- [ ] Modern CNN mimarileri (AlexNet → ResNet → DenseNet)
- [ ] Batch normalization ve diğer normalizasyon teknikleri
- [ ] Transfer learning

**CUDA Proje:** Optimized 2D Convolution

### Faz 3: Diziler için Ağlar (Bölüm 9-10)
**Hedef:** Zamansal verileri modellemek
- [ ] RNN temelleri
- [ ] LSTM ve GRU
- [ ] Bidirectional RNNs
- [ ] Seq2Seq modelleri

### Faz 4: Attention & Transformers (Bölüm 11)
**Hedef:** Modern NLP'nin kalbi
- [ ] Attention mekanizması
- [ ] Multi-head attention
- [ ] Transformer mimarisi
- [ ] Vision Transformers

**CUDA Proje:** Optimized Multi-Head Attention

### Faz 5: Optimizasyon & Performans (Bölüm 12-13)
**Hedef:** Eğitim optimizasyonu ve hızlandırma
- [ ] Optimizasyon algoritmaları (SGD → Adam)
- [ ] Learning rate scheduling
- [ ] GPU programlama ve profiling
- [ ] Model paralelizasyonu

**CUDA Proje:** Custom Optimizer Kernels

### Faz 6: Uygulama Alanları (Bölüm 14-21)
**Hedef:** Gerçek dünya problemleri
- [ ] Computer Vision uygulamaları
- [ ] NLP uygulamaları
- [ ] Reinforcement Learning
- [ ] GANs ve üretken modeller

---

## 🛠️ Projelendirme Stratejisi

Her bölüm için:

### 1. İlk Okuma (1 gün)
- D2L kitabından bölümü oku
- Temel kavramları not al
- Alıştırmaları incele

### 2. Notebook Implementasyonu (2-3 gün)
- Jupyter notebook'ta teorik açıklamalar yaz
- PyTorch ile implementasyon
- Görselleştirmeler ve denemeler
- D2L kitabındaki alıştırmaları çöz

### 3. Python Modül Geliştirme (1 gün)
- Yeniden kullanılabilir kod yazmak için `src/d2l_custom/` altına ekle
- Test yazımı
- Dokümantasyon

### 4. CUDA İmplementasyonu (2-4 gün, opsiyonel)
- Kritik operasyonlar için CUDA kernel yaz
- Performans karşılaştırması (CPU vs GPU)
- Optimizasyon deneyleri

---

## 📈 İlerleme Takibi

### Haftalık Hedefler
- **Hafta 1-2:** Bölüm 1-3 (Temel kavramlar)
- **Hafta 3-4:** Bölüm 4-6 (Classification + MLP)
- **Hafta 5-7:** Bölüm 7-8 (CNNs)
- **Hafta 8-10:** Bölüm 9-11 (RNNs + Attention)
- **Hafta 11-12:** Bölüm 12-13 (Optimization)
- **Hafta 13+:** Bölüm 14-21 (Uygulamalar)

### Milestone'lar
- 🎯 **Milestone 1:** İlk end-to-end sınıflandırma modeli (MNIST)
- 🎯 **Milestone 2:** ResNet implementasyonu ve eğitimi
- 🎯 **Milestone 3:** Özel CUDA kernel kütüphanesi
- 🎯 **Milestone 4:** Transformer scratch'ten implementasyon
- 🎯 **Milestone 5:** Büyük proje (CV veya NLP)

---

## 💡 Kaynaklar

### Ana Kaynaklar
- [D2L Website](https://d2l.ai/) — İnteraktif kitap
- [D2L PyTorch](https://d2l.ai/chapter_installation/index.html) — PyTorch versiyonu

### CUDA Kaynakları
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA'nın CUDA templates
- [Optimizing CUDA Kernels](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

### Community
- [D2L Discussion Forum](https://discuss.d2l.ai/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [CUDA subreddit](https://www.reddit.com/r/CUDA/)

---

## 📝 Notlar

- Her notebook standalone çalışabilir olmalı
- CUDA kernelleri PyTorch fallback'e sahip olmalı
- Code review için her hafta bir checkpoint
- Büyük modeller için checkpoint sistemi kur

---

**Son Güncelleme:** 2026-02-09  
**Repo Versiyonu:** 0.1.0  
**Lisans:** MIT (Educational Use)
