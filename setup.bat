@echo off
echo ===============================
echo d2l Starting Script - Windows
echo ===============================

:: 1️⃣ Ortam oluştur
conda create -n d2l_env python=3.11 -y

:: 2️⃣ Ortamı aktif et
call conda activate d2l_env

:: 3️⃣ PyTorch + CUDA 12.2 yükle
conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia -y

:: 4️⃣ d2l yükle
pip install d2l

:: 5️⃣ Kurulum kontrolü
python -c "import torch; import d2l; print('CUDA Version:', torch.version.cuda, '| GPU available:', torch.cuda.is_available(), '| d2l version:', d2l.__version__)"

echo ===============================
echo Kurulum tamamlandi!
echo Ortamı kullanmak icin: conda activate d2l_env
echo ===============================
pause