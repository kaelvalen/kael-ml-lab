@echo off
:: ============================================================
:: D2L Project Setup Script - Windows
:: ============================================================

echo =========================================
echo   D2L Project Setup - Windows
echo =========================================
echo.

set ENV_NAME=d2l
set PYTHON_VERSION=3.11

:: ── 1. Conda kontrolü ──────────────────────────────────────
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [HATA] Conda bulunamadi. Lutfen once Miniconda/Anaconda kurun.
    echo   -^> https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: ── 2. Conda ortami olustur ────────────────────────────────
echo [1/6] Conda ortami olusturuluyor: %ENV_NAME%
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y -q
if %errorlevel% neq 0 exit /b %errorlevel%

:: ── 3. Ortami aktifle ──────────────────────────────────────
echo [2/6] Ortam aktiflestiriliyor...
call conda activate %ENV_NAME%

:: ── 4. PyTorch + CUDA kur ──────────────────────────────────
echo [3/6] PyTorch + CUDA kuruluyor...
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y -q
if %errorlevel% neq 0 (
    echo   -^> CUDA bulunamadi, CPU surumu kuruluyor...
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y -q
)

:: ── 5. Python bagimliliklari ───────────────────────────────
echo [4/6] Python bagimliliklari kuruluyor...
pip install -r requirements.txt -q

:: ── 6. Proje paketini kur ──────────────────────────────────
echo [5/6] Proje paketi kuruluyor...
pip install -e . -q

:: ── 7. CUDA build (opsiyonel) ──────────────────────────────
echo [6/6] CUDA modulleri kontrol ediliyor...
where nvcc >nul 2>nul
if %errorlevel% equ 0 (
    where cmake >nul 2>nul
    if %errorlevel% equ 0 (
        echo   -^> CUDA modulleri derleniyor...
        if not exist cuda\build mkdir cuda\build
        cd cuda\build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release
        cd ..\..
        echo   [OK] CUDA modulleri derlendi
    ) else (
        echo   -^> cmake bulunamadi, CUDA build atlanıyor
    )
) else (
    echo   -^> nvcc bulunamadi, CUDA build atlanıyor
)

:: ── Dogrulama ──────────────────────────────────────────────
echo.
echo =========================================
echo   Kurulum Kontrolu
echo =========================================
python -c "import torch; print(f'PyTorch      : {torch.__version__}'); print(f'CUDA Mevcut  : {torch.cuda.is_available()}'); print(f'GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Yok\"}'); import d2l; print(f'd2l          : {d2l.__version__}')"
echo.
echo [OK] Kurulum tamamlandi!
echo.
echo Kullanim:
echo   conda activate %ENV_NAME%
echo   jupyter lab
echo.
pause
