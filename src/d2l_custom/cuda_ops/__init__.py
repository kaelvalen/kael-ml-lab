"""
CUDA operasyonları için Python arayüzü.

Bu modül, C++/CUDA ile yazılmış özel kernellerin Python'dan
kullanılmasını sağlar. CUDA modülleri derlenmemişse,
PyTorch fallback kullanılır.
"""

import torch
import warnings

_cuda_ext = None

try:
    from . import _cuda_kernels as _cuda_ext
except ImportError:
    warnings.warn(
        "CUDA kernelleri yüklenemedi. PyTorch fallback kullanılacak. "
        "CUDA modüllerini derlemek için: make cuda-build",
        RuntimeWarning,
        stacklevel=2,
    )


def custom_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Özel CUDA matris çarpımı.

    CUDA modülü mevcutsa özel kernel kullanır,
    yoksa torch.mm fallback.

    Args:
        A: (M, K) boyutunda tensör
        B: (K, N) boyutunda tensör

    Returns:
        (M, N) boyutunda sonuç tensörü
    """
    if _cuda_ext is not None and A.is_cuda and B.is_cuda:
        return _cuda_ext.matmul(A, B)
    return torch.mm(A, B)


def custom_relu(X: torch.Tensor) -> torch.Tensor:
    """Özel CUDA ReLU aktivasyonu.

    Args:
        X: Giriş tensörü

    Returns:
        ReLU uygulanmış tensör
    """
    if _cuda_ext is not None and X.is_cuda:
        return _cuda_ext.relu(X)
    return torch.relu(X)


def custom_softmax(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Özel CUDA softmax.

    Args:
        X: Giriş tensörü
        dim: Softmax boyutu

    Returns:
        Softmax uygulanmış tensör
    """
    if _cuda_ext is not None and X.is_cuda:
        return _cuda_ext.softmax(X, dim)
    return torch.softmax(X, dim=dim)


def custom_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Özel CUDA 2D konvolüsyon.

    Args:
        input: (N, C_in, H, W) boyutunda giriş tensörü
        weight: (C_out, C_in, kH, kW) boyutunda filtre tensörü
        bias: (C_out,) boyutunda bias tensörü (opsiyonel)
        stride: Adım boyutu
        padding: Padding boyutu

    Returns:
        Konvolüsyon sonucu
    """
    if _cuda_ext is not None and input.is_cuda:
        return _cuda_ext.conv2d(input, weight, bias, stride, padding)
    return torch.nn.functional.conv2d(input, weight, bias, stride=stride, padding=padding)


def benchmark_kernel(func, *args, warmup: int = 10, repeats: int = 100, **kwargs) -> float:
    """CUDA kernel performansını ölç.

    Args:
        func: Ölçülecek fonksiyon
        *args: Fonksiyon argümanları
        warmup: Isınma iterasyonu sayısı
        repeats: Ölçüm tekrar sayısı

    Returns:
        Ortalama süre (ms)
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeats
