"""
GPU ve cihaz yönetimi araçları.
"""

import torch
from torch import nn


def try_gpu(i: int = 0) -> torch.device:
    """GPU mevcutsa gpu(i) döndür, yoksa cpu döndür."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_all_gpus() -> list[torch.device]:
    """Tüm mevcut GPU'ları döndür, yoksa [cpu] döndür."""
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]


def gpu_info():
    """GPU bilgisini yazdır."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Bellek : {props.total_mem / 1024**3:.1f} GB")
            print(f"  SM     : {props.multi_processor_count}")
            print(f"  Compute: {props.major}.{props.minor}")
    else:
        print("CUDA mevcut değil.")
