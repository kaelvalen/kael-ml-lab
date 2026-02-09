"""
Yardımcı fonksiyonlar ve sınıflar.
"""

import time
import numpy as np
import torch
from torch import nn


class Timer:
    """Birden fazla süreyi kaydetmek için zamanlayıcı."""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Zamanlayıcıyı başlat."""
        self.tik = time.time()

    def stop(self):
        """Zamanlayıcıyı durdur ve süreyi kaydet."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Ortalama süreyi döndür."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Toplam süreyi döndür."""
        return sum(self.times)

    def cumsum(self):
        """Kümülatif süreyi döndür."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """n değişken üzerinde toplam biriktirmek için."""

    def __init__(self, n: int):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def count_parameters(model: nn.Module) -> int:
    """Eğitilebilir parametre sayısını döndür."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(module: nn.Module, method: str = "xavier"):
    """Ağırlık başlatma."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if method == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif method == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
