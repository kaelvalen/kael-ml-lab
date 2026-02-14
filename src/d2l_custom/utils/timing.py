"""
Zamanlama ve performans araçları.
"""

import time
import numpy as np


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
