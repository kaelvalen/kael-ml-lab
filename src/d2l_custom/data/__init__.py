"""
Veri yükleme ve ön işleme yardımcıları.
"""

from .synthetic import synthetic_data
from .loaders import load_data, get_fashion_mnist

__all__ = [
    "synthetic_data",
    "load_data",
    "get_fashion_mnist",
]
