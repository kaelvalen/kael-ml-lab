"""
Sentetik veri üretimi.
"""

import torch


def synthetic_data(w, b, num_examples: int):
    """Sentetik doğrusal regresyon verisi üret.

    Args:
        w: Ağırlık vektörü
        b: Bias skaler
        num_examples: Örnek sayısı

    Returns:
        (X, y) tuple
    """
    X = torch.randn(num_examples, len(w))
    y = X @ w + b
    y += torch.randn(y.shape) * 0.01
    return X, y.reshape(-1, 1)
