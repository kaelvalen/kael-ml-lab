"""
Temel model sınıfları.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LinearRegression(nn.Module):
    """Basit doğrusal regresyon modeli."""

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.net = nn.LazyLinear(out_features)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)


class SoftmaxRegression(nn.Module):
    """Softmax regresyon modeli."""

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_outputs),
        )

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
