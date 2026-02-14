"""
Neural network model implementasyonları.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Çok katmanlı algılayıcı (Multilayer Perceptron)."""

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int],
        out_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_size = in_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net.reshape(X.shape[0], -1)


class ResidualBlock(nn.Module):
    """Artık (Residual) blok."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(X)
        return F.relu(out)
