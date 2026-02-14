"""
Model yardımcı fonksiyonları.
"""

import torch
from torch import nn


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
