"""
Model eğitimcisi ve yardımcı fonksiyonlar.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from ..utils import Timer, Accumulator


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Tek bir eğitim epoch'u çalıştır.

    Args:
        model: Eğitilecek model
        train_loader: Eğitim veri yükleyici
        loss_fn: Kayıp fonksiyonu
        optimizer: Optimizör
        device: Hesaplama cihazı

    Returns:
        (ortalama_loss, doğruluk) tuple
    """
    model.train()
    metric = Accumulator(3)  # loss toplamı, doğru tahmin, toplam örnek

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metric.add(
                loss.item() * X.shape[0],
                accuracy_count(y_hat, y),
                X.shape[0],
            )

    return metric[0] / metric[2], metric[1] / metric[2]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> tuple[float, float]:
    """Modeli değerlendir.

    Args:
        model: Değerlendirilecek model
        data_loader: Veri yükleyici
        loss_fn: Kayıp fonksiyonu
        device: Hesaplama cihazı

    Returns:
        (ortalama_loss, doğruluk) tuple
    """
    model.eval()
    metric = Accumulator(3)

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        metric.add(loss.item() * X.shape[0], accuracy_count(y_hat, y), X.shape[0])

    return metric[0] / metric[2], metric[1] / metric[2]


def accuracy_count(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """Doğru tahmin sayısını hesapla."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        preds = y_hat.argmax(dim=1)
    else:
        preds = y_hat.reshape(-1)
    cmp = preds.type(y.dtype) == y
    return float(cmp.sum())
