"""
Eğitim (training) yardımcıları.
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


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device = None,
    scheduler=None,
    verbose: bool = True,
) -> dict:
    """Tam eğitim döngüsü.

    Args:
        model: Eğitilecek model
        train_loader: Eğitim verisi
        test_loader: Test verisi
        loss_fn: Kayıp fonksiyonu
        optimizer: Optimizör
        num_epochs: Epoch sayısı
        device: Hesaplama cihazı
        scheduler: LR scheduler (opsiyonel)
        verbose: Çıktı yazdır

    Returns:
        Eğitim geçmişi dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    timer = Timer()

    for epoch in range(num_epochs):
        timer.start()
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        elapsed = timer.stop()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if scheduler:
            scheduler.step()

        if verbose:
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    return history
