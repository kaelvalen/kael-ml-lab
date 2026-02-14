"""
Tam eğitim döngüsü implementasyonu.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from .trainer import train_epoch, evaluate
from ..utils import Timer


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
