"""
Veri yükleme ve ön işleme yardımcıları.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


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


def load_data(
    X,
    y,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """Veriyi train/test olarak böl ve DataLoader oluştur.

    Args:
        X: Özellik tensörü
        y: Hedef tensör
        batch_size: Mini-batch boyutu
        train_ratio: Eğitim verisi oranı
        num_workers: DataLoader worker sayısı
        shuffle: Veriyi karıştır

    Returns:
        (train_loader, test_loader) tuple
    """
    dataset = TensorDataset(X, y)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_fashion_mnist(batch_size: int = 256, resize=None, num_workers: int = 4):
    """Fashion-MNIST veri setini yükle.

    Args:
        batch_size: Mini-batch boyutu
        resize: Yeniden boyutlandırma (opsiyonel)
        num_workers: DataLoader worker sayısı

    Returns:
        (train_loader, test_loader) tuple
    """
    import torchvision
    from torchvision import transforms

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )

    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers),
    )
