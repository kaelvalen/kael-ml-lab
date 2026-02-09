"""
Görselleştirme yardımcıları.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display


def use_svg_display():
    """SVG formatında görselleştirme kullan."""
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(6, 4)):
    """Figür boyutunu ayarla."""
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Eksen özelliklerini ayarla."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(6, 4),
    axes=None,
):
    """Veri noktalarını çiz."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        return hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__")

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def plot_training_curve(
    train_losses: list,
    val_losses: list = None,
    train_accs: list = None,
    val_accs: list = None,
    figsize=(12, 4),
):
    """Eğitim eğrilerini çiz.

    Args:
        train_losses: Eğitim kayıpları
        val_losses: Doğrulama kayıpları (opsiyonel)
        train_accs: Eğitim doğrulukları (opsiyonel)
        val_accs: Doğrulama doğrulukları (opsiyonel)
    """
    set_figsize(figsize)
    num_plots = 1 + (train_accs is not None)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    # Loss grafiği
    axes[0].plot(train_losses, label="Train Loss")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Kayıp Eğrisi")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy grafiği
    if train_accs is not None:
        axes[1].plot(train_accs, label="Train Acc")
        if val_accs:
            axes[1].plot(val_accs, label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Doğruluk Eğrisi")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Görüntü listesini grid olarak göster."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if hasattr(img, "numpy"):
            img = img.numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()
