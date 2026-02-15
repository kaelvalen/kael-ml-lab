import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib_inline import backend_inline

def use_svg_display():
    """Use png format to display plot in jupyter"""
    backend_inline.set_matplotlib_formats('png')

def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
    ylim=None, xscale='linear', yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot X and optional Y with labels and other arguments"""
    
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) 
            and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X) , X
    elif has_one_axis(Y):
        Y = [Y] 
    if len(X) != len(Y):
        X = X * len(Y)
    
    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

x = torch.arange(0, 3, 0.1)
def f(x):
    return 2 ** x - 3

# Calculate tangent line at x=1
x0 = torch.tensor(1.0)
f_x0 = f(x0)  # f(1) = 2^1 - 3 = -1
# Derivative of 2^x - 3 is 2^x * ln(2)
f_prime_x0 = (2 ** x0) * torch.log(torch.tensor(2.0))  # f'(1) = 2 * ln(2)

# Tangent line: y = f(1) + f'(1) * (x - 1)
def tangent_line(x):
    return f_x0 + f_prime_x0 * (x - x0)

plot(x, [f(x), tangent_line(x)], 'x', 'f(x)', legend=['f(x)', 'tangent line at x=1'])

# Add this at the end (after the plot call)
plt.show()