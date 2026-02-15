import torch
import matplotlib.pyplot as plt
import numpy as np

x , y= torch.tensor(2.0), torch.tensor(3.0)

def f(x):
    return 3 * x ** 2 - 4 * x
for h in 10.0 ** np.arange(-3, 5, 1):
    print(f"h = {h: .5f}, numerical limit = {f(x + h) - f(x) / h: .5f}")
