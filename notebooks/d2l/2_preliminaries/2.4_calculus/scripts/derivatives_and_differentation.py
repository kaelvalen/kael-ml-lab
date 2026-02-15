import torch
import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return x ** 3 + 2 * x ** 2 + 3 * x + 4

# Method 1: Using autograd for a single point
x = torch.tensor(3.0, requires_grad=True)
y = f(x)
y.backward()  # dy/dx = 2x = 4
print(x.grad)  # tensor(4.)

# Method 2: Computing derivative function
def f_prime(x):
    x = x.clone().detach().requires_grad_(True)
    y = f(x)
    y.backward()
    return x.grad

# Method 3: Symbolic differentiation (analytical)
# For f(x) = x^2, f'(x) = 2x
def analytical_derivative(x):
    return 2 * x