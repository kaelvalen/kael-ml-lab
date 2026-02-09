import numpy as np
import torch

x = torch.arange(6).reshape(2, 3)
print('x =', x)
y = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
print('y =', y)
print('x + y =', x + y)

x.numel()  # number of elements in x
y.numel()  # number of elements in y
print('x shape =', x.shape)
print('y shape =', y.shape)

X = x.reshape(3, 2)
print('X =', X)
Y = y.reshape(3, 2)
print('Y =', Y)
print('X + Y =', X + Y)

torch.zeros((2, 3))  # all elements are zeros
torch.ones((2, 3, 4, 5))  # all elements are ones

torch.randn((3, 4))  # random numbers from the standard normal distribution
torch.randn_like(Y)  # random numbers with the same shape as Y

print('X == Y =', X == Y)  # type: ignore # elementwise comparison

print('X[-1] =', X[-1]) # the last row of X
print('Y[1] =', Y[1]) # the second row of Y
print('X[1, 0] =', X[1, 0]) # the first element of the second row of X
print('Y[0, 1] =', Y[0, 1]) # the second element of the first row of Y

X[1, 1] = 9  # set the second element of the second row of X to 9
print('X =', X)
Y[0, 1] = 8  # set the second element of the first row of Y to 8
print('Y =', Y)

X[0:2, 0:2] = 0  # set the first two rows and the first two columns of X to 0
print('X =', X)
Y[0:2, 0:2] = 0  # set the first two rows and the first two columns of Y to 0
print('Y =', Y)

print('torch.exp(X) =', torch.exp(X))  # elementwise exponentiation
print('torch.exp(Y) =', torch.exp(Y))  # elementwise exponentiation

print('X + Y =', X + Y)  # type: ignore # elementwise addition
print('X - Y =', X - Y)  # elementwise subtraction
print('X * Y =', X * Y)  # elementwise multiplication
print('X / Y =', X / Y)  # elementwise division

torch.cat((X, Y), dim=0)  # concatenate X and Y along the first dimension
torch.cat((X, Y), dim=1)  # concatenate X and Y along the second dimension

print('X, Y =', (X, Y))

print('X + Y =', X + Y) # type: ignore # elementwise addition
print('X + 5 =', X + 5)
bias = torch.ones_like(X) * 2
X += bias  # add bias to X
print('X =', X)

Y = torch.tensor([[1, 2], [3, 4]])
print('id(Y):', id(Y))
X = torch.tensor([[5, 6], [7, 8]])
print('id(X):', id(X))

before = id(Y)
Y = Y + X
print('id(Y) unchanged:', id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = Y + X
print('id(Z):', id(Z))

before = id(X)
X += Y
print('id(X) unchanged:', id(X) == before)

x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
A = x.numpy()
B = torch.from_numpy(A)
print('type(B), type(A) =', (type(B), type(A)))
a = torch.tensor([3.5])
print('a, a.item(), float(a), int(a) =', (a, a.item(), float(a), int(a)))