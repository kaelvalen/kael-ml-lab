import torch
import time

A = torch.randn(100, 200)
B = torch.randn(100, 200)
C = torch.randn(100, 200)

tensor_3d = torch.stack([A, B, C], dim=0)
print(tensor_3d.shape)