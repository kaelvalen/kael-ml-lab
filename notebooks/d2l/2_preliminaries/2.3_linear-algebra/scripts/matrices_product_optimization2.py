import torch
import time

A = torch.randn(10, 16)
B = torch.randn(16, 5)
C = torch.randn(5, 16)

start = time.time()
result2 = A @ B
print(f"Time: {time.time() - start:.10f}s")

start = time.time()
result3 = A @ C.T
print(f"Time: {time.time() - start:.10f}s")

