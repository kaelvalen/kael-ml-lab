import torch
import time

A = torch.randn(1024, 65536)
B = torch.randn(65536, 32)
C = torch.randn(32, 16384)

# (AB)C
start = time.time()
result1 = (A @ B) @ C
print(f"(AB)C: {time.time() - start:.2f}s")

# A(BC)
start = time.time()
result2 = A @ (B @ C)
print(f"A(BC): {time.time() - start:.2f}s")

# Sonuç aynı mı?
print(torch.allclose(result1, result2, atol=1e-5))

# (AB)C: 0.05s
# A(BC): 5.17s
