"""
D2L Custom - Test Suite
"""

import torch
from d2l_custom.models import LinearRegression


def test_linear_regression():
    """Test LinearRegression model."""
    model = LinearRegression(in_features=10)
    X = torch.randn(32, 10)
    y_hat = model(X)
    assert y_hat.shape == (32, 1)


def test_cuda_availability():
    """Check CUDA availability."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        x = torch.randn(100, 100, device=device)
        assert x.is_cuda
