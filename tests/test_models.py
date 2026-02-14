"""
D2L Custom - Model Tests
"""

import pytest
import torch
from d2l_custom.models import LinearRegression, SoftmaxRegression, MLP, ResidualBlock


class TestLinearRegression:
    """Test LinearRegression model."""
    
    def test_forward_shape(self):
        model = LinearRegression(in_features=10, out_features=1)
        X = torch.randn(32, 10)
        y_hat = model(X)
        assert y_hat.shape == (32, 1)
    
    def test_loss_computation(self):
        model = LinearRegression(in_features=5, out_features=1)
        X = torch.randn(16, 5)
        y = torch.randn(16, 1)
        y_hat = model(X)
        loss = model.loss(y_hat, y)
        assert loss.item() >= 0
        assert loss.dim() == 0


class TestSoftmaxRegression:
    """Test SoftmaxRegression model."""
    
    def test_forward_shape(self):
        model = SoftmaxRegression(num_inputs=784, num_outputs=10)
        X = torch.randn(32, 784)
        y_hat = model(X)
        assert y_hat.shape == (32, 10)
    
    def test_loss_computation(self):
        model = SoftmaxRegression(num_inputs=20, num_outputs=5)
        X = torch.randn(16, 20)
        y = torch.randint(0, 5, (16,))
        y_hat = model(X)
        loss = model.loss(y_hat, y)
        assert loss.item() >= 0
        assert loss.dim() == 0


class TestMLP:
    """Test MLP model."""
    
    def test_forward_shape(self):
        model = MLP(in_features=10, hidden_sizes=[20, 15], out_features=3)
        X = torch.randn(32, 10)
        y_hat = model(X)
        assert y_hat.shape == (32, 3)
    
    def test_single_hidden_layer(self):
        model = MLP(in_features=5, hidden_sizes=[10], out_features=1)
        X = torch.randn(8, 5)
        y_hat = model(X)
        assert y_hat.shape == (8, 1)
    
    def test_dropout_effect(self):
        model = MLP(in_features=10, hidden_sizes=[20], out_features=1, dropout=0.5)
        model.train()
        X = torch.randn(32, 10)
        y1 = model(X)
        y2 = model(X)
        # With dropout, outputs should be different
        assert not torch.allclose(y1, y2, atol=1e-6)


class TestResidualBlock:
    """Test ResidualBlock model."""
    
    def test_forward_shape_same_channels(self):
        block = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        X = torch.randn(16, 64, 32, 32)
        y_hat = block(X)
        assert y_hat.shape == (16, 64, 32, 32)
    
    def test_forward_shape_different_channels(self):
        block = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        X = torch.randn(16, 32, 32, 32)
        y_hat = block(X)
        assert y_hat.shape == (16, 64, 16, 16)
    
    def test_gradient_flow(self):
        block = ResidualBlock(in_channels=16, out_channels=16)
        X = torch.randn(8, 16, 8, 8, requires_grad=True)
        y = block(X)
        loss = y.sum()
        loss.backward()
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()


def test_cuda_availability():
    """Check CUDA availability and basic functionality."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        x = torch.randn(100, 100, device=device)
        assert x.is_cuda
        
        # Test model on CUDA
        model = LinearRegression(in_features=100, out_features=1).to(device)
        y = model(x)
        assert y.is_cuda
    else:
        pytest.skip("CUDA not available")
