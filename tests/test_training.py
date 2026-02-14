"""
D2L Custom - Training Tests
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from d2l_custom.models import LinearRegression
from d2l_custom.training import train_epoch, evaluate, accuracy_count, train
from d2l_custom.data import synthetic_data, load_data


class TestTrainingFunctions:
    """Test training utility functions."""
    
    @pytest.fixture
    def simple_model(self):
        return LinearRegression(in_features=10, out_features=1)
    
    @pytest.fixture
    def sample_data(self):
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        return loader, X, y
    
    def test_train_epoch(self, simple_model, sample_data):
        loader, X, y = sample_data
        device = torch.device("cpu")
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        train_loss, train_acc = train_epoch(
            simple_model, loader, loss_fn, optimizer, device
        )
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 1
    
    def test_evaluate(self, simple_model, sample_data):
        loader, X, y = sample_data
        device = torch.device("cpu")
        loss_fn = nn.MSELoss()
        
        test_loss, test_acc = evaluate(simple_model, loader, loss_fn, device)
        
        assert isinstance(test_loss, float)
        assert isinstance(test_acc, float)
        assert test_loss >= 0
        assert 0 <= test_acc <= 1
    
    def test_accuracy_count_regression(self):
        # For regression, accuracy_count should work with continuous values
        y_hat = torch.randn(10, 1)
        y = torch.randn(10, 1)
        count = accuracy_count(y_hat, y)
        assert isinstance(count, float)
    
    def test_accuracy_count_classification(self):
        # For classification
        y_hat = torch.randn(10, 5)  # 5 classes
        y = torch.randint(0, 5, (10,))
        count = accuracy_count(y_hat, y)
        assert isinstance(count, float)
        assert 0 <= count <= 10
    
    def test_train_full_loop(self):
        # Generate synthetic data
        true_w = torch.tensor([2.0, -3.4, 5.6])
        true_b = 1.2
        X, y = synthetic_data(true_w, true_b, 1000)
        
        # Create data loaders
        train_loader, test_loader = load_data(X, y, batch_size=32, train_ratio=0.8)
        
        # Initialize model and optimizer
        model = LinearRegression(in_features=3, out_features=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Train for a few epochs
        history = train(
            model, train_loader, test_loader, loss_fn, optimizer, 
            num_epochs=3, verbose=False
        )
        
        # Check history structure
        assert "train_loss" in history
        assert "train_acc" in history
        assert "test_loss" in history
        assert "test_acc" in history
        assert len(history["train_loss"]) == 3
        assert len(history["train_acc"]) == 3
        assert len(history["test_loss"]) == 3
        assert len(history["test_acc"]) == 3
    
    def test_train_with_scheduler(self):
        X, y = synthetic_data(torch.tensor([1.0]), 0.0, 100)
        train_loader, test_loader = load_data(X, y, batch_size=16)
        
        model = LinearRegression(in_features=1, out_features=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        loss_fn = nn.MSELoss()
        
        history = train(
            model, train_loader, test_loader, loss_fn, optimizer,
            num_epochs=2, scheduler=scheduler, verbose=False
        )
        
        assert len(history["train_loss"]) == 2
    
    def test_train_device_selection(self):
        X, y = synthetic_data(torch.tensor([1.0]), 0.0, 50)
        train_loader, test_loader = load_data(X, y, batch_size=8)
        
        model = LinearRegression(in_features=1, out_features=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Test with explicit device
        device = torch.device("cpu")
        history = train(
            model, train_loader, test_loader, loss_fn, optimizer,
            num_epochs=1, device=device, verbose=False
        )
        
        assert len(history["train_loss"]) == 1
        assert next(model.parameters()).device == device


class TestDataFunctions:
    """Test data utility functions."""
    
    def test_synthetic_data(self):
        w = torch.tensor([2.0, -1.0])
        b = 0.5
        X, y = synthetic_data(w, b, 100)
        
        assert X.shape == (100, 2)
        assert y.shape == (100, 1)
        assert torch.allclose(X @ w + b, y, atol=0.1)  # Allow noise tolerance
    
    def test_load_data(self):
        X = torch.randn(100, 5)
        y = torch.randn(100, 1)
        
        train_loader, test_loader = load_data(X, y, batch_size=16, train_ratio=0.8)
        
        # Check train loader
        train_data = list(train_loader)
        assert len(train_data) > 0
        
        # Check test loader
        test_data = list(test_loader)
        assert len(test_data) > 0
        
        # Check total samples
        train_samples = sum(batch[0].shape[0] for batch in train_data)
        test_samples = sum(batch[0].shape[0] for batch in test_data)
        assert train_samples + test_samples == 100
        assert abs(train_samples - 80) <= 16  # Allow for batch size rounding
