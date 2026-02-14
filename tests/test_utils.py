"""
D2L Custom - Utility Tests
"""

import time
import pytest
import torch
import numpy as np
from torch import nn
from d2l_custom.utils import Timer, Accumulator, try_gpu, try_all_gpus, count_parameters, init_weights


class TestTimer:
    """Test Timer utility."""
    
    def test_timer_basic_functionality(self):
        timer = Timer()
        time.sleep(0.01)  # 10ms
        elapsed = timer.stop()
        assert elapsed >= 0.01
        assert len(timer.times) == 1
    
    def test_timer_multiple_stops(self):
        timer = Timer()
        for _ in range(3):
            time.sleep(0.001)
            timer.stop()
        assert len(timer.times) == 3
        assert timer.avg() > 0
        assert timer.sum() > 0
        assert len(timer.cumsum()) == 3
    
    def test_timer_restart(self):
        timer = Timer()
        timer.stop()
        old_len = len(timer.times)
        timer.start()
        timer.stop()
        assert len(timer.times) == old_len + 1


class TestAccumulator:
    """Test Accumulator utility."""
    
    def test_accumulator_basic(self):
        acc = Accumulator(3)
        acc.add(1, 2, 3)
        acc.add(4, 5, 6)
        assert acc[0] == 5
        assert acc[1] == 7
        assert acc[2] == 9
    
    def test_accumulator_reset(self):
        acc = Accumulator(2)
        acc.add(10, 20)
        acc.reset()
        assert acc[0] == 0
        assert acc[1] == 0
    
    def test_accumulator_float_conversion(self):
        acc = Accumulator(2)
        acc.add(1, 2.5)
        assert acc[0] == 1.0
        assert acc[1] == 2.5


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_try_gpu(self):
        device = try_gpu(0)
        assert isinstance(device, torch.device)
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"
    
    def test_try_gpu_invalid_index(self):
        device = try_gpu(999)  # Invalid GPU index
        assert device.type == "cpu"
    
    def test_try_all_gpus(self):
        devices = try_all_gpus()
        assert isinstance(devices, list)
        assert len(devices) > 0
        if torch.cuda.is_available():
            assert all(d.type == "cuda" for d in devices)
        else:
            assert devices == [torch.device("cpu")]


class TestModelUtils:
    """Test model utility functions."""
    
    def test_count_parameters(self):
        # Simple linear model
        model = nn.Linear(10, 5)
        param_count = count_parameters(model)
        # Linear layer: 10*5 + 5 = 55 parameters
        assert param_count == 55
    
    def test_count_parameters_with_frozen(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        
        param_count = count_parameters(model)
        # Only second layer: 20*5 + 5 = 105 parameters
        assert param_count == 105
    
    def test_init_weights_xavier(self):
        model = nn.Linear(10, 5)
        init_weights(model, method="xavier")
        # Check that weights are not all zeros
        assert not torch.allclose(model.weight, torch.zeros_like(model.weight))
    
    def test_init_weights_kaiming(self):
        model = nn.Conv2d(3, 16, 3)
        init_weights(model, method="kaiming")
        assert not torch.allclose(model.weight, torch.zeros_like(model.weight))
    
    def test_init_weights_normal(self):
        model = nn.Linear(5, 3)
        init_weights(model, method="normal")
        assert not torch.allclose(model.weight, torch.zeros_like(model.weight))
    
    def test_init_weights_bias_zero(self):
        model = nn.Linear(5, 3)
        init_weights(model, method="xavier")
        assert torch.allclose(model.bias, torch.zeros_like(model.bias))
    
    def test_init_weights_no_effect_on_non_target_layers(self):
        # Create a model with different layer types
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        # Store original BatchNorm parameters
        orig_weight = model[1].weight.clone()
        orig_bias = model[1].bias.clone()
        
        init_weights(model, method="xavier")
        
        # BatchNorm parameters should be unchanged
        assert torch.allclose(model[1].weight, orig_weight)
        assert torch.allclose(model[1].bias, orig_bias)
