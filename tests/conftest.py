"""
pytest configuration and fixtures for D2L Custom tests
"""

import pytest
import torch


@pytest.fixture(scope="session")
def torch_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def cpu_device():
    """Provide CPU device for tests."""
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    """Provide GPU device for tests if available."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def set_default_dtype():
    """Set default float dtype for tests."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(original_dtype)
