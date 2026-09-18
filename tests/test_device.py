"""
Tests for device management utilities.
"""

import numpy as np
import pytest
import torch

from bayesian_changepoint_detection.device import (
    ensure_tensor,
    get_device,
    get_device_info,
    to_tensor,
)

pytestmark = pytest.mark.behaviour


class TestDeviceManagement:
    """Test device management functionality."""

    def test_get_device_auto(self):
        """Test automatic device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_get_device_explicit(self):
        """Test explicit device specification."""
        cpu_device = get_device("cpu")
        assert cpu_device == torch.device("cpu")

        # Test with torch.device object
        device_obj = torch.device("cpu")
        result = get_device(device_obj)
        assert result == device_obj

    def test_to_tensor_numpy(self):
        """Test conversion from numpy array to tensor."""
        data = np.array([1, 2, 3])
        tensor = to_tensor(data, device="cpu")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device == torch.device("cpu")
        assert torch.allclose(tensor, torch.tensor([1, 2, 3], dtype=torch.float32))

    def test_to_tensor_list(self):
        """Test conversion from list to tensor."""
        data = [1, 2, 3]
        tensor = to_tensor(data, device="cpu", dtype=torch.int64)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.int64
        assert torch.equal(tensor, torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_to_tensor_existing_tensor(self):
        """Test handling of existing tensor."""
        original = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor = to_tensor(original, device="cpu", dtype=torch.float32)

        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor([1, 2, 3], dtype=torch.float32))

    def test_ensure_tensor(self):
        """Test ensure_tensor functionality."""
        # Test with numpy array
        data = np.array([1, 2, 3])
        tensor = ensure_tensor(data, device="cpu")
        assert isinstance(tensor, torch.Tensor)

        # Test with existing tensor on same device
        existing = torch.tensor([1, 2, 3])
        result = ensure_tensor(existing, device="cpu")
        assert result.device == torch.device("cpu")

        # Test with existing tensor on different device
        cpu_tensor = torch.tensor([1, 2, 3], device="cpu")
        result = ensure_tensor(cpu_tensor, device="cpu")
        assert result.device == torch.device("cpu")

    def test_get_device_info(self):
        """Test device information retrieval."""
        info = get_device_info()

        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "device_count" in info
        assert "devices" in info
        assert "mps_available" in info

        # CPU should always be available
        assert "cpu" in info["devices"]

        # Check consistency
        if info["cuda_available"]:
            assert info["device_count"] > 0
            assert any("cuda" in device for device in info["devices"])
        else:
            assert info["device_count"] == 0


@pytest.mark.gpu
class TestGPUDevice:
    """Test GPU-specific functionality (requires GPU)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device functionality."""
        device = get_device("cuda")
        assert device.type == "cuda"

        # Test tensor creation on CUDA
        data = [1, 2, 3]
        tensor = to_tensor(data, device="cuda")
        assert tensor.device.type == "cuda"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_device(self):
        """Test MPS device functionality (Apple Silicon)."""
        device = get_device("mps")
        assert device.type == "mps"

        # Test tensor creation on MPS
        data = [1, 2, 3]
        tensor = to_tensor(data, device="mps")
        assert tensor.device.type == "mps"
