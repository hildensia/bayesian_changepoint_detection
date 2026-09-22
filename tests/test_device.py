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

pytestmark = pytest.mark.behavior


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


class TestDefaultDtype:
    """Without ``dtype``, to_tensor keeps float64 precision (off MPS)."""

    @pytest.mark.parametrize(
        "data, expected",
        [
            (np.array([1.5, 2.5]), torch.float64),
            (np.array([1.5], dtype=np.float32), torch.float32),
            (np.array([1, 2]), torch.float32),
            ([1.0, 2.0], torch.float64),
            ([[1, 2.5], [3, 4]], torch.float64),
            ([1, 2], torch.float32),
            (1e8 + 0.5, torch.float64),
            (3, torch.float32),
            (np.float64(2.5), torch.float64),
            (torch.tensor([1.0], dtype=torch.float64), torch.float64),
            (torch.tensor([1.0], dtype=torch.float16), torch.float32),
            (torch.tensor([1, 2]), torch.float32),
            ([], torch.float32),
        ],
    )
    def test_rules(self, data, expected):
        assert to_tensor(data, device="cpu").dtype == expected

    def test_result_does_not_alias_the_input(self):
        array = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(array, device="cpu")
        tensor[0] = 99.0
        assert array[0] == 1.0

    def test_complex_input_stays_complex(self):
        # So that the detectors' "data must be real" check can reject it.
        assert to_tensor(np.array([1 + 2j]), device="cpu").is_complex()
        assert to_tensor([1.0, 2 + 1j], device="cpu").is_complex()
        assert to_tensor(3j, device="cpu").is_complex()

    def test_explicit_dtype_wins(self):
        tensor = to_tensor(np.array([1.5]), device="cpu", dtype=torch.float32)
        assert tensor.dtype == torch.float32

    def test_float64_numpy_keeps_its_low_digits(self):
        # float32 cannot represent 1e8 + 0.25 (spacing 8 near 1e8).
        value = 1e8 + 0.25
        assert ensure_tensor(np.array([value]), device="cpu").item() == value
        assert ensure_tensor([value], device="cpu").item() == value

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_gets_float32(self):
        assert to_tensor(np.array([1.5]), device="mps").dtype == torch.float32
        assert to_tensor([1.5], device="mps").dtype == torch.float32
