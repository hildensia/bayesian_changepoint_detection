"""
Device management utilities for PyTorch tensors.

This module provides utilities for automatic device detection and tensor management
across CPU and GPU platforms.
"""

from typing import Optional, Union

import torch


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the appropriate PyTorch device.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Desired device. If None, automatically selects the best available device.

    Returns
    -------
    torch.device
        The selected device.

    Examples
    --------
    >>> device = get_device()  # Auto-select best device
    >>> device = get_device('cpu')  # Force CPU
    >>> device = get_device('cuda:0')  # Force specific GPU
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device)


def to_tensor(
    data,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert data to PyTorch tensor on specified device.

    Without an explicit ``dtype``, floating-point input keeps its precision:
    float64 NumPy arrays, Python floats and float64 tensors stay float64,
    float32 stays float32. Integer and boolean input becomes float32, and
    complex input stays complex (so detectors can reject it). On MPS, which
    has no float64, real input becomes float32. The result is always a new
    tensor, never a view of the caller's array. (Versions up to
    1.1.0 made everything float32, so a float64 NumPy series far from zero
    lost its low digits before any detector saw it.)

    Parameters
    ----------
    data : array-like
        Input data to convert to tensor.
    device : str, torch.device, or None, optional
        Target device for the tensor.
    dtype : torch.dtype, optional
        Desired data type for the tensor; overrides the rule above.

    Returns
    -------
    torch.Tensor
        Converted tensor on the specified device.

    Examples
    --------
    >>> import numpy as np
    >>> to_tensor(np.array([1, 2, 3]), device="cpu").dtype
    torch.float32
    >>> to_tensor(np.array([1.5, 2.5]), device="cpu").dtype
    torch.float64
    >>> tensor = to_tensor(np.array([1.5]), device='cuda', dtype=torch.float32)
    """
    device = get_device(device)
    if dtype is None:
        dtype = _default_float_dtype(data, device)
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    # torch.tensor copies: the result never aliases the caller's array.
    return torch.tensor(data, dtype=dtype, device=device)


def _default_float_dtype(data, device: torch.device) -> torch.dtype:
    """float64 for float64 input off MPS, float32 otherwise; complex input
    stays complex so that the detectors' "data must be real" check sees it
    (casting it to a real dtype would silently drop the imaginary part).

    Needs no NumPy import (the package depends only on torch): tensors and
    NumPy arrays or scalars expose ``dtype``; Python floats are float64.
    """
    source = getattr(data, "dtype", None)
    name = str(source).replace("torch.", "") if source is not None else ""
    if name.startswith("complex"):
        return torch.complex64 if device.type == "mps" else torch.complex128
    if device.type == "mps":
        return torch.float32
    if source is not None:
        return torch.float64 if name == "float64" else torch.float32
    return torch.float64 if _contains_float(data) else torch.float32


def _contains_float(data) -> bool:
    if isinstance(data, float):
        return True
    if isinstance(data, (list, tuple)):
        return any(_contains_float(value) for value in data)
    return False


def ensure_tensor(
    data, device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Ensure data is a PyTorch tensor, converting if necessary.

    Parameters
    ----------
    data : array-like or torch.Tensor
        Input data.
    device : str, torch.device, or None, optional
        Target device for the tensor.

    Returns
    -------
    torch.Tensor
        Tensor on the specified device.
    """
    if not isinstance(data, torch.Tensor):
        return to_tensor(data, device=device)

    target_device = get_device(device)
    if data.device != target_device:
        return data.to(target_device)

    return data


def get_device_info() -> dict:
    """
    Get information about available devices.

    Returns
    -------
    dict
        Dictionary containing device information.

    Examples
    --------
    >>> info = get_device_info()
    >>> print(f"CUDA available: {info['cuda_available']}")
    >>> print(f"Device count: {info['device_count']}")
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "current_device": None,
        "mps_available": False,
        "devices": [],
    }

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["devices"] = [f"cuda:{i}" for i in range(info["device_count"])]

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
        info["devices"].append("mps")

    info["devices"].append("cpu")

    return info
