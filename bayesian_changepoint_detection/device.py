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

    Parameters
    ----------
    data : array-like
        Input data to convert to tensor.
    device : str, torch.device, or None, optional
        Target device for the tensor.
    dtype : torch.dtype, optional
        Desired data type for the tensor.

    Returns
    -------
    torch.Tensor
        Converted tensor on the specified device.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3])
    >>> tensor = to_tensor(data)
    >>> tensor = to_tensor(data, device='cuda', dtype=torch.float32)
    """
    if dtype is None:
        dtype = torch.float32

    device = get_device(device)

    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    else:
        return torch.tensor(data, device=device, dtype=dtype)


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
