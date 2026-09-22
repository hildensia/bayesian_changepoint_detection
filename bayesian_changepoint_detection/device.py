"""
Device management utilities for PyTorch tensors.

This module provides device selection (CPU by default, accelerators on request)
and tensor coercion.
"""

from typing import Optional, Union

import torch


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the appropriate PyTorch device.

    The default is the CPU. Accelerators are opt-in: pass ``"auto"`` for the
    first available of CUDA, MPS and CPU, or name the device. (Up to 1.1.0
    ``None`` meant ``"auto"``; on the laptops where it was measured the CPU
    was 6-30x faster than MPS for the online detector, and the offline
    detector cannot run on MPS at all, so the automatic choice was a poor
    default.)

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Desired device. ``None`` selects the CPU; ``"auto"`` selects the
        best available device.

    Returns
    -------
    torch.device
        The selected device.

    Examples
    --------
    >>> get_device()
    device(type='cpu')
    >>> device = get_device("auto")  # CUDA, then MPS, then CPU
    >>> device = get_device("cuda:0")  # a specific GPU
    """
    if device is None:
        return torch.device("cpu")
    if isinstance(device, str) and device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
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
    has no float64, real input becomes float32. Non-tensor input is always
    copied, so the result never aliases the caller's array or list (a
    tensor already on the right device and dtype is returned as is). (Versions up to
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
    if name.startswith("complex") or (source is None and _contains(data, complex)):
        return torch.complex64 if device.type == "mps" else torch.complex128
    if device.type == "mps":
        return torch.float32
    if source is not None:
        return torch.float64 if name == "float64" else torch.float32
    return torch.float64 if _contains(data, float) else torch.float32


def _contains(data, kind: type) -> bool:
    """Whether a Python scalar or (nested) list/tuple holds a ``kind``."""
    if isinstance(data, kind):
        return True
    if isinstance(data, (list, tuple)):
        return any(_contains(value, kind) for value in data)
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
