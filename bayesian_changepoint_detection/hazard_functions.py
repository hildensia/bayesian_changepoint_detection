"""
Hazard functions for Bayesian changepoint detection.

Hazard functions specify the prior probability of a changepoint occurring
at each time step, given the run length (time since last changepoint).
"""

import torch
from typing import Union, Optional
from .device import ensure_tensor, get_device


def constant_hazard(
    lam: float, 
    r: Union[torch.Tensor, int], 
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Constant hazard function for Bayesian online changepoint detection.
    
    This function returns a constant probability (1/lam) for a changepoint
    occurring at any time step, regardless of the current run length.
    
    Parameters
    ----------
    lam : float
        The expected run length (higher values = lower changepoint probability).
        Must be positive.
    r : torch.Tensor or int
        Run length tensor or shape specification. If int, creates a tensor
        of that size filled with the constant hazard value.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.
        
    Returns
    -------
    torch.Tensor
        Tensor of hazard probabilities with the same shape as r.
        
    Examples
    --------
    >>> import torch
    >>> # Create hazard for run lengths 0 to 9
    >>> hazard = constant_hazard(10.0, 10)
    >>> print(hazard)  # All values will be 0.1
    
    >>> # Use with existing run length tensor
    >>> r = torch.arange(5)
    >>> hazard = constant_hazard(20.0, r)
    >>> print(hazard)  # All values will be 0.05
    
    Notes
    -----
    The constant hazard function assumes that the probability of a changepoint
    is independent of how long the current segment has been running. This is
    a common choice for modeling changepoints in stationary processes.
    """
    if lam <= 0:
        raise ValueError("Lambda must be positive")

    if isinstance(r, int):
        return torch.full(
            (r,), 1.0 / lam, device=get_device(device), dtype=torch.float32
        )

    # Follow the run-length tensor's device unless one is requested explicitly,
    # so the hazard never drags tensors onto a different (auto-selected) device.
    if not isinstance(r, torch.Tensor):
        r = ensure_tensor(r, device=get_device(device))
    elif device is not None:
        r = r.to(get_device(device))
    return torch.full_like(r, 1.0 / lam, dtype=torch.float32)
