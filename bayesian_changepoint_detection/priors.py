"""
Prior probability distributions for Bayesian changepoint detection.

This module provides various prior distributions for modeling the probability
of changepoints in time series data.
"""

import torch
import torch.distributions as dist
from typing import Union, Optional
from .device import ensure_tensor, get_device


def const_prior(
    t: Union[int, torch.Tensor], 
    p: float = 0.25,
    device: Optional[Union[str, torch.device]] = None
) -> Union[float, torch.Tensor]:
    """
    Constant prior on segment length.

    Returns ``log(p)`` for every length. This is not a probability mass
    function on its own; it is the conventional choice of the original
    library, used as ``partial(const_prior, p=1 / (n + 1))`` for a series of
    ``n`` observations. ``offline_changepoint_detection`` needs the prior mass
    on lengths ``1 .. n - 1`` to stay below 1, i.e. ``p * (n - 1) < 1``, and
    raises otherwise.
    
    Parameters
    ----------
    t : int or torch.Tensor
        Time index or tensor of time indices.
    p : float, optional
        Constant probability value (default: 0.25).
        Must be between 0 and 1.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.
        
    Returns
    -------
    float or torch.Tensor
        Log probability value(s).
        
    Examples
    --------
    >>> # Single time point
    >>> log_prob = const_prior(5, p=0.1)
    >>> print(log_prob)  # log(0.1)
    
    >>> # Multiple time points
    >>> t = torch.arange(10)
    >>> log_probs = const_prior(t, p=0.2)
    >>> print(log_probs.shape)  # torch.Size([10])
    
    Notes
    -----
    Under this prior every segmentation with the same number of changepoints
    has the same prior probability, regardless of where the changepoints
    fall.
    """
    if not 0 < p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    
    log_p = torch.log(torch.tensor(p, dtype=torch.float32))
    
    if isinstance(t, int):
        return log_p.item()
    else:
        device = get_device(device)
        t_tensor = ensure_tensor(t, device=device)
        return log_p.expand_as(t_tensor)


def geometric_prior(
    t: Union[int, torch.Tensor],
    p: float = 0.25,
    device: Optional[Union[str, torch.device]] = None
) -> Union[float, torch.Tensor]:
    """
    Geometric prior on segment length.

    ``P(length = t) = (1 - p)^(t - 1) p`` for ``t >= 1``: the number of
    trials up to and including the first success when each observation ends
    the segment with probability ``p``. The mean segment length is ``1 / p``.
    Lengths ``t <= 0`` are impossible and get log probability ``-inf``.

    Parameters
    ----------
    t : int or torch.Tensor
        Segment length(s).
    p : float, optional
        Probability that a segment ends at each observation (default: 0.25).
        Must be in ``(0, 1]``.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.

    Returns
    -------
    float or torch.Tensor
        Log probability value(s).

    Examples
    --------
    >>> import math
    >>> math.isclose(geometric_prior(1, p=0.1), math.log(0.1))
    True
    >>> math.isclose(geometric_prior(3, p=0.1), math.log(0.9 * 0.9 * 0.1))
    True
    >>> geometric_prior(torch.arange(1, 11), p=0.2).shape
    torch.Size([10])

    Notes
    -----
    ``torch.distributions.Geometric`` counts *failures before* the first
    success (support ``0, 1, 2, ...``), so it is evaluated at ``t - 1``. The
    pre-PyTorch versions of this library used the same ``(1 - p)^(t - 1) p``
    form.
    """
    if not 0 < p <= 1:
        raise ValueError("Probability p must be between 0 and 1")

    device = get_device(device)
    geom_dist = dist.Geometric(probs=torch.tensor(p, device=device, dtype=torch.float32))

    if isinstance(t, int):
        if t < 1:
            return float('-inf')
        return geom_dist.log_prob(torch.tensor(t - 1, device=device, dtype=torch.float32)).item()

    t_tensor = ensure_tensor(t, device=device).to(torch.float32)
    log_probs = torch.full_like(t_tensor, float('-inf'))
    valid = t_tensor >= 1
    if torch.any(valid):
        log_probs[valid] = geom_dist.log_prob(t_tensor[valid] - 1)
    return log_probs


def negative_binomial_prior(
    t: Union[int, torch.Tensor],
    k: int = 1,
    p: float = 0.25,
    device: Optional[Union[str, torch.device]] = None
) -> Union[float, torch.Tensor]:
    """
    Negative binomial prior on segment length.

    ``P(length = t) = C(t - 1, k - 1) p^k (1 - p)^(t - k)`` for ``t >= k``:
    the number of trials needed to obtain ``k`` successes when each trial
    succeeds with probability ``p``. The mean segment length is ``k / p``.
    Lengths ``t < k`` are impossible and get log probability ``-inf``. With
    ``k = 1`` this is exactly ``geometric_prior``.

    Parameters
    ----------
    t : int or torch.Tensor
        Segment length(s).
    k : int, optional
        Number of successes required (default: 1). Must be positive.
    p : float, optional
        Success probability of each trial (default: 0.25). Must be in
        ``(0, 1]``.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.

    Returns
    -------
    float or torch.Tensor
        Log probability value(s).

    Examples
    --------
    >>> import math
    >>> math.isclose(negative_binomial_prior(3, k=2, p=0.5), math.log(2 * 0.25 * 0.5))
    True
    >>> negative_binomial_prior(1, k=2, p=0.5)
    -inf
    >>> negative_binomial_prior(torch.arange(1, 11), k=3, p=0.2).shape
    torch.Size([10])

    Notes
    -----
    Computed in closed form with ``lgamma`` rather than through
    ``torch.distributions.NegativeBinomial``, whose ``probs`` is the
    probability of the *counted* outcome (the complement of ``p`` here);
    versions 1.0.x used that class with ``probs=p`` and therefore had ``p``
    and ``1 - p`` swapped. Equivalent to ``scipy.stats.nbinom(k, p).pmf(t - k)``.
    """
    if not 0 < p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    if k <= 0:
        raise ValueError("Number of successes k must be positive")

    device = get_device(device)
    scalar = isinstance(t, int)
    # Evaluate in float64 on the CPU (lgamma differences lose digits in
    # float32, and MPS has no float64), then move to the requested device.
    t_tensor = torch.tensor([t], dtype=torch.float64) if scalar \
        else ensure_tensor(t, device="cpu").to(torch.float64)
    kk = torch.tensor(float(k), dtype=torch.float64)
    pp = torch.tensor(float(p), dtype=torch.float64)

    log_probs = torch.full_like(t_tensor, float('-inf'))
    valid = t_tensor >= k
    if torch.any(valid):
        tv = t_tensor[valid]
        # log C(t-1, k-1) + k log p + (t-k) log(1-p); xlogy keeps p = 1 finite.
        log_probs[valid] = (
            torch.lgamma(tv) - torch.lgamma(kk) - torch.lgamma(tv - kk + 1)
            + kk * torch.log(pp)
            + torch.xlogy(tv - kk, 1 - pp)
        )
    log_probs = log_probs.to(device=device, dtype=torch.float32)
    return log_probs.item() if scalar else log_probs
