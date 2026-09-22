"""
Hazard functions for Bayesian changepoint detection.

Hazard functions specify the prior probability of a changepoint occurring
at each time step, given the run length (time since last changepoint).
"""

import functools
import math
from typing import Optional, Union

import torch

from .device import ensure_tensor, get_device


def constant_hazard(
    lam: float,
    r: Union[torch.Tensor, int],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Constant hazard function for Bayesian online changepoint detection.

    This function returns a constant probability (1/lam) for a changepoint
    occurring at any time step, regardless of the current run length.

    Parameters
    ----------
    lam : float
        The expected run length (higher values = lower changepoint probability).
        Must be at least 1, because the hazard ``1 / lam`` is a probability.
        ``lam = 1`` puts a changepoint before every observation; ``inf``
        gives hazard 0 (no changepoints).
    r : torch.Tensor or int
        Run length tensor or shape specification. If int, creates a tensor
        of that size filled with the constant hazard value.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.

    Returns
    -------
    torch.Tensor
        Tensor of hazard probabilities with the same shape as r.

    Raises
    ------
    ValueError
        If ``lam`` is below 1 or NaN.

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
    # `not lam >= 1` rather than `lam < 1` so that NaN is rejected too.
    if not lam >= 1:
        raise ValueError(
            f"lam must be at least 1 (the hazard 1/lam is a probability), got {lam}"
        )

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


@functools.lru_cache(maxsize=32)
def _negative_binomial_hazard_table(k: int, p: float, size: int) -> torch.Tensor:
    """Hazard for run lengths ``0 .. size-1`` (float64, CPU), cached."""
    if p == 1.0:
        tail = 1
    else:
        # Beyond this many extra lengths the pmf tail is below 1e-18 of the
        # mass already counted (its terms shrink by (1 - p) per step).
        tail = int(math.ceil(math.log(1e-18) / math.log1p(-p))) + k
    lengths = torch.arange(1, size + tail + 1, dtype=torch.float64)
    kk = torch.tensor(float(k), dtype=torch.float64)
    log_pmf = torch.full_like(lengths, float("-inf"))
    valid = lengths >= k
    lv = lengths[valid]
    log_pmf[valid] = (
        torch.lgamma(lv)
        - torch.lgamma(kk)
        - torch.lgamma(lv - kk + 1)
        + k * math.log(p)
        + torch.xlogy(lv - kk, torch.tensor(1.0 - p, dtype=torch.float64))
    )
    # log P(length >= m) for every m, as a reversed cumulative log-sum.
    log_survival = torch.flip(torch.logcumsumexp(torch.flip(log_pmf, [0]), 0), [0])
    # H(r) = P(length = r + 1 | length > r) = pmf(r + 1) / P(length >= r + 1)
    log_hazard = log_pmf[:size] - log_survival[:size]
    hazard = torch.exp(log_hazard)
    # Beyond the support's end (p = 1 and r + 1 > k) the ratio is 0/0; a
    # segment that long cannot exist, so ending it has probability 1.
    return torch.nan_to_num(hazard, nan=1.0)


def negative_binomial_hazard(
    k: int,
    p: float,
    r: Union[torch.Tensor, int],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Hazard of a negative binomial segment-length distribution.

    The online counterpart of ``priors.negative_binomial_prior(t, k, p)``:
    segment lengths follow ``P(length = t) = C(t - 1, k - 1) p^k
    (1 - p)^(t - k)`` for ``t >= k`` (mean ``k / p``), and the hazard at run
    length ``r`` is the probability that a segment which has lasted ``r``
    observations ends at the next one,
    ``H(r) = P(length = r + 1) / P(length >= r + 1)``
    (Adams & MacKay 2007, section 2.1). With ``k = 1`` the lengths are
    geometric and the hazard is the constant ``p``, i.e.
    ``constant_hazard(1 / p, r)``. With ``k > 1`` short segments are
    unlikely: ``H(r) = 0`` for ``r + 1 < k``, and the hazard rises towards
    ``p`` as the run grows.

    Parameters
    ----------
    k : int
        Number of successes (shape); must be at least 1.
    p : float
        Success probability, in ``(0, 1]``.
    r : torch.Tensor or int
        Run lengths (non-negative integers), or an int ``n`` for run
        lengths ``0 .. n - 1``.
    device : str, torch.device, or None, optional
        Device for the output; defaults to the device of ``r``.

    Returns
    -------
    torch.Tensor
        Hazard probabilities, float32, same shape as ``r``.

    Raises
    ------
    ValueError
        If ``k < 1``, ``p`` is outside ``(0, 1]``, or a run length is
        negative.

    Examples
    --------
    >>> from functools import partial
    >>> hazard = partial(negative_binomial_hazard, 3, 0.02)  # mean length 150
    >>> R, _ = online_changepoint_detection(data, hazard, StudentT())
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if isinstance(r, int):
        target = get_device(device)
        r = torch.arange(r)
    else:
        target = get_device(device) if device is not None else None
        if not isinstance(r, torch.Tensor):
            r = ensure_tensor(r, device=get_device(device))
        target = target or r.device
    run_lengths = r.detach().to("cpu", torch.float64)
    if run_lengths.numel() == 0:
        return torch.zeros(r.shape, dtype=torch.float32, device=target)
    if bool((run_lengths < 0).any()):
        raise ValueError("run lengths must be non-negative")
    index = run_lengths.round().long()
    needed = int(index.max()) + 1
    size = 1 << max(needed - 1, 0).bit_length()  # next power of two: reuse the cache
    table = _negative_binomial_hazard_table(k, float(p), size)
    return table[index].to(device=target, dtype=torch.float32)
