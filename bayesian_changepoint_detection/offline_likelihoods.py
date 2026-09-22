"""
Offline likelihood functions for Bayesian changepoint detection.

This module provides likelihood functions for offline (batch) changepoint detection
using PyTorch for efficient computation and GPU acceleration.

All likelihoods expose two evaluation entry points:

- ``pdf(data, t, s)``: log marginal likelihood of the segment ``data[t:s]``
  (``s`` exclusive). Kept for backward compatibility.
- ``pdf_rows(data, t)``: log marginal likelihoods of ``data[t:s]`` for every
  ``s`` in ``t+1 .. n`` in a single vectorized pass. The dynamic programming
  driver uses this method; it is the reason offline detection runs in seconds
  instead of minutes (see GitHub issue #47).

Sufficient statistics (cumulative sums of ``x`` and ``x**2``, and cumulative
outer products where needed) are computed once per dataset by ``setup()`` and
reused for every segment query.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch

from .device import ensure_tensor, get_device

_LOG_PI = math.log(math.pi)
_LOG_2PI = math.log(2.0 * math.pi)
_V0_FLOOR = 1e-8  # floor on the data-derived prior variance


def _multigammaln(a: torch.Tensor, p: int) -> torch.Tensor:
    """Log of the multivariate gamma function, vectorized over ``a``."""
    j = torch.arange(p, device=a.device, dtype=a.dtype)
    return (p * (p - 1) / 4.0) * _LOG_PI + torch.lgamma(a.unsqueeze(-1) - j / 2.0).sum(
        -1
    )


class BaseLikelihood(ABC):
    """
    Abstract base class for offline likelihood functions.

    Subclasses must implement ``pdf`` and should override ``_compute_stats``
    and ``pdf_rows`` for vectorized evaluation. The default ``pdf_rows`` falls
    back to calling ``pdf`` once per segment, so existing subclasses keep
    working unchanged.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on (CPU or GPU).
    cache_enabled : bool, optional
        Retained for backward compatibility. Sufficient statistics are now
        precomputed once per dataset by ``setup()``; there is no per-call
        cache to enable or disable.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True,
    ):
        self.device = get_device(device)
        self.cache_enabled = cache_enabled
        self._stats_key = None
        self._prepared = None

    def setup(self, data: torch.Tensor) -> torch.Tensor:
        """
        Prepare per-dataset sufficient statistics.

        Idempotent and cheap when called repeatedly with the same tensor:
        statistics are recomputed only when the underlying storage, its
        in-place mutation counter, the shape, strides, dtype, or device of
        ``data`` change.

        Returns the prepared ``[T, D]`` tensor the statistics refer to.
        """
        # Key on the tensor the caller passed, *before* any device or dtype
        # conversion: converting allocates a fresh tensor, so keying on the
        # converted one would miss the cache on every call with float32 input.
        # ``data_ptr`` identifies storage, not contents; ``_version`` is
        # PyTorch's per-tensor in-place mutation counter, so ``x[0] = 1``
        # after a previous call invalidates the cached statistics.
        key = self._cache_key(data)
        if key is not None:
            # The prepared tensor lives on the model's device; if that was
            # changed since the statistics were computed (e.g. the driver's
            # MPS -> CPU float64 fallback), the cache no longer applies.
            key = key + (self.device,)
        if key is not None and key == self._stats_key:
            return self._prepared
        prepared = self._prepare_data(data)
        self._compute_stats(prepared)
        self._stats_key = key
        self._prepared = prepared
        return prepared

    @staticmethod
    def _cache_key(data) -> Optional[tuple]:
        if not isinstance(data, torch.Tensor):
            return None  # lists / arrays: no stable identity, always recompute
        try:
            version = data._version
        except RuntimeError:
            # Inference-mode tensors carry no version counter; treat them as
            # uncacheable rather than failing.
            return None
        return (
            data.data_ptr(),
            version,
            tuple(data.shape),
            tuple(data.stride()),
            data.dtype,
            data.device,
        )

    def _prepare_data(self, data: torch.Tensor) -> torch.Tensor:
        """Move data to the target device, promote precision, make it 2-D."""
        # float64 for numerically demanding cumulative statistics; MPS has no
        # float64 support, so stay in float32 there. Cast before moving so a
        # float64 CPU tensor is never transferred to MPS as float64.
        target = get_device(self.device)
        dtype = torch.float32 if target.type == "mps" else torch.float64
        if isinstance(data, torch.Tensor):
            data = data.to(dtype=dtype).to(device=target)
        else:
            data = ensure_tensor(data, device=target).to(dtype)
        if data.dim() == 1:
            data = data.unsqueeze(1)
        return data

    def _compute_stats(self, data: torch.Tensor) -> None:  # noqa: B027
        """Compute per-dataset sufficient statistics. Default: none.

        Not abstract on purpose: likelihoods without precomputed statistics
        (e.g. third-party ones that only implement ``pdf``) need no override.
        """

    @abstractmethod
    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute the log marginal likelihood of the segment ``data[t:s]``.

        Parameters
        ----------
        data : torch.Tensor
            The complete time series data.
        t : int
            Start index of the segment (inclusive).
        s : int
            End index of the segment (exclusive).

        Returns
        -------
        float
            Log marginal likelihood of ``data[t:s]``.
        """
        raise NotImplementedError("PDF method must be implemented in subclass.")

    def pdf_rows(self, data: torch.Tensor, t: int) -> torch.Tensor:
        """
        Log marginal likelihoods of ``data[t:s]`` for all ``s`` in ``t+1 .. n``.

        Returns a tensor of shape ``[n - t]`` whose element ``j`` equals
        ``pdf(data, t, t + 1 + j)``. Subclasses override this with a fully
        vectorized implementation; this fallback loops over ``pdf`` so that
        third-party likelihoods only implementing ``pdf`` keep working.
        """
        n = data.shape[0]
        values = [self.pdf(data, t, s) for s in range(t + 1, n + 1)]
        return torch.tensor(values, dtype=torch.float64, device=torch.device("cpu"))


class _CumsumLikelihood(BaseLikelihood):
    """Shared machinery: cumulative first and second moments per dimension."""

    def _compute_stats(self, data: torch.Tensor) -> None:
        n, d = data.shape
        zero = torch.zeros(1, d, dtype=data.dtype, device=data.device)
        # S1[k] = sum of data[:k], S2[k] = sum of data[:k]**2  (shape [n+1, d])
        self._S1 = torch.cat([zero, torch.cumsum(data, dim=0)])
        self._S2 = torch.cat([zero, torch.cumsum(data**2, dim=0)])

    def _segment_moments(self, t: int, s_hi: int):
        """Lengths, first and second moments of data[t:s] for s = t+1 .. s_hi."""
        sum_x = self._S1[t + 1 : s_hi + 1] - self._S1[t]
        sum_x2 = self._S2[t + 1 : s_hi + 1] - self._S2[t]
        lengths = torch.arange(1, s_hi - t + 1, dtype=sum_x.dtype, device=sum_x.device)
        return lengths, sum_x, sum_x2


class StudentT(_CumsumLikelihood):
    """
    Student's t (Normal-Gamma) marginal likelihood for offline detection.

    Computes the exact closed-form log marginal likelihood of a segment under
    a Normal likelihood with conjugate Normal-Gamma prior on (mean, precision)
    (Murphy, "Conjugate Bayesian analysis of the Gaussian distribution", 2007,
    eq. 95-97). Multivariate input is treated as independent dimensions whose
    log marginals are summed, matching the historical behavior of this class.

    Parameters
    ----------
    alpha0 : float, optional
        Prior shape parameter for precision (default: 1.0).
    beta0 : float, optional
        Prior rate parameter for precision (default: 1.0).
    kappa0 : float, optional
        Prior precision scaling for the mean (default: 1.0).
    mu0 : float, optional
        Prior mean (default: 0.0).
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Retained for backward compatibility (see ``BaseLikelihood``).

    Examples
    --------
    >>> import torch
    >>> likelihood = StudentT()
    >>> data = torch.randn(100)
    >>> log_prob = likelihood.pdf(data, 10, 50)  # Segment from 10 to 50

    Notes
    -----
    Earlier versions evaluated every segment point under the posterior
    predictive with the *final* posterior parameters, which is an
    approximation of the marginal likelihood. This implementation computes
    the exact marginal in closed form; it is also what makes full
    vectorization possible.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True,
        *,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        kappa0: float = 1.0,
        mu0: float = 0.0,
    ):
        # ``device`` and ``cache_enabled`` keep their historical positions so
        # ``StudentT("cpu")`` still works; the prior hyperparameters are new
        # and keyword-only.
        super().__init__(device, cache_enabled)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.kappa0 = kappa0
        self.mu0 = mu0

    def _log_marginal(
        self,
        lengths: torch.Tensor,
        sum_x: torch.Tensor,
        sum_x2: torch.Tensor,
    ) -> torch.Tensor:
        """Exact per-dimension log marginal likelihood, summed over dimensions.

        lengths: [m], sum_x/sum_x2: [m, d]  ->  returns [m].
        """
        n = lengths.unsqueeze(-1)  # [m, 1]
        mean = sum_x / n
        # sum of squared deviations; clamp guards tiny negative rounding error
        ss = torch.clamp(sum_x2 - sum_x**2 / n, min=0.0)

        kappa_n = self.kappa0 + n
        alpha_n = self.alpha0 + n / 2.0
        beta_n = (
            self.beta0
            + 0.5 * ss
            + self.kappa0 * n * (mean - self.mu0) ** 2 / (2.0 * kappa_n)
        )

        log_marginal = (
            torch.lgamma(alpha_n)
            - math.lgamma(self.alpha0)
            + self.alpha0 * math.log(self.beta0)
            - alpha_n * torch.log(beta_n)
            + 0.5 * (math.log(self.kappa0) - torch.log(kappa_n))
            - (n / 2.0) * _LOG_2PI
        )
        return log_marginal.sum(dim=-1)

    def pdf_rows(self, data: torch.Tensor, t: int) -> torch.Tensor:
        data = self.setup(data)
        n = data.shape[0]
        lengths, sum_x, sum_x2 = self._segment_moments(t, n)
        return self._log_marginal(lengths, sum_x, sum_x2)

    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute the log marginal likelihood of ``data[t:s]``.

        Parameters
        ----------
        data : torch.Tensor
            Complete time series data.
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).

        Returns
        -------
        float
            Log marginal likelihood of the segment.
        """
        if s <= t:
            return 0.0
        data = self.setup(data)
        lengths, sum_x, sum_x2 = self._segment_moments(t, s)
        return self._log_marginal(lengths[-1:], sum_x[-1:], sum_x2[-1:]).item()


class IndependentFeaturesLikelihood(_CumsumLikelihood):
    """
    Independent features likelihood for multivariate data.

    Assumes features are independent with unknown means and variances,
    following section 3.1 of Xuan & Murphy (2007). The math matches the
    original NumPy implementation of this package exactly.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Retained for backward compatibility (see ``BaseLikelihood``).

    Examples
    --------
    >>> import torch
    >>> likelihood = IndependentFeaturesLikelihood()
    >>> data = torch.randn(100, 5)  # 100 time points, 5 dimensions
    >>> log_prob = likelihood.pdf(data, 10, 50)
    """

    def _log_marginal(
        self,
        lengths: torch.Tensor,
        sum_x: torch.Tensor,
        sum_x2: torch.Tensor,
    ) -> torch.Tensor:
        m, d = sum_x.shape
        n = lengths  # [m]
        # Weakest proper prior: N0 = d, V0 = variance of the flattened segment
        # (population variance over all n*d entries), exactly as in the
        # original implementation.
        total = sum_x.sum(dim=1)
        total_sq = sum_x2.sum(dim=1)
        count = n * d
        v0 = total_sq / count - (total / count) ** 2  # [m]
        # A length-one univariate segment (or any constant segment) has zero
        # variance, and rounding can make it slightly negative; without a
        # floor, log(v0) is -inf/nan and poisons Q. Same floor as before.
        v0 = torch.clamp(v0, min=_V0_FLOOR)

        n0 = float(d)
        vn = v0.unsqueeze(-1) + sum_x2  # [m, d]

        return d * (
            -(n / 2.0) * _LOG_PI
            + (n0 / 2.0) * torch.log(v0)
            - math.lgamma(n0 / 2.0)
            + torch.lgamma((n0 + n) / 2.0)
        ) - ((n0 + n) / 2.0) * torch.log(vn).sum(dim=-1)

    def pdf_rows(self, data: torch.Tensor, t: int) -> torch.Tensor:
        data = self.setup(data)
        n = data.shape[0]
        lengths, sum_x, sum_x2 = self._segment_moments(t, n)
        return self._log_marginal(lengths, sum_x, sum_x2)

    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute the log marginal likelihood of ``data[t:s]`` assuming
        independent features.

        Parameters
        ----------
        data : torch.Tensor
            Complete time series data (shape: [T] or [T, D]).
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).

        Returns
        -------
        float
            Log marginal likelihood of the segment.
        """
        if s <= t:
            return 0.0
        data = self.setup(data)
        lengths, sum_x, sum_x2 = self._segment_moments(t, s)
        return self._log_marginal(lengths[-1:], sum_x[-1:], sum_x2[-1:]).item()


class FullCovarianceLikelihood(_CumsumLikelihood):
    """
    Full covariance likelihood for multivariate data.

    Models the full covariance structure following section 3.2 of
    Xuan & Murphy (2007). The math matches the original NumPy implementation
    of this package exactly.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Retained for backward compatibility (see ``BaseLikelihood``).

    Examples
    --------
    >>> import torch
    >>> likelihood = FullCovarianceLikelihood()
    >>> data = torch.randn(100, 3)  # 100 time points, 3 dimensions
    >>> log_prob = likelihood.pdf(data, 10, 50)
    """

    def _compute_stats(self, data: torch.Tensor) -> None:
        super()._compute_stats(data)
        n, d = data.shape
        outer = torch.einsum("ni,nj->nij", data, data)
        zero = torch.zeros(1, d, d, dtype=data.dtype, device=data.device)
        # C[k] = sum of outer products of data[:k]  (shape [n+1, d, d])
        self._C = torch.cat([zero, torch.cumsum(outer, dim=0)])

    def _log_marginal(
        self,
        lengths: torch.Tensor,
        sum_x: torch.Tensor,
        sum_x2: torch.Tensor,
        sum_outer: torch.Tensor,
    ) -> torch.Tensor:
        m, d = sum_x.shape
        n = lengths
        # Weakest proper prior: N0 = d, V0 = var(flattened segment) * I.
        total = sum_x.sum(dim=1)
        total_sq = sum_x2.sum(dim=1)
        count = n * d
        v0 = total_sq / count - (total / count) ** 2  # [m]
        # A length-one univariate segment (or any constant segment) has zero
        # variance, and rounding can make it slightly negative; without a
        # floor, log(v0) is -inf/nan and poisons Q. Same floor as before.
        v0 = torch.clamp(v0, min=_V0_FLOOR)

        n0 = float(d)
        eye = torch.eye(d, dtype=sum_x.dtype, device=sum_x.device)
        vn = v0.unsqueeze(-1).unsqueeze(-1) * eye + sum_outer  # [m, d, d]

        logdet_v0 = d * torch.log(v0)
        logdet_vn = torch.linalg.slogdet(vn)[1]

        mg0 = _multigammaln(torch.full_like(n, n0 / 2.0), d)
        mgn = _multigammaln((n0 + n) / 2.0, d)

        return (
            -(d * n / 2.0) * _LOG_PI
            + (n0 / 2.0) * logdet_v0
            - mg0
            + mgn
            - ((n0 + n) / 2.0) * logdet_vn
        )

    def pdf_rows(self, data: torch.Tensor, t: int) -> torch.Tensor:
        data = self.setup(data)
        n = data.shape[0]
        lengths, sum_x, sum_x2 = self._segment_moments(t, n)
        sum_outer = self._C[t + 1 : n + 1] - self._C[t]
        return self._log_marginal(lengths, sum_x, sum_x2, sum_outer)

    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute the log marginal likelihood of ``data[t:s]`` using the full
        covariance model.

        Parameters
        ----------
        data : torch.Tensor
            Complete time series data (shape: [T] or [T, D]).
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).

        Returns
        -------
        float
            Log marginal likelihood of the segment.
        """
        if s <= t:
            return 0.0
        data = self.setup(data)
        lengths, sum_x, sum_x2 = self._segment_moments(t, s)
        sum_outer = self._C[s : s + 1] - self._C[t]
        return self._log_marginal(
            lengths[-1:], sum_x[-1:], sum_x2[-1:], sum_outer
        ).item()


class MultivariateT(_CumsumLikelihood):
    """
    Multivariate Student's t (Normal-Wishart) likelihood for offline detection.

    Computes the exact log marginal likelihood of a segment under a
    multivariate Normal likelihood with conjugate Normal-Wishart prior on
    (mean vector, precision matrix).

    Parameters
    ----------
    dims : int, optional
        Number of dimensions. If None, taken from the data on every call;
        if given, data with a different dimension raises ``ValueError``.
    dof0 : float, optional
        Prior degrees of freedom (default: dims + 1).
    kappa0 : float, optional
        Prior precision for mean (default: 1.0).
    mu0 : torch.Tensor, optional
        Prior mean vector (default: zero vector).
    Psi0 : torch.Tensor, optional
        Scale matrix of the prior on the *covariance* side (the inverse of
        the Wishart scale ``W`` used by the online ``MultivariateT``):
        ``Psi0 = dof0 * C`` encodes a prior covariance ``C``. Default
        ``dof0 * I``, i.e. unit prior covariance, matching the online class.
        Versions up to 1.1.0 used ``I``, which is ``dof0`` times tighter.
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Retained for backward compatibility (see ``BaseLikelihood``).

    Examples
    --------
    >>> import torch
    >>> likelihood = MultivariateT(dims=3)
    >>> data = torch.randn(100, 3)
    >>> log_prob = likelihood.pdf(data, 10, 50)
    """

    def __init__(
        self,
        dims: Optional[int] = None,
        dof0: Optional[float] = None,
        kappa0: float = 1.0,
        mu0: Optional[torch.Tensor] = None,
        Psi0: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True,
    ):
        super().__init__(device, cache_enabled)
        self.dims = dims
        self.kappa0 = kappa0
        self.dof0 = dof0
        self.mu0 = mu0
        self.Psi0 = Psi0

    def _resolved_params(self, data: torch.Tensor):
        d = data.shape[1]
        # dims=None means "take it from the data", on every call: the model
        # can be reused on series of different dimension. An explicit dims
        # that disagrees with the data is an error, not silently overridden.
        if self.dims is not None and d != self.dims:
            raise ValueError(
                f"MultivariateT(dims={self.dims}) got observations of "
                f"dimension {d} (data shape {list(data.shape)})"
            )
        dof0 = self.dof0 if self.dof0 is not None else d + 1
        if self.mu0 is None:
            mu0 = torch.zeros(d, dtype=data.dtype, device=data.device)
        else:
            mu0 = ensure_tensor(self.mu0, device=data.device).to(data.dtype)
        if self.Psi0 is None:
            # Unit prior covariance: E[precision] = dof0 * Psi0^{-1} = I.
            psi0 = dof0 * torch.eye(d, dtype=data.dtype, device=data.device)
        else:
            psi0 = ensure_tensor(self.Psi0, device=data.device).to(data.dtype)
        return dof0, mu0, psi0

    def _compute_stats(self, data: torch.Tensor) -> None:
        super()._compute_stats(data)
        n, d = data.shape
        outer = torch.einsum("ni,nj->nij", data, data)
        zero = torch.zeros(1, d, d, dtype=data.dtype, device=data.device)
        self._C = torch.cat([zero, torch.cumsum(outer, dim=0)])

    def _log_marginal(
        self,
        data: torch.Tensor,
        lengths: torch.Tensor,
        sum_x: torch.Tensor,
        sum_outer: torch.Tensor,
    ) -> torch.Tensor:
        m, d = sum_x.shape
        dof0, mu0, psi0 = self._resolved_params(data)
        n = lengths

        mean = sum_x / n.unsqueeze(-1)  # [m, d]
        # Scatter matrix around the segment mean:
        # S = sum(x x^T) - n * mean mean^T
        scatter = sum_outer - n.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "mi,mj->mij", mean, mean
        )

        kappa_n = self.kappa0 + n
        dof_n = dof0 + n
        diff = mean - mu0
        psi_n = (
            psi0
            + scatter
            + (self.kappa0 * n / kappa_n).unsqueeze(-1).unsqueeze(-1)
            * torch.einsum("mi,mj->mij", diff, diff)
        )

        logdet_psi0 = torch.linalg.slogdet(psi0)[1]
        logdet_psi_n = torch.linalg.slogdet(psi_n)[1]

        return (
            _multigammaln(dof_n / 2.0, d)
            - _multigammaln(torch.full_like(n, dof0 / 2.0), d)
            + (dof0 / 2.0) * logdet_psi0
            - (dof_n / 2.0) * logdet_psi_n
            + (d / 2.0) * (math.log(self.kappa0) - torch.log(kappa_n))
            - (n * d / 2.0) * _LOG_PI
        )

    def pdf_rows(self, data: torch.Tensor, t: int) -> torch.Tensor:
        data = self.setup(data)
        n = data.shape[0]
        lengths, sum_x, _ = self._segment_moments(t, n)
        sum_outer = self._C[t + 1 : n + 1] - self._C[t]
        return self._log_marginal(data, lengths, sum_x, sum_outer)

    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute the log marginal likelihood of ``data[t:s]`` under the
        multivariate Student's t model.

        Parameters
        ----------
        data : torch.Tensor
            Complete time series data (shape: [T] or [T, D]).
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).

        Returns
        -------
        float
            Log marginal likelihood of the segment.
        """
        if s <= t:
            return 0.0
        data = self.setup(data)
        lengths, sum_x, _ = self._segment_moments(t, s)
        sum_outer = self._C[s : s + 1] - self._C[t]
        return self._log_marginal(data, lengths[-1:], sum_x[-1:], sum_outer).item()


def _check_counts(data: torch.Tensor) -> None:
    """Raise unless every entry is a non-negative integer (a count)."""
    if bool((data < 0).any()) or bool((data != torch.round(data)).any()):
        raise ValueError("Poisson likelihood needs non-negative integer counts")


class Poisson(_CumsumLikelihood):
    """
    Poisson (Gamma-Poisson) marginal likelihood for offline detection of
    changes in the rate of count data.

    Each segment's counts are i.i.d. Poisson with an unknown rate ``lambda``
    under a conjugate ``Gamma(alpha0, beta0)`` prior (shape, rate). The
    marginal likelihood of a segment of ``n`` counts with sum ``S`` is, in
    closed form (Gelman et al., *Bayesian Data Analysis*, 3rd ed., section
    2.6, Poisson model with gamma prior):

    ``log p = lgamma(alpha0 + S) - lgamma(alpha0) + alpha0 log(beta0)
    - (alpha0 + S) log(beta0 + n) - sum_i lgamma(x_i + 1)``.

    Multivariate input is treated as independent Poisson dimensions, as
    ``StudentT`` does, and their log marginals are summed.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Retained for backward compatibility (see ``BaseLikelihood``).
    alpha0 : float, optional
        Shape of the Gamma prior on the rate (default 1.0).
    beta0 : float, optional
        Rate of the Gamma prior on the rate (default 1.0). The prior mean
        rate is ``alpha0 / beta0``; a small ``beta0`` makes the prior vague.

    Raises
    ------
    ValueError
        If the data contain negative or non-integer values.

    Examples
    --------
    >>> import torch
    >>> likelihood = Poisson(alpha0=1.0, beta0=0.1)
    >>> counts = torch.poisson(torch.full((100,), 4.0))
    >>> log_marginal = likelihood.pdf(counts, 10, 50)
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True,
        *,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        if not (alpha0 > 0 and beta0 > 0):
            raise ValueError(
                f"alpha0 and beta0 must be positive, got {alpha0} and {beta0}"
            )
        super().__init__(device, cache_enabled)
        self.alpha0 = alpha0
        self.beta0 = beta0

    def _compute_stats(self, data: torch.Tensor) -> None:
        _check_counts(data)
        super()._compute_stats(data)
        zero = torch.zeros(1, data.shape[1], dtype=data.dtype, device=data.device)
        # L[k] = sum of lgamma(x + 1) = log(x!) over data[:k]
        self._L = torch.cat([zero, torch.cumsum(torch.lgamma(data + 1), dim=0)])

    def _log_marginal(self, t: int, s_hi: int) -> torch.Tensor:
        lengths, sum_x, _ = self._segment_moments(t, s_hi)
        log_factorials = self._L[t + 1 : s_hi + 1] - self._L[t]
        alpha_n = self.alpha0 + sum_x  # [m, d]
        beta_n = self.beta0 + lengths.unsqueeze(-1)  # [m, 1]
        log_marginal = (
            torch.lgamma(alpha_n)
            - math.lgamma(self.alpha0)
            + self.alpha0 * math.log(self.beta0)
            - alpha_n * torch.log(beta_n)
            - log_factorials
        )
        return log_marginal.sum(dim=-1)

    def pdf_rows(self, data: torch.Tensor, t: int) -> torch.Tensor:
        data = self.setup(data)
        return self._log_marginal(t, data.shape[0])

    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Log marginal likelihood of the counts ``data[t:s]``.

        Parameters
        ----------
        data : torch.Tensor
            Complete series of counts.
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).

        Returns
        -------
        float
            Log marginal likelihood of the segment.
        """
        if s <= t:
            return 0.0
        data = self.setup(data)
        return self._log_marginal(t, s)[-1].item()


class NormalKnownVariance(_CumsumLikelihood):
    """
    Normal likelihood with known variance and a conjugate Normal prior on the
    mean, for offline detection of changes in the mean when the noise level
    is known.

    Within a segment ``x_i ~ N(mu, variance)`` i.i.d. with
    ``mu ~ N(mu0, prior_variance)``. Integrating ``mu`` out, the ``n`` values
    of a segment are jointly Normal with mean ``mu0`` and covariance
    ``variance I + prior_variance 1 1^T`` (Murphy, "Conjugate Bayesian
    analysis of the Gaussian distribution", 2007, section 2), so with
    ``d_i = x_i - mu0`` and ``c = variance + n prior_variance``:

    ``log p = -n/2 log(2 pi) - (n-1)/2 log(variance) - 1/2 log(c)
    - (sum d_i^2 - prior_variance (sum d_i)^2 / c) / (2 variance)``.

    Multivariate input is treated as independent dimensions sharing the same
    hyperparameters, and their log marginals are summed.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Retained for backward compatibility (see ``BaseLikelihood``).
    variance : float, optional
        The known observation variance (default 1.0). Only changes in the
        mean are modeled; if the true variance differs, variance changes and
        misfit show up as mean changes.
    mu0 : float, optional
        Prior mean of the segment mean (default 0.0).
    prior_variance : float, optional
        Prior variance of the segment mean (default 1.0). Make it large
        compared with the spread of segment means you expect.

    Examples
    --------
    >>> import torch
    >>> likelihood = NormalKnownVariance(variance=0.25, prior_variance=100.0)
    >>> data = torch.cat([torch.randn(50) * 0.5, torch.randn(50) * 0.5 + 2])
    >>> log_marginal = likelihood.pdf(data, 0, 50)
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True,
        *,
        variance: float = 1.0,
        mu0: float = 0.0,
        prior_variance: float = 1.0,
    ):
        if not (variance > 0 and prior_variance > 0):
            raise ValueError(
                "variance and prior_variance must be positive, got "
                f"{variance} and {prior_variance}"
            )
        super().__init__(device, cache_enabled)
        self.variance = variance
        self.mu0 = mu0
        self.prior_variance = prior_variance

    def _compute_stats(self, data: torch.Tensor) -> None:
        # Prefix sums of the data centered on its mean: the within-segment
        # scatter below enters the marginal linearly, so computing it from
        # uncentered sums would lose it to cancellation for data far from 0
        # (240 nats of error at an offset of 1e8 before this was centered).
        # One global shift cannot center every segment: with regimes 1e6
        # noise standard deviations apart the marginals of single-regime
        # segments are still off by ~2e-3 nats (measured in the review of
        # #95); such a change is detected regardless.
        self._shift = data.mean(dim=0)  # [d]
        super()._compute_stats(data - self._shift)

    def _log_marginal(self, t: int, s_hi: int) -> torch.Tensor:
        # Centered sums; the shift cancels in the scatter and is added back
        # to the segment mean.
        lengths, sum_x, sum_x2 = self._segment_moments(t, s_hi)
        n = lengths.unsqueeze(-1)  # [m, 1]
        mean = sum_x / n  # segment mean minus the shift
        scatter = torch.clamp(sum_x2 - sum_x * mean, min=0.0)
        c = self.variance + n * self.prior_variance
        # sum (x - mu0)^2 - prior_variance (sum (x - mu0))^2 / c
        #   = scatter + n (mean - mu0)^2 variance / c, so
        # (shift - mu0) first: exact when both are large and close, where
        # (mean + shift) would round to the grid of the large value.
        deviation = mean + (self._shift - self.mu0)
        quadratic = scatter / self.variance + n * deviation**2 / c
        log_marginal = (
            -0.5 * n * _LOG_2PI
            - 0.5 * (n - 1) * math.log(self.variance)
            - 0.5 * torch.log(c)
            - 0.5 * quadratic
        )
        return log_marginal.sum(dim=-1)

    def pdf_rows(self, data: torch.Tensor, t: int) -> torch.Tensor:
        data = self.setup(data)
        return self._log_marginal(t, data.shape[0])

    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Log marginal likelihood of ``data[t:s]``.

        Parameters
        ----------
        data : torch.Tensor
            Complete time series.
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).

        Returns
        -------
        float
            Log marginal likelihood of the segment.
        """
        if s <= t:
            return 0.0
        data = self.setup(data)
        return self._log_marginal(t, s)[-1].item()
