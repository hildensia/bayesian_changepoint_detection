"""
Online likelihood functions for Bayesian changepoint detection.

This module provides likelihood functions for online (sequential) changepoint detection
using PyTorch for efficient computation and GPU acceleration.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch

from .device import ensure_tensor, get_device


class BaseLikelihood(ABC):
    """
    Abstract base class for online likelihood functions.

    This class provides a template for implementing likelihood functions
    for online Bayesian changepoint detection. Subclasses must implement
    the pdf and update_theta methods.

    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on (CPU or GPU).
    """

    #: Names of the tensor attributes indexed by run length along their first
    #: dimension (entry ``r`` holds the posterior after a run of length
    #: ``r``). ``prune`` slices exactly these; a subclass that leaves it empty
    #: cannot be used with a bounded run length.
    _run_length_state: tuple = ()

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = get_device(device)
        self.t = 0  # Current time step

    def _subtract_shift(self, x: torch.Tensor, shift) -> torch.Tensor:
        """``x - shift`` as float32, with the subtraction in float64 where
        the device has it (not MPS). ``shift`` is a tuple of floats."""
        wide = torch.float32 if self.device.type == "mps" else torch.float64
        shift = torch.as_tensor(shift, dtype=wide, device=self.device)
        return (x.to(wide) - shift).to(torch.float32)

    def prune(self, n: int) -> None:
        """
        Keep the posterior parameters of run lengths ``0 .. n-1`` only.

        Used by ``OnlineChangepointDetector(max_run_length=...)`` to bound
        memory on an unbounded stream: after pruning, ``pdf`` returns ``n``
        densities.

        Raises
        ------
        NotImplementedError
            If the subclass does not declare ``_run_length_state``.
        """
        if not self._run_length_state:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare _run_length_state, "
                "so its per-run-length parameters cannot be pruned"
            )
        for name in self._run_length_state:
            setattr(self, name, getattr(self, name)[:n])

    def to(self, device: Union[str, torch.device]) -> "BaseLikelihood":
        """
        Move the likelihood model (and all its tensor state) to a device.

        Returns self, mirroring ``torch.nn.Module.to``.
        """
        device = get_device(device)
        if device == self.device:
            return self
        self.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    @abstractmethod
    def pdf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability density function for the observed data.

        Parameters
        ----------
        data : torch.Tensor
            The data point to evaluate (shape: [1] for univariate, [D] for multivariate).

        Returns
        -------
        torch.Tensor
            Log probability densities for all run lengths.
        """
        raise NotImplementedError("PDF method must be implemented in subclass.")

    @abstractmethod
    def update_theta(self, data: torch.Tensor, **kwargs) -> None:
        """
        Update the posterior parameters given new data.

        Parameters
        ----------
        data : torch.Tensor
            The new data point to incorporate.
        **kwargs
            Additional arguments (e.g., timestep t).
        """
        raise NotImplementedError(
            "update_theta method must be implemented in subclass."
        )


class StudentT(BaseLikelihood):
    """
    Univariate Student's t-distribution likelihood for online changepoint detection.

    Uses a Normal-Gamma conjugate prior, resulting in a Student's t predictive
    distribution. This is suitable for univariate data with unknown mean and variance.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter of the Gamma prior on precision (default: 0.1).
    beta : float, optional
        Rate parameter of the Gamma prior on precision (default: 0.1).
    kappa : float, optional
        Precision parameter of the Normal prior on mean (default: 1.0).
    mu : float, optional
        Mean parameter of the Normal prior on mean (default: 0.0).
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Examples
    --------
    >>> import torch
    >>> likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
    >>> data = torch.tensor(1.5)
    >>> log_probs = likelihood.pdf(data)
    >>> likelihood.update_theta(data)

    Notes
    -----
    The Student's t-distribution arises naturally as the predictive distribution
    when using Normal-Gamma conjugate priors for Gaussian data with unknown
    mean and variance.
    """

    _run_length_state = ("alpha", "beta", "kappa", "_mu_c")

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
        kappa: float = 1.0,
        mu: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(device)

        # Store initial hyperparameters
        self.alpha0 = alpha
        self.beta0 = beta
        self.kappa0 = kappa
        self.mu0 = mu

        # Initialize parameter vectors (will grow over time)
        self.alpha = torch.tensor([alpha], device=self.device, dtype=torch.float32)
        self.beta = torch.tensor([beta], device=self.device, dtype=torch.float32)
        self.kappa = torch.tensor([kappa], device=self.device, dtype=torch.float32)
        # Means are kept relative to the first observation (``_shift``), so
        # the float32 state does not cancel for data far from zero (the
        # model is translation-equivariant: shifting the data and ``mu``
        # together leaves every predictive density unchanged).
        self._shift = 0.0
        self._shift_set = False
        self._mu0_c = float(mu)
        self._mu_c = torch.tensor([mu], device=self.device, dtype=torch.float32)

    @property
    def mu(self) -> torch.Tensor:
        """Posterior mean for every run length (float32, for inspection)."""
        return self._mu_c + self._shift

    def _center(self, data: torch.Tensor) -> torch.Tensor:
        """The observation relative to the shift, which is set to the first
        observation seen (the state is still the prior at that point)."""
        if not self._shift_set:
            self._shift = float(data.reshape(()))
            self._shift_set = True
            self._mu0_c = float(self.mu0) - self._shift
            self._mu_c = torch.full_like(self._mu_c, self._mu0_c)
        # A Python float: the subtraction is float64 on every device, and a
        # scalar enters the tensor arithmetic below without building tensors.
        return float(data.reshape(())) - self._shift

    def pdf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability density under Student's t-distribution.

        Parameters
        ----------
        data : torch.Tensor
            Scalar data point to evaluate.

        Returns
        -------
        torch.Tensor
            Log probability densities for all current run lengths.
        """
        data = ensure_tensor(data, device=self.device)
        if data.numel() != 1:
            raise ValueError("StudentT expects scalar input data")
        data = self._center(data)

        self.t += 1

        # Student's t-distribution parameters
        df = 2 * self.alpha
        loc = self._mu_c
        scale = torch.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))

        # Log probabilities for all run lengths at once (same formula as
        # torch.distributions.StudentT.log_prob, vectorized over run lengths)
        z = (data - loc) / scale
        log_probs = (
            torch.lgamma((df + 1) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * torch.log(math.pi * df)
            - torch.log(scale)
            - ((df + 1) / 2) * torch.log1p(z**2 / df)
        )

        return log_probs

    def update_theta(self, data: torch.Tensor, **kwargs) -> None:
        """
        Update posterior parameters using conjugate prior updates.

        Parameters
        ----------
        data : torch.Tensor
            New data point to incorporate.
        """
        data = self._center(ensure_tensor(data, device=self.device))

        # Compute updated parameters
        mu_new = (self.kappa * self._mu_c + data) / (self.kappa + 1)
        kappa_new = self.kappa + 1.0
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + (self.kappa * (data - self._mu_c) ** 2) / (
            2.0 * (self.kappa + 1.0)
        )

        # Concatenate with initial parameters to maintain history
        self._mu_c = torch.cat(
            [
                torch.tensor([self._mu0_c], device=self.device, dtype=torch.float32),
                mu_new,
            ]
        )
        self.kappa = torch.cat(
            [
                torch.tensor([self.kappa0], device=self.device, dtype=torch.float32),
                kappa_new,
            ]
        )
        self.alpha = torch.cat(
            [
                torch.tensor([self.alpha0], device=self.device, dtype=torch.float32),
                alpha_new,
            ]
        )
        self.beta = torch.cat(
            [
                torch.tensor([self.beta0], device=self.device, dtype=torch.float32),
                beta_new,
            ]
        )


class MultivariateT(BaseLikelihood):
    """
    Multivariate Student's t-distribution likelihood for online changepoint detection.

    Uses a Normal-Wishart conjugate prior, resulting in a multivariate Student's t
    predictive distribution. Suitable for multivariate data with unknown mean and covariance.

    Parameters
    ----------
    dims : int
        Dimensionality of the data.
    dof : int, optional
        Initial degrees of freedom for Wishart prior (default: dims + 1).
    kappa : float, optional
        Precision parameter for Normal prior on mean (default: 1.0).
    mu : torch.Tensor or None, optional
        Prior mean vector (default: zero vector).
    scale : torch.Tensor or None, optional
        Prior scale matrix ``W`` of the Wishart distribution on the precision
        (default: ``I / dof``, which gives a prior mean precision of ``I``,
        i.e. unit prior covariance). Note this is a precision-side quantity:
        the posterior predictive covariance is proportional to ``W^{-1}``, so
        to encode a prior covariance ``C`` pass ``scale = inv(C) / dof``.
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Attributes
    ----------
    scale_inv : torch.Tensor
        The state actually maintained: ``T = W^{-1}`` for every run length,
        shape ``[t, dims, dims]`` (Murphy 2007, eq. 255). The update is the
        rank-one sum ``T + kappa/(kappa+1) (x-mu)(x-mu)^T``; the predictive
        uses a Cholesky factor of ``T``. No matrix is inverted per step.
    scale : torch.Tensor
        ``W = inv(scale_inv)`` per run length, computed on access for
        compatibility; not used internally.

    Notes
    -----
    All state is float32. The predictive's ``lgamma`` terms lose accuracy as
    the degrees of freedom grow with the run length: about 1e-6 nats at
    ``nu = 100``, 1e-3 at ``nu = 3000``, 6e-3 at ``nu = 50000``. Runs that
    long are rare in practice, but that is the accuracy ceiling.

    Examples
    --------
    >>> import torch
    >>> likelihood = MultivariateT(dims=3)
    >>> data = torch.randn(3)
    >>> log_probs = likelihood.pdf(data)
    >>> likelihood.update_theta(data)

    Notes
    -----
    The multivariate Student's t-distribution generalizes the univariate case
    to multiple dimensions, naturally handling correlations between variables.
    """

    _run_length_state = ("dof", "kappa", "_mu_c", "scale_inv")

    def __init__(
        self,
        dims: int,
        dof: Optional[int] = None,
        kappa: float = 1.0,
        mu: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(device)

        self.dims = dims

        # Set default parameters
        if dof is None:
            dof = dims + 1
        if mu is None:
            mu = torch.zeros(dims, device=self.device, dtype=torch.float32)
        else:
            mu = ensure_tensor(mu, device=self.device)
        if scale is None:
            # Unit prior covariance: E[precision] = dof * W = I  =>  W = I / dof.
            # (The pre-1.0 code used W = I with the same intent, which actually
            # encodes a prior covariance of I / dof and is too tight for D >> 1.)
            scale = torch.eye(dims, device=self.device, dtype=torch.float32) / dof
        else:
            scale = ensure_tensor(scale, device=self.device)

        # Store initial parameters
        self.dof0 = dof
        self.kappa0 = kappa
        self.mu0 = mu.clone()
        self.scale0 = scale.clone()
        # The recursion runs on T = W^{-1}; invert the prior once, here.
        # (float64 on the CPU: MPS has no float64, and this runs once.)
        self.scale_inv0 = torch.linalg.inv(scale.detach().cpu().double()).to(
            device=self.device, dtype=torch.float32
        )

        # Initialize parameter arrays (will grow over time)
        self.dof = torch.tensor([dof], device=self.device, dtype=torch.float32)
        self.kappa = torch.tensor([kappa], device=self.device, dtype=torch.float32)
        # Means relative to the first observation (``_shift``, a tuple of
        # floats so ``to()`` leaves it alone); see ``StudentT``.
        self._shift = (0.0,) * dims
        self._shift_set = False
        self._mu0_c = mu.to(torch.float32)
        self._mu_c = self._mu0_c.unsqueeze(0)  # Shape: [1, dims]
        self.scale_inv = self.scale_inv0.unsqueeze(0)  # Shape: [1, dims, dims]

    @property
    def mu(self) -> torch.Tensor:
        """Posterior mean for every run length, ``[t, dims]`` (float32)."""
        return self._mu_c + torch.as_tensor(
            self._shift, dtype=torch.float32, device=self._mu_c.device
        )

    def _center(self, data: torch.Tensor) -> torch.Tensor:
        """The observation relative to the shift (the first observation)."""
        if not self._shift_set:
            self._shift = tuple(float(v) for v in data.detach().cpu().double())
            self._shift_set = True
            self._mu0_c = self._subtract_shift(self.mu0, self._shift)
            self._mu_c = self._mu0_c.unsqueeze(0).expand_as(self._mu_c).clone()
        return self._subtract_shift(data, self._shift)

    @property
    def scale(self) -> torch.Tensor:
        """Wishart scale ``W`` per run length (``inv(scale_inv)``), for inspection."""
        return torch.linalg.inv(self.scale_inv)

    def _cholesky(self) -> torch.Tensor:
        """Lower Cholesky factor of ``scale_inv`` for every run length.

        ``T`` only ever grows by positive semi-definite rank-one terms from an
        SPD prior, so it is SPD in exact arithmetic; if float32 rounding on a
        long, badly scaled run still breaks the factorization, retry once with
        a jitter proportional to the matrix scale rather than fail.
        """
        L, info = torch.linalg.cholesky_ex(self.scale_inv)
        if bool((info != 0).any()):
            # Jitter only the run lengths whose factorization failed.
            failed = (info != 0).to(torch.float32)
            trace = torch.diagonal(self.scale_inv, dim1=-2, dim2=-1).sum(-1)
            jitter = (1e-6 * trace / self.dims).clamp(min=1e-6) * failed
            eye = torch.eye(self.dims, device=self.device, dtype=torch.float32)
            L, info = torch.linalg.cholesky_ex(
                self.scale_inv + jitter.unsqueeze(-1).unsqueeze(-1) * eye
            )
            if bool((info != 0).any()):
                raise torch.linalg.LinAlgError(
                    "MultivariateT: the inverse Wishart scale is not positive "
                    "definite for some run length; check the data scale or "
                    "pass a better-conditioned prior `scale`."
                )
        return L

    def pdf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability density under multivariate Student's t-distribution.

        Parameters
        ----------
        data : torch.Tensor
            Data vector to evaluate (shape: [dims]).

        Returns
        -------
        torch.Tensor
            Log probability densities for all current run lengths.
        """
        data = ensure_tensor(data, device=self.device)
        if data.shape != (self.dims,):
            raise ValueError(f"Expected data shape [{self.dims}], got {data.shape}")
        data = self._center(data)

        self.t += 1

        # Posterior predictive of the Normal-Wishart model (Murphy 2007,
        # "Conjugate Bayesian analysis of the Gaussian distribution", eq. 258):
        #   x ~ t_{nu - D + 1}(mu, T (kappa + 1) / (kappa (nu - D + 1)))
        # with T = W^{-1} the state kept in ``self.scale_inv``. The shape
        # matrix is Sigma = T / scale_factor, so with T = L L^T:
        #   (x-mu)^T Sigma^{-1} (x-mu) = scale_factor * ||L^{-1} (x-mu)||^2
        #   log|Sigma| = 2 sum log diag(L) - D log(scale_factor).
        t_dof = self.dof - self.dims + 1
        scale_factor = (self.kappa * t_dof) / (self.kappa + 1)

        L = self._cholesky()  # [t, D, D]
        diff = data.unsqueeze(0) - self._mu_c  # [t, D]
        y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
        mahal_dist = scale_factor * (y.squeeze(-1) ** 2).sum(-1)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(
            -1
        ) - self.dims * torch.log(scale_factor)

        log_probs = (
            torch.lgamma((t_dof + self.dims) / 2)
            - torch.lgamma(t_dof / 2)
            - (self.dims / 2) * torch.log(t_dof * torch.pi)
            - 0.5 * logdet
            - ((t_dof + self.dims) / 2) * torch.log1p(mahal_dist / t_dof)
        )

        return log_probs

    def update_theta(self, data: torch.Tensor, **kwargs) -> None:
        """
        Update posterior parameters using Normal-Wishart conjugate updates.

        Parameters
        ----------
        data : torch.Tensor
            New data vector to incorporate.
        """
        data = self._center(ensure_tensor(data, device=self.device))

        # Compute differences from current means
        centered = data.unsqueeze(0) - self._mu_c  # Shape: [t, dims]

        # Update parameters using conjugate prior formulas
        mu_new = (self.kappa.unsqueeze(1) * self._mu_c + data.unsqueeze(0)) / (
            self.kappa + 1
        ).unsqueeze(1)

        kappa_new = self.kappa + 1
        dof_new = self.dof + 1

        # T_n = T + kappa/(kappa+1) (x-mu)(x-mu)^T  (Murphy 2007, eq. 255):
        # a rank-one update of the inverse Wishart scale, no inversion.
        scale_inv_new = self.scale_inv + (self.kappa / (self.kappa + 1)).unsqueeze(
            -1
        ).unsqueeze(-1) * torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1))

        # Concatenate with initial parameters
        self._mu_c = torch.cat([self._mu0_c.unsqueeze(0), mu_new])
        self.kappa = torch.cat(
            [
                torch.tensor([self.kappa0], device=self.device, dtype=torch.float32),
                kappa_new,
            ]
        )
        self.dof = torch.cat(
            [
                torch.tensor([self.dof0], device=self.device, dtype=torch.float32),
                dof_new,
            ]
        )
        self.scale_inv = torch.cat([self.scale_inv0.unsqueeze(0), scale_inv_new])


class Poisson(BaseLikelihood):
    """
    Poisson likelihood with a conjugate Gamma prior, for online detection of
    changes in the rate of count data.

    For each run length the rate has a ``Gamma(alpha, beta)`` posterior
    (shape, rate); the predictive of the next count is negative binomial:

    ``p(x) = Gamma(alpha + x) / (Gamma(alpha) x!) (beta / (beta + 1))^alpha
    (1 / (beta + 1))^x``,

    and after observing ``x`` the posterior becomes ``Gamma(alpha + x,
    beta + 1)`` (Gelman et al., *Bayesian Data Analysis*, 3rd ed., section
    2.6).

    Parameters
    ----------
    alpha : float, optional
        Prior shape (default 1.0).
    beta : float, optional
        Prior rate (default 1.0). The prior mean rate is ``alpha / beta``; a
        small ``beta`` makes the prior vague.
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Raises
    ------
    ValueError
        From ``pdf`` if an observation is not a single non-negative integer.

    Examples
    --------
    >>> import torch
    >>> likelihood = Poisson(alpha=1.0, beta=0.1)
    >>> log_probs = likelihood.pdf(torch.tensor(3.0))
    >>> likelihood.update_theta(torch.tensor(3.0))
    """

    _run_length_state = ("alpha", "beta")

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if not (alpha > 0 and beta > 0):
            raise ValueError(f"alpha and beta must be positive, got {alpha} and {beta}")
        super().__init__(device)
        self.alpha0 = alpha
        self.beta0 = beta
        self.alpha = torch.tensor([alpha], device=self.device, dtype=torch.float32)
        self.beta = torch.tensor([beta], device=self.device, dtype=torch.float32)

    def pdf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Log negative-binomial predictive of one count for every run length.

        Parameters
        ----------
        data : torch.Tensor
            A single non-negative integer count.

        Returns
        -------
        torch.Tensor
            Log predictive probabilities, one per current run length.
        """
        data = ensure_tensor(data, device=self.device)
        if data.numel() != 1:
            raise ValueError("Poisson expects scalar input data")
        x = data.reshape(()).to(torch.float32)
        if bool(x < 0) or bool(x != torch.round(x)):
            raise ValueError(
                f"Poisson expects a non-negative integer count, got {x.item()}"
            )
        self.t += 1
        return (
            torch.lgamma(self.alpha + x)
            - torch.lgamma(self.alpha)
            - torch.lgamma(x + 1)
            + self.alpha * torch.log(self.beta / (self.beta + 1))
            - x * torch.log1p(self.beta)
        )

    def update_theta(self, data: torch.Tensor, **kwargs) -> None:
        """
        Conjugate update ``alpha += x``, ``beta += 1`` for every run length,
        with the prior prepended for run length 0.

        Parameters
        ----------
        data : torch.Tensor
            The count just observed.
        """
        x = ensure_tensor(data, device=self.device).reshape(()).to(torch.float32)
        prior_alpha = torch.tensor(
            [self.alpha0], device=self.device, dtype=torch.float32
        )
        prior_beta = torch.tensor([self.beta0], device=self.device, dtype=torch.float32)
        self.alpha = torch.cat([prior_alpha, self.alpha + x])
        self.beta = torch.cat([prior_beta, self.beta + 1.0])


class NormalKnownVariance(BaseLikelihood):
    """
    Normal likelihood with known variance and a conjugate Normal prior on the
    mean, for online detection of changes in the mean when the noise level is
    known.

    For each run length the segment mean has a ``N(mu_r, v_r)`` posterior;
    the predictive of the next observation is ``N(mu_r, v_r + variance)``,
    and after observing ``x`` (Murphy, "Conjugate Bayesian analysis of the
    Gaussian distribution", 2007, section 2):

    ``v' = 1 / (1 / v_r + 1 / variance)``,
    ``mu' = v' (mu_r / v_r + x / variance)``.

    Parameters
    ----------
    variance : float, optional
        The known observation variance (default 1.0).
    mu : float, optional
        Prior mean of the segment mean (default 0.0).
    prior_variance : float, optional
        Prior variance of the segment mean (default 1.0).
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Examples
    --------
    >>> import torch
    >>> likelihood = NormalKnownVariance(variance=0.25, prior_variance=100.0)
    >>> log_probs = likelihood.pdf(torch.tensor(0.3))
    >>> likelihood.update_theta(torch.tensor(0.3))
    """

    _run_length_state = ("_mu_c", "var")

    def __init__(
        self,
        variance: float = 1.0,
        mu: float = 0.0,
        prior_variance: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if not (variance > 0 and prior_variance > 0):
            raise ValueError(
                "variance and prior_variance must be positive, got "
                f"{variance} and {prior_variance}"
            )
        super().__init__(device)
        self.variance = variance
        self.mu0 = mu
        self.prior_variance = prior_variance
        # Means relative to the first observation; see ``StudentT``.
        self._shift = 0.0
        self._shift_set = False
        self._mu0_c = float(mu)
        self._mu_c = torch.tensor([mu], device=self.device, dtype=torch.float32)
        self.var = torch.tensor(
            [prior_variance], device=self.device, dtype=torch.float32
        )

    @property
    def mu(self) -> torch.Tensor:
        """Posterior mean for every run length (float32, for inspection)."""
        return self._mu_c + self._shift

    def _center(self, data: torch.Tensor) -> torch.Tensor:
        if not self._shift_set:
            self._shift = float(data.reshape(()))
            self._shift_set = True
            self._mu0_c = float(self.mu0) - self._shift
            self._mu_c = torch.full_like(self._mu_c, self._mu0_c)
        # A Python float: the subtraction is float64 on every device, and a
        # scalar enters the tensor arithmetic below without building tensors.
        return float(data.reshape(())) - self._shift

    def pdf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Log Normal predictive of one observation for every run length.

        Parameters
        ----------
        data : torch.Tensor
            Scalar observation.

        Returns
        -------
        torch.Tensor
            Log predictive densities, one per current run length.
        """
        data = ensure_tensor(data, device=self.device)
        if data.numel() != 1:
            raise ValueError("NormalKnownVariance expects scalar input data")
        x = self._center(data)
        self.t += 1
        predictive_var = self.var + self.variance
        return -0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(predictive_var)
            + (x - self._mu_c) ** 2 / predictive_var
        )

    def update_theta(self, data: torch.Tensor, **kwargs) -> None:
        """
        Conjugate update of the mean's posterior for every run length, with
        the prior prepended for run length 0.

        Parameters
        ----------
        data : torch.Tensor
            The observation just seen.
        """
        x = self._center(ensure_tensor(data, device=self.device))
        var_new = 1.0 / (1.0 / self.var + 1.0 / self.variance)
        mu_new = var_new * (self._mu_c / self.var + x / self.variance)
        prior_mu = torch.tensor([self._mu0_c], device=self.device, dtype=torch.float32)
        prior_var = torch.tensor(
            [self.prior_variance], device=self.device, dtype=torch.float32
        )
        self._mu_c = torch.cat([prior_mu, mu_new])
        self.var = torch.cat([prior_var, var_new])
