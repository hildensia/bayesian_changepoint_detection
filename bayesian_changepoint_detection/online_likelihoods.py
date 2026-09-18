"""
Online likelihood functions for Bayesian changepoint detection.

This module provides likelihood functions for online (sequential) changepoint detection
using PyTorch for efficient computation and GPU acceleration.
"""

import math

import torch
import torch.distributions as dist
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
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
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = get_device(device)
        self.t = 0  # Current time step

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
        raise NotImplementedError(
            "PDF method must be implemented in subclass."
        )
    
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
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
        kappa: float = 1.0,
        mu: float = 0.0,
        device: Optional[Union[str, torch.device]] = None
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
        self.mu = torch.tensor([mu], device=self.device, dtype=torch.float32)
    
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
        
        self.t += 1

        # Student's t-distribution parameters
        df = 2 * self.alpha
        loc = self.mu
        scale = torch.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))

        # Log probabilities for all run lengths at once (same formula as
        # torch.distributions.StudentT.log_prob, vectorized over run lengths)
        z = (data - loc) / scale
        log_probs = (
            torch.lgamma((df + 1) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * torch.log(math.pi * df)
            - torch.log(scale)
            - ((df + 1) / 2) * torch.log1p(z ** 2 / df)
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
        data = ensure_tensor(data, device=self.device)
        
        # Compute updated parameters
        mu_new = (self.kappa * self.mu + data) / (self.kappa + 1)
        kappa_new = self.kappa + 1.0
        alpha_new = self.alpha + 0.5
        beta_new = (
            self.beta + 
            (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))
        )
        
        # Concatenate with initial parameters to maintain history
        self.mu = torch.cat([
            torch.tensor([self.mu0], device=self.device, dtype=torch.float32),
            mu_new
        ])
        self.kappa = torch.cat([
            torch.tensor([self.kappa0], device=self.device, dtype=torch.float32),
            kappa_new
        ])
        self.alpha = torch.cat([
            torch.tensor([self.alpha0], device=self.device, dtype=torch.float32),
            alpha_new
        ])
        self.beta = torch.cat([
            torch.tensor([self.beta0], device=self.device, dtype=torch.float32),
            beta_new
        ])


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
    
    def __init__(
        self,
        dims: int,
        dof: Optional[int] = None,
        kappa: float = 1.0,
        mu: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None
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
        self.mu = mu.unsqueeze(0)  # Shape: [1, dims]
        self.scale_inv = self.scale_inv0.unsqueeze(0)  # Shape: [1, dims, dims]

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

        L = self._cholesky()                                   # [t, D, D]
        diff = data.unsqueeze(0) - self.mu                     # [t, D]
        y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
        mahal_dist = scale_factor * (y.squeeze(-1) ** 2).sum(-1)
        logdet = (
            2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            - self.dims * torch.log(scale_factor)
        )

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
        data = ensure_tensor(data, device=self.device)
        
        # Compute differences from current means
        centered = data.unsqueeze(0) - self.mu  # Shape: [t, dims]
        
        # Update parameters using conjugate prior formulas
        mu_new = (
            self.kappa.unsqueeze(1) * self.mu + data.unsqueeze(0)
        ) / (self.kappa + 1).unsqueeze(1)
        
        kappa_new = self.kappa + 1
        dof_new = self.dof + 1
        
        # T_n = T + kappa/(kappa+1) (x-mu)(x-mu)^T  (Murphy 2007, eq. 255):
        # a rank-one update of the inverse Wishart scale, no inversion.
        scale_inv_new = self.scale_inv + (
            self.kappa / (self.kappa + 1)
        ).unsqueeze(-1).unsqueeze(-1) * torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1))
        
        # Concatenate with initial parameters
        self.mu = torch.cat([self.mu0.unsqueeze(0), mu_new])
        self.kappa = torch.cat([
            torch.tensor([self.kappa0], device=self.device, dtype=torch.float32),
            kappa_new
        ])
        self.dof = torch.cat([
            torch.tensor([self.dof0], device=self.device, dtype=torch.float32),
            dof_new
        ])
        self.scale_inv = torch.cat([self.scale_inv0.unsqueeze(0), scale_inv_new])