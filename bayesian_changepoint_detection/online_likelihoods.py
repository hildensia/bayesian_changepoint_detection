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
        (default: identity). The prior mean of the precision is ``dof * W``,
        so the identity corresponds to unit prior covariance. Note this is a
        precision-side quantity: the posterior predictive covariance is
        proportional to ``W^{-1}``.
    device : str, torch.device, or None, optional
        Device to place tensors on.
        
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
            scale = torch.eye(dims, device=self.device, dtype=torch.float32)
        else:
            scale = ensure_tensor(scale, device=self.device)
        
        # Store initial parameters
        self.dof0 = dof
        self.kappa0 = kappa
        self.mu0 = mu.clone()
        self.scale0 = scale.clone()
        
        # Initialize parameter arrays (will grow over time)
        self.dof = torch.tensor([dof], device=self.device, dtype=torch.float32)
        self.kappa = torch.tensor([kappa], device=self.device, dtype=torch.float32)
        self.mu = mu.unsqueeze(0)  # Shape: [1, dims]
        self.scale = scale.unsqueeze(0)  # Shape: [1, dims, dims]
    
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
        #   x ~ t_{nu - D + 1}(mu, W^{-1} (kappa + 1) / (kappa (nu - D + 1)))
        # where ``self.scale`` is the Wishart scale matrix W on the *precision*
        # (the same parametrization ``update_theta`` maintains, and the one the
        # original NumPy implementation used). The t-distribution's shape
        # matrix is therefore W^{-1} / scale_factor, and its precision is
        # W * scale_factor, which needs no matrix inverse at all.
        t_dof = self.dof - self.dims + 1
        scale_factor = (self.kappa * t_dof) / (self.kappa + 1)

        precision = self.scale * scale_factor.unsqueeze(-1).unsqueeze(-1)
        # log|shape| = -log|W| - D log(scale_factor)
        logdet = -torch.logdet(self.scale) - self.dims * torch.log(scale_factor)

        # Mahalanobis distance for every run length
        diff = data.unsqueeze(0) - self.mu  # [t, dims]
        mahal_dist = torch.einsum("ti,tij,tj->t", diff, precision, diff)

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
        
        # Update scale matrices
        scale_update = (
            self.kappa.unsqueeze(1).unsqueeze(2) / 
            (self.kappa + 1).unsqueeze(1).unsqueeze(2)
        ) * torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1))
        
        # Regularized inverse to ensure numerical stability
        inv_scale = torch.inverse(self.scale + 1e-6 * torch.eye(self.dims, device=self.device, dtype=torch.float32).unsqueeze(0))
        scale_new = torch.inverse(
            inv_scale + scale_update
        )
        
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
        self.scale = torch.cat([self.scale0.unsqueeze(0), scale_new])