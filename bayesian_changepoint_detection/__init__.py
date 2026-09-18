#!/usr/bin/env python3
"""
Bayesian Changepoint Detection Library.

A PyTorch-based library for Bayesian changepoint detection in time series data.
Implements both online and offline methods with GPU acceleration support.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    # Single source of truth is the "version" field in pyproject.toml
    __version__ = _version("bayescd")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "unknown"

from . import generate_data, offline_likelihoods, online_likelihoods
from .bayesian_models import (
    changepoint_probabilities,
    compute_run_length_posterior,
    get_map_changepoints,
    offline_changepoint_detection,
    online_changepoint_detection,
    viterbi_changepoints,
)
from .device import ensure_tensor, get_device, get_device_info, to_tensor
from .hazard_functions import constant_hazard
from .online_likelihoods import MultivariateT, StudentT
from .priors import const_prior, geometric_prior, negative_binomial_prior

__all__ = [
    "get_device",
    "to_tensor",
    "ensure_tensor",
    "get_device_info",
    "online_changepoint_detection",
    "offline_changepoint_detection",
    "changepoint_probabilities",
    "get_map_changepoints",
    "compute_run_length_posterior",
    "viterbi_changepoints",
    "constant_hazard",
    "const_prior",
    "geometric_prior",
    "negative_binomial_prior",
    "StudentT",
    "MultivariateT",
    "online_likelihoods",
    "offline_likelihoods",
    "generate_data",
]
