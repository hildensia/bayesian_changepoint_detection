"""
Numeric regression tests for the batched online predictive densities.

The online ``pdf`` methods evaluate one observation under the posterior
predictive of every run length in a single vectorized expression. These tests
compare each run length's value against an independent reference:
``torch.distributions.StudentT`` for the univariate model and
``scipy.stats.multivariate_t`` for the multivariate one.
"""

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_t

from bayesian_changepoint_detection.online_likelihoods import MultivariateT, StudentT


def _feed(model, observations):
    """Run the pdf/update cycle over a sequence and return the last pdf."""
    log_probs = None
    for x in observations:
        log_probs = model.pdf(x)
        model.update_theta(x)
    return log_probs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_studentt_matches_torch_distribution_per_run_length(seed):
    torch.manual_seed(seed)
    model = StudentT(alpha=0.5, beta=0.3, kappa=2.0, mu=0.5, device="cpu")
    data = torch.randn(25) * 1.5 + 0.2
    _feed(model, data[:-1])

    x = data[-1]
    got = model.pdf(x)

    df = 2 * model.alpha
    scale = torch.sqrt(model.beta * (model.kappa + 1) / (model.alpha * model.kappa))
    expected = torch.distributions.StudentT(df, loc=model.mu, scale=scale).log_prob(x)

    assert got.shape == expected.shape == (25,)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed,dims", [(0, 2), (1, 3), (2, 4)])
def test_multivariate_t_matches_scipy_per_run_length(seed, dims):
    torch.manual_seed(seed)
    model = MultivariateT(dims=dims, device="cpu")
    data = torch.randn(20, dims) + torch.arange(dims, dtype=torch.float32)
    _feed(model, data[:-1])

    x = data[-1]
    got = model.pdf(x).double()

    # Independent reference: one scipy evaluation per run length, using the
    # same posterior-predictive parametrization as the implementation
    # (Murphy 2007, Normal-Wishart posterior predictive).
    expected = []
    for r in range(model.mu.shape[0]):
        t_dof = float(model.dof[r] - dims + 1)
        scale_factor = float(model.kappa[r] * t_dof / (model.kappa[r] + 1))
        shape = (model.scale[r] / scale_factor).double().numpy()
        shape = shape + 1e-6 * np.eye(dims)  # same regularization as pdf()
        expected.append(
            multivariate_t(loc=model.mu[r].double().numpy(), shape=shape, df=t_dof)
            .logpdf(x.double().numpy())
        )
    expected = torch.tensor(expected, dtype=torch.float64)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)
