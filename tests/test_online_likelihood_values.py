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


@pytest.mark.math
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


@pytest.mark.math
@pytest.mark.parametrize("seed,dims", [(0, 2), (1, 3), (2, 4)])
def test_multivariate_t_matches_scipy_per_run_length(seed, dims):
    torch.manual_seed(seed)
    model = MultivariateT(dims=dims, device="cpu")
    data = torch.randn(20, dims) + torch.arange(dims, dtype=torch.float32)
    _feed(model, data[:-1])

    x = data[-1]
    got = model.pdf(x).double()

    # Independent reference: one scipy evaluation per run length using the
    # Normal-Wishart posterior predictive (Murphy 2007, eq. 258):
    # t_{nu-D+1}(mu, W^{-1} (kappa+1) / (kappa (nu-D+1))), where ``model.scale``
    # is the Wishart scale W on the precision.
    expected = []
    for r in range(model.mu.shape[0]):
        t_dof = float(model.dof[r] - dims + 1)
        shape = (
            np.linalg.inv(model.scale[r].double().numpy())
            * (float(model.kappa[r]) + 1)
            / (float(model.kappa[r]) * t_dof)
        )
        expected.append(
            multivariate_t(
                loc=model.mu[r].double().numpy(), shape=shape, df=t_dof
            ).logpdf(x.double().numpy())
        )
    expected = torch.tensor(expected, dtype=torch.float64)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.math
def test_multivariate_run_length_posterior_matches_independent_reference():
    """End-to-end: the run-length posterior produced with MultivariateT must
    match a NumPy/scipy implementation of the Normal-Wishart BOCPD recursion
    written directly from Murphy (2007). Before the predictive fix the MAP
    path collapsed to run lengths 1-3 on stationary data."""
    from functools import partial

    from bayesian_changepoint_detection import (
        constant_hazard,
        online_changepoint_detection,
    )
    from tests._reference_bocpd import reference_mv_bocpd

    rng = np.random.default_rng(0)
    dims = 3
    X = np.concatenate([rng.normal(0, 1, (30, dims)), rng.normal(3, 1, (30, dims))])
    expected = reference_mv_bocpd(
        X,
        lam=40,
        dof0=dims + 1,
        kappa0=1.0,
        mu0=np.zeros(dims),
        W0=np.eye(dims) / (dims + 1),  # the library default: unit prior covariance
    )
    R, _ = online_changepoint_detection(
        torch.tensor(X, dtype=torch.float32),
        partial(constant_hazard, 40, device="cpu"),
        MultivariateT(dims=dims, device="cpu"),
        device="cpu",
    )
    assert np.abs(R.numpy() - expected).max() < 1e-5  # measured 5e-7 in float32
    assert np.array_equal(R.numpy().argmax(axis=0), expected.argmax(axis=0))


@pytest.mark.behavior
def test_default_multivariate_prior_has_unit_covariance():
    """E[precision] = dof * W must be the identity by default."""
    dims = 4
    model = MultivariateT(dims=dims, device="cpu")
    assert torch.allclose(model.dof0 * model.scale0, torch.eye(dims))


@pytest.mark.math
@pytest.mark.parametrize("n,dims,sd", [(600, 2, 1.0), (1500, 3, 0.1)])
def test_multivariate_t_long_run_predictive_does_not_drift(n, dims, sd):
    """The predictive after a long run must match the closed-form
    Normal-Wishart posterior (Murphy 2007, eqs. 255-258) computed in float64
    from the batch sufficient statistics. Versions up to 1.1.0 kept W and
    added 1e-6 I to it before every inversion; on a shrinking W that bias
    compounds (7.6% in the posterior scale after 500 points, 67% after 3000)
    and the log predictive was off by 0.06 and 0.78 nats respectively."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, sd, (n, dims))
    model = MultivariateT(dims=dims, device="cpu")
    for i in range(n - 1):
        x = torch.tensor(X[i], dtype=torch.float32)
        model.pdf(x)
        model.update_theta(x)
    got = model.pdf(torch.tensor(X[-1], dtype=torch.float32))[-1].item()  # longest run

    Y = X[:-1]
    k0, nu0, mu0 = 1.0, dims + 1, np.zeros(dims)
    T0 = np.linalg.inv(np.eye(dims) / (dims + 1))  # inverse of the default W
    N = len(Y)
    ybar = Y.mean(0)
    S = (Y - ybar).T @ (Y - ybar)
    kN, nuN = k0 + N, nu0 + N
    muN = (k0 * mu0 + N * ybar) / kN
    TN = T0 + S + k0 * N / kN * np.outer(ybar - mu0, ybar - mu0)
    tdof = nuN - dims + 1
    expected = multivariate_t.logpdf(
        X[-1], loc=muN, shape=TN * (kN + 1) / (kN * tdof), df=tdof
    )
    # float32 lgamma at tdof ~ 1500 costs ~1e-3; master is off by 1e-1 here.
    assert abs(got - expected) < 3e-3
    assert np.allclose(model.scale_inv[-1].double().numpy(), TN, rtol=1e-4, atol=1e-2)


@pytest.mark.behavior
def test_scale_property_is_the_inverse_of_the_state():
    model = MultivariateT(dims=3, device="cpu")
    for x in torch.randn(5, 3):
        model.pdf(x)
        model.update_theta(x)
    eye = torch.eye(3).expand(model.scale_inv.shape[0], 3, 3)
    assert torch.allclose(torch.bmm(model.scale, model.scale_inv), eye, atol=1e-4)
    assert torch.allclose(model.scale[0], model.scale0, atol=1e-6)


@pytest.mark.math
def test_offline_and_online_multivariate_t_defaults_are_the_same_prior():
    """The offline MultivariateT's ``Psi0`` is the covariance-side scale
    (``inv(W)`` of the online class). With the defaults on both sides the
    chain-rule product of online one-step predictives over a segment must
    equal the offline closed-form marginal of that segment. Up to 1.1.0 the
    offline default was ``I`` instead of ``dof0 * I``, a prior ``dof0``
    times tighter than the online default (issue #75)."""
    from bayesian_changepoint_detection import offline_likelihoods

    torch.manual_seed(0)
    d = 3
    X = torch.randn(8, d) + 1.0
    online = MultivariateT(dims=d, device="cpu")
    chain = 0.0
    for x in X:
        chain += online.pdf(x)[-1].item()  # longest run: the whole prefix
        online.update_theta(x)
    offline = offline_likelihoods.MultivariateT(device="cpu")
    offline.setup(X.double())
    assert abs(chain - offline.pdf(X.double(), 0, len(X))) < 1e-3

    # and the documented rule Psi0 = dof0 * C matches scale = inv(C) / dof
    C = torch.tensor([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    dof = d + 1
    online = MultivariateT(
        dims=d, dof=dof, scale=torch.linalg.inv(C) / dof, device="cpu"
    )
    chain = 0.0
    for x in X:
        chain += online.pdf(x)[-1].item()
        online.update_theta(x)
    offline = offline_likelihoods.MultivariateT(
        dof0=dof, Psi0=dof * C.double(), device="cpu"
    )
    offline.setup(X.double())
    assert abs(chain - offline.pdf(X.double(), 0, len(X))) < 1e-3
