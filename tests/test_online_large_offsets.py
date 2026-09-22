"""
Online likelihoods stay accurate for data far from zero.

The online state is float32. Before the likelihoods kept their means
relative to the first observation, an offset of 1e6 with unit noise moved
the run-length posterior by up to 0.84 and added three spurious
changepoints. ``math`` tests compare with float64 references that share no
code with the library (a NumPy Normal-Gamma recursion written here, and
``reference_mv_bocpd``); ``behavior`` tests pin translation equivariance.
"""

from functools import partial

import numpy as np
import pytest
import torch
from scipy.special import gammaln

from bayesian_changepoint_detection import constant_hazard, online_changepoint_detection
from bayesian_changepoint_detection.online_likelihoods import (
    MultivariateT,
    NormalKnownVariance,
    StudentT,
)
from tests._reference_bocpd import reference_mv_bocpd

HAZARD = partial(constant_hazard, 250, device="cpu")


def reference_student_t(x, lam, alpha0, beta0, kappa0, mu0):
    """Adams & MacKay with the Normal-Gamma predictive, float64 NumPy."""
    T = len(x)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    a, b = np.array([alpha0]), np.array([beta0])
    k, m = np.array([kappa0]), np.array([mu0])
    for t in range(T):
        df = 2 * a
        scale = np.sqrt(b * (k + 1) / (a * k))
        z = (x[t] - m) / scale
        log_p = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(np.pi * df)
            - np.log(scale)
            - (df + 1) / 2 * np.log1p(z * z / df)
        )
        p = np.exp(log_p - log_p.max())
        R[1 : t + 2, t + 1] = R[: t + 1, t] * p * (1 - 1 / lam)
        R[0, t + 1] = np.sum(R[: t + 1, t] * p / lam)
        R[:, t + 1] /= R[:, t + 1].sum()
        m_new = (k * m + x[t]) / (k + 1)
        b_new = b + k * (x[t] - m) ** 2 / (2 * (k + 1))
        m, k = np.r_[mu0, m_new], np.r_[kappa0, k + 1]
        a, b = np.r_[alpha0, a + 0.5], np.r_[beta0, b_new]
    return R


def three_segments(offset, length=100, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(offset, 1, length),
            rng.normal(offset + 3, 1, length),
            rng.normal(offset, 1, length),
        ]
    )


@pytest.mark.math
@pytest.mark.parametrize("offset", [0.0, 1e3, 1e5, 1e6, 1e8])
def test_student_t_matches_float64_reference(offset):
    x = three_segments(offset)
    expected = reference_student_t(x, 250, 0.1, 0.01, 1.0, offset)
    R, map_run_lengths = online_changepoint_detection(
        torch.tensor(x, dtype=torch.float64),
        HAZARD,
        StudentT(alpha=0.1, beta=0.01, kappa=1, mu=offset, device="cpu"),
        device="cpu",
    )
    assert np.abs(R.double().numpy() - expected).max() < 1e-4  # 3e-6 measured
    assert np.array_equal(map_run_lengths.numpy(), expected.argmax(axis=0))


@pytest.mark.math
@pytest.mark.parametrize("offset", [0.0, 1e4, 1e6])
def test_multivariate_t_matches_float64_reference(offset):
    rng = np.random.default_rng(1)
    dims = 2
    X = np.concatenate(
        [rng.normal(offset, 1, (25, dims)), rng.normal(offset + 3, 1, (25, dims))]
    )
    expected = reference_mv_bocpd(
        X,
        lam=40,
        dof0=dims + 1,
        kappa0=1.0,
        mu0=np.full(dims, offset),
        W0=np.eye(dims) / (dims + 1),
    )
    R, _ = online_changepoint_detection(
        torch.tensor(X, dtype=torch.float64),
        partial(constant_hazard, 40, device="cpu"),
        MultivariateT(
            dims=dims, mu=torch.full((dims,), offset, dtype=torch.float64), device="cpu"
        ),
        device="cpu",
    )
    assert np.abs(R.double().numpy() - expected).max() < 1e-4
    assert np.array_equal(R.numpy().argmax(axis=0), expected.argmax(axis=0))


@pytest.mark.behavior
@pytest.mark.parametrize(
    "make",
    [
        lambda mu: StudentT(alpha=0.1, beta=0.01, kappa=1, mu=mu, device="cpu"),
        lambda mu: NormalKnownVariance(
            variance=1.0, mu=mu, prior_variance=9.0, device="cpu"
        ),
    ],
    ids=["StudentT", "NormalKnownVariance"],
)
def test_translation_equivariance(make):
    # Shifting the data and the prior mean together must not change R.
    x = three_segments(0.0, length=60, seed=2)
    base, _ = online_changepoint_detection(
        torch.tensor(x, dtype=torch.float64), HAZARD, make(0.0), device="cpu"
    )
    shifted, _ = online_changepoint_detection(
        torch.tensor(x + 1e7, dtype=torch.float64), HAZARD, make(1e7), device="cpu"
    )
    assert torch.allclose(base, shifted, atol=1e-5)


@pytest.mark.behavior
def test_mu_is_reported_in_data_units():
    model = StudentT(mu=5.0, device="cpu")
    assert model.mu.tolist() == [5.0]
    model.pdf(torch.tensor(1e6, dtype=torch.float64))
    model.update_theta(torch.tensor(1e6, dtype=torch.float64))
    # Run length 0 is the prior; run length 1 has seen one point at 1e6.
    assert model.mu[0].item() == pytest.approx(5.0, abs=0.1)
    assert model.mu[1].item() == pytest.approx((5.0 + 1e6) / 2, rel=1e-6)
