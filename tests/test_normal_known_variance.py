"""
Tests for the Normal likelihoods with known variance, online and offline.

``math`` tests check the closed forms against scipy: the offline segment
marginal against ``scipy.stats.multivariate_normal`` with the joint
covariance ``variance I + prior_variance 1 1^T`` built explicitly, and
against the chain rule of one-step Normal predictives; the online
predictive against ``scipy.stats.norm`` after hand-computed conjugate
updates. ``behavior`` tests pin detection rates, streaming and input checks.
"""

import math
from fractions import Fraction
from functools import partial

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal, norm

from bayesian_changepoint_detection import (
    OnlineChangepointDetector,
    changepoint_probabilities,
    const_prior,
    constant_hazard,
    offline_changepoint_detection,
    offline_likelihoods,
    online_changepoint_detection,
    online_likelihoods,
)

SIGMA = 0.5  # the known noise standard deviation used by the synthetic data


def mean_shift(seed, means=(0.0, 1.5, 0.0), length=60, dims=None):
    gen = torch.Generator().manual_seed(seed)
    shape = (length,) if dims is None else (length, dims)
    return torch.cat(
        [
            torch.randn(shape, generator=gen, dtype=torch.float64) * SIGMA + m
            for m in means
        ]
    )


def posterior_after(values, mu0, prior_var, var):
    """Conjugate Normal update for the mean, one value at a time."""
    mu, v = mu0, prior_var
    for x in values:
        v_new = 1.0 / (1.0 / v + 1.0 / var)
        mu, v = v_new * (mu / v + x / var), v_new
    return mu, v


def chain_rule(values, mu0, prior_var, var):
    total, mu, v = 0.0, mu0, prior_var
    for x in values:
        total += norm.logpdf(x, mu, np.sqrt(v + var))
        mu, v = posterior_after([x], mu, v, var)
    return total


@pytest.mark.math
@pytest.mark.parametrize(
    "var, mu0, prior_var", [(1.0, 0.0, 1.0), (0.25, 0.7, 10.0), (4.0, -2.0, 0.3)]
)
def test_offline_marginal_is_the_joint_normal(var, mu0, prior_var):
    data = mean_shift(0, length=12)
    model = offline_likelihoods.NormalKnownVariance(
        device="cpu", variance=var, mu0=mu0, prior_variance=prior_var
    )
    array = data.numpy()
    for t, s in [(0, 1), (0, 12), (5, 20), (10, 36)]:
        n = s - t
        cov = var * np.eye(n) + prior_var * np.ones((n, n))
        joint = multivariate_normal.logpdf(array[t:s], mean=np.full(n, mu0), cov=cov)
        assert model.pdf(data, t, s) == pytest.approx(joint, abs=1e-9)
        assert model.pdf(data, t, s) == pytest.approx(
            chain_rule(array[t:s], mu0, prior_var, var), abs=1e-9
        )
    rows = model.pdf_rows(data, 4).numpy()
    for j, value in enumerate(rows):
        expected = chain_rule(array[4 : 4 + 1 + j], mu0, prior_var, var)
        assert value == pytest.approx(expected, abs=1e-9)


def exact_log_marginal(values, var, mu0, prior_var):
    """The closed form evaluated in exact rational arithmetic (the float
    data are exact rationals), rounded only at the final logs."""
    n = len(values)
    d = [Fraction(x) - Fraction(mu0) for x in values]
    c = Fraction(var) + n * Fraction(prior_var)
    quadratic = (
        sum(v * v for v in d) - Fraction(prior_var) * sum(d) ** 2 / c
    ) / Fraction(var)
    return (
        -0.5 * n * math.log(2 * math.pi)
        - 0.5 * (n - 1) * math.log(var)
        - 0.5 * math.log(c)
        - 0.5 * float(quadratic)
    )


@pytest.mark.math
@pytest.mark.parametrize("offset", [1e4, 1e6, 1e8])
@pytest.mark.parametrize("mu0_at_data", [False, True], ids=["mu0=0", "mu0=offset"])
def test_offline_marginal_is_accurate_far_from_zero(offset, mu0_at_data):
    # Uncentered prefix sums lost 1e-2 nats at offset 1e6 and 240 at 1e8
    # (review of #95); the sums are now centered on the data mean.
    gen = torch.Generator().manual_seed(7)
    data = torch.randn(100, generator=gen, dtype=torch.float64) + offset
    mu0 = offset if mu0_at_data else 0.0
    model = offline_likelihoods.NormalKnownVariance(
        device="cpu", variance=1.0, mu0=mu0, prior_variance=1.0
    )
    for t, s in [(0, 100), (10, 60), (95, 100)]:
        expected = exact_log_marginal(data[t:s].tolist(), 1.0, mu0, 1.0)
        assert model.pdf(data, t, s) == pytest.approx(expected, rel=1e-12, abs=1e-9)


@pytest.mark.math
def test_offline_multivariate_sums_independent_dimensions():
    data = mean_shift(1, length=10, dims=2)
    model = offline_likelihoods.NormalKnownVariance(
        device="cpu", variance=0.25, prior_variance=5.0
    )
    expected = sum(chain_rule(data[3:17, d].numpy(), 0.0, 5.0, 0.25) for d in range(2))
    assert model.pdf(data, 3, 17) == pytest.approx(expected, abs=1e-9)


@pytest.mark.math
def test_online_predictive_is_normal():
    values = [0.3, -0.1, 1.2, 0.8]
    var, mu0, prior_var = 0.25, 0.5, 2.0
    model = online_likelihoods.NormalKnownVariance(
        variance=var, mu=mu0, prior_variance=prior_var, device="cpu"
    )
    for x in values:
        model.update_theta(torch.tensor(x))
    x_new = 0.9
    log_probs = model.pdf(torch.tensor(x_new)).double().numpy()
    for r in range(len(values) + 1):
        mu, v = posterior_after(values[len(values) - r :], mu0, prior_var, var)
        expected = norm.logpdf(x_new, mu, np.sqrt(v + var))
        assert log_probs[r] == pytest.approx(expected, abs=1e-5)


def detected(probs, true, window=3):
    """At least 0.9 of the probability within +-window of each true change,
    and no single position outside those windows above 0.5.

    A windowed criterion, because the posterior legitimately splits a change
    across neighboring positions when the values there are ambiguous (seed
    9 offline: 0.36 / 0.19 / 0.45 at 58 / 59 / 60)."""
    inside = torch.zeros_like(probs, dtype=torch.bool)
    for change in true:
        if probs[change - window : change + window + 1].sum() < 0.9:
            return False
        inside[change - window : change + window + 1] = True
    return float(probs[~inside].max()) < 0.5


# Rate over 20 draws: mean 0 -> 1.5 -> 0, noise sd 0.5 (known), 60 points
# per segment. Measured when written: online 17/20 (the misses hold 0.79 to
# 0.88 of the mass in a window, 10 observations after the change), offline
# 19/20. The thresholds are one below.
SEEDS = range(20)
ONLINE_THRESHOLD = 16
OFFLINE_THRESHOLD = 18


@pytest.mark.behavior
def test_online_detects_mean_changes():
    hits = 0
    for seed in SEEDS:
        R, _ = online_changepoint_detection(
            mean_shift(seed).float(),
            partial(constant_hazard, 100, device="cpu"),
            online_likelihoods.NormalKnownVariance(
                variance=SIGMA**2, prior_variance=4.0, device="cpu"
            ),
            device="cpu",
        )
        probs = changepoint_probabilities(R, lag=10).clone()
        probs[0] = 0.0  # the prior, not a detection
        hits += detected(probs, [60, 120])
    assert hits >= ONLINE_THRESHOLD


@pytest.mark.behavior
def test_offline_detects_mean_changes():
    hits = 0
    for seed in SEEDS:
        data = mean_shift(seed)
        _, _, log_pcp = offline_changepoint_detection(
            data,
            partial(const_prior, p=1 / (len(data) + 1)),
            offline_likelihoods.NormalKnownVariance(
                device="cpu", variance=SIGMA**2, prior_variance=4.0
            ),
            device="cpu",
        )
        hits += detected(torch.exp(log_pcp).sum(0), [59, 119])
    assert hits >= OFFLINE_THRESHOLD


@pytest.mark.behavior
def test_streaming_with_a_bound():
    data = mean_shift(5, means=(0.0, 1.5), length=300).float()
    likelihood = online_likelihoods.NormalKnownVariance(
        variance=SIGMA**2, prior_variance=4.0, device="cpu"
    )
    detector = OnlineChangepointDetector(
        partial(constant_hazard, 200, device="cpu"), likelihood, max_run_length=100
    )
    map_start = {}
    for x in data:
        detector.update(x)
        map_start[detector.t] = detector.t - detector.map_run_length
    assert likelihood.mu.numel() == likelihood.var.numel() == 101
    assert abs(map_start[340] - 300) <= 3


@pytest.mark.behavior
@pytest.mark.parametrize("variance, prior_variance", [(0.0, 1.0), (1.0, -2.0)])
def test_variances_must_be_positive(variance, prior_variance):
    with pytest.raises(ValueError, match="positive"):
        online_likelihoods.NormalKnownVariance(
            variance=variance, prior_variance=prior_variance, device="cpu"
        )
    with pytest.raises(ValueError, match="positive"):
        offline_likelihoods.NormalKnownVariance(
            device="cpu", variance=variance, prior_variance=prior_variance
        )


@pytest.mark.behavior
def test_online_rejects_vector_observations():
    model = online_likelihoods.NormalKnownVariance(device="cpu")
    with pytest.raises(ValueError, match="scalar"):
        model.pdf(torch.randn(2))
