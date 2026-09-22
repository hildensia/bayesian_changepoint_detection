"""
Tests for ``negative_binomial_hazard``.

``math``: the hazard equals ``pmf(r + 1) / P(length >= r + 1)`` computed
with ``scipy.stats.nbinom`` (length - k is scipy's number of failures), and
it is the conditional probability that ``negative_binomial_prior`` implies.
``behavior``: the k = 1 reduction, shapes, devices, input checks, and use
with the online detector.
"""

from functools import partial

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

from bayesian_changepoint_detection import (
    StudentT,
    changepoint_probabilities,
    constant_hazard,
    negative_binomial_hazard,
    negative_binomial_prior,
    online_changepoint_detection,
)


def scipy_hazard(k, p, r):
    """H(r) = P(L = r + 1) / P(L >= r + 1) with L - k ~ nbinom(k, p)."""
    if r + 1 < k:
        return 0.0
    survival = nbinom.sf(r - k, k, p)  # P(L - k > r - k) = P(L >= r + 1)
    return nbinom.pmf(r + 1 - k, k, p) / survival if survival > 0 else 1.0


@pytest.mark.math
@pytest.mark.parametrize("k, p", [(1, 0.1), (3, 0.05), (5, 0.3), (2, 1.0), (10, 0.02)])
def test_matches_scipy(k, p):
    r = torch.arange(300)
    hazard = negative_binomial_hazard(k, p, r, device="cpu").double().numpy()
    expected = np.array([scipy_hazard(k, p, int(v)) for v in r])
    np.testing.assert_allclose(hazard, expected, rtol=1e-6, atol=1e-9)


@pytest.mark.math
def test_is_the_conditional_probability_of_the_offline_prior():
    # H(r) = g(r + 1) / (1 - sum_{l <= r} g(l)) with g the offline prior.
    k, p = 4, 0.1
    lengths = torch.arange(1, 400)
    g = np.exp(
        negative_binomial_prior(lengths, k=k, p=p, device="cpu").double().numpy()
    )
    survival = 1.0 - np.concatenate([[0.0], np.cumsum(g)])  # P(L >= l), l = 1..
    r = np.arange(60)
    expected = g[r] / survival[r]
    hazard = negative_binomial_hazard(k, p, torch.as_tensor(r), device="cpu")
    np.testing.assert_allclose(hazard.double().numpy(), expected, rtol=1e-5, atol=1e-9)


@pytest.mark.behavior
def test_k_one_is_the_constant_hazard():
    r = torch.arange(50)
    assert torch.allclose(
        negative_binomial_hazard(1, 0.04, r, device="cpu"),
        constant_hazard(25.0, r, device="cpu"),
    )


@pytest.mark.behavior
def test_shapes_dtype_and_device():
    assert negative_binomial_hazard(2, 0.5, 7, device="cpu").shape == (7,)
    r = torch.arange(12).reshape(3, 4)
    hazard = negative_binomial_hazard(2, 0.5, r)
    assert hazard.shape == (3, 4)
    assert hazard.dtype == torch.float32
    assert hazard.device == r.device
    assert negative_binomial_hazard(2, 0.5, torch.arange(0)).numel() == 0


@pytest.mark.behavior
def test_long_runs_approach_p():
    hazard = negative_binomial_hazard(3, 0.01, torch.tensor([20000]), device="cpu")
    assert hazard.item() == pytest.approx(0.01, rel=0.02)


@pytest.mark.behavior
@pytest.mark.parametrize(
    "k, p", [(0, 0.5), (-1, 0.5), (1.5, 0.5), (True, 0.5), (1, 0.0), (1, 1.5)]
)
def test_invalid_parameters(k, p):
    with pytest.raises(ValueError):
        negative_binomial_hazard(k, p, 5, device="cpu")


@pytest.mark.behavior
def test_negative_run_lengths_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        negative_binomial_hazard(2, 0.5, torch.tensor([1, -1]), device="cpu")


@pytest.mark.behavior
def test_drives_the_online_detector():
    # Rate over 20 draws (mean shift 0 -> 3 at 80, unit noise): at least 0.9
    # of the lag-10 probability within +-3 of the change and no other
    # position above 0.5. Measured when written: 19/20, the same draws as
    # constant_hazard(150) (the miss holds 0.79 in the window).
    hits = 0
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        data = torch.cat(
            [torch.randn(80, generator=gen), torch.randn(80, generator=gen) + 3]
        )
        R, _ = online_changepoint_detection(
            data,
            partial(negative_binomial_hazard, 3, 0.02, device="cpu"),  # mean 150
            StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cpu"),
            device="cpu",
        )
        probs = changepoint_probabilities(R, lag=10).clone()
        probs[0] = 0.0  # the prior, not a detection
        window = probs[77:84]
        outside = torch.cat([probs[:77], probs[84:]])
        hits += bool(window.sum() >= 0.9) and bool(outside.max() < 0.5)
    assert hits >= 18
