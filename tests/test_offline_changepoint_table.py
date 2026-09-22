"""
The offline changepoint table ``Pcp`` against a direct evaluation of its
recursion (Fearnhead 2006), written independently in NumPy/SciPy with one
explicit sum per entry, from the ``P`` and ``Q`` the detector returns.

The detector builds the table from a precomputed matrix and stops once a
row's total probability falls below exp(-1000). These tests pin that both
shortcuts leave the table unchanged: finite entries agree to rounding and
every skipped entry is one the direct evaluation puts below exp(-1000),
i.e. exactly 0 in probability space. (The recursion itself is checked
against an exhaustive enumeration of segmentations in
``test_offline_prior_recursion.py``, on series too short to reach the
cutoff.)
"""

from functools import partial

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from bayesian_changepoint_detection import (
    const_prior,
    geometric_prior,
    offline_changepoint_detection,
)
from bayesian_changepoint_detection.offline_likelihoods import StudentT

pytestmark = pytest.mark.behavior


def _direct_pcp(P, Q, log_g):
    """``Pcp`` entry by entry; ``log_g[l]`` is the log prior of length ``l``."""
    n = P.shape[0]
    pcp = np.full((n - 1, n - 1), -np.inf)
    for t in range(n - 1):
        pcp[0, t] = P[0, t] + Q[t + 1] + log_g[t + 1] - Q[0]
    for j in range(1, n - 1):
        for t in range(j, n - 1):
            s = np.arange(j - 1, t)
            terms = pcp[j - 1, s] + P[s + 1, t] + Q[t + 1] + log_g[t - s] - Q[s + 1]
            pcp[j, t] = logsumexp(terms)
    return pcp


def _series(n, seed):
    generator = torch.Generator().manual_seed(seed)
    means = torch.tensor([0.0, 3.0, -1.0, 2.0]).repeat_interleave(n // 4)
    return means + torch.randn(n, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize(
    "prior",
    [
        pytest.param(lambda n: partial(const_prior, p=1.0 / (n + 1)), id="const"),
        pytest.param(lambda n: partial(geometric_prior, p=0.02), id="geometric"),
    ],
)
def test_table_matches_direct_recursion_past_the_cutoff(prior):
    n = 240
    data = _series(n, seed=3)
    prior_function = prior(n)
    Q, P, Pcp = offline_changepoint_detection(
        data, prior_function, StudentT(device="cpu"), device="cpu"
    )
    log_g = np.full(n + 1, -np.inf)
    log_g[1:] = [float(prior_function(length)) for length in range(1, n + 1)]
    expected = _direct_pcp(P.numpy(), Q.numpy(), log_g)
    got = Pcp.numpy()

    # The cutoff fired: some rows were skipped.
    skipped = np.isneginf(got) & np.isfinite(expected)
    assert skipped.any()
    # Every skipped entry is below exp(-1000) and so 0 in float64 anyway.
    assert expected[skipped].max() < -1000
    np.testing.assert_array_equal(np.exp(got[skipped]), np.exp(expected[skipped]))
    # Everything computed agrees with the direct recursion to rounding.
    computed = np.isfinite(got)
    assert np.array_equal(computed, np.isfinite(expected) & ~skipped)
    np.testing.assert_allclose(got[computed], expected[computed], rtol=1e-12, atol=1e-9)
    # And the marginal changepoint probabilities are identical to rounding.
    np.testing.assert_allclose(
        np.exp(got).sum(0), np.exp(expected).sum(0), rtol=0, atol=1e-12
    )


def test_short_series_computes_every_row():
    # Twelve points cannot reach exp(-1000): no row is skipped.
    data = _series(12, seed=0)
    n = len(data)
    prior_function = partial(const_prior, p=1.0 / (n + 1))
    Q, P, Pcp = offline_changepoint_detection(
        data, prior_function, StudentT(device="cpu"), device="cpu"
    )
    log_g = np.full(n + 1, -np.inf)
    log_g[1:] = [float(prior_function(length)) for length in range(1, n + 1)]
    expected = _direct_pcp(P.numpy(), Q.numpy(), log_g)
    np.testing.assert_allclose(Pcp.numpy(), expected, rtol=1e-12, atol=1e-12)
