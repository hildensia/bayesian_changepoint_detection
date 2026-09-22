"""
Tests for the Poisson (Gamma-Poisson) likelihoods, online and offline.

``math`` tests check the closed forms against scipy: the online predictive
against ``scipy.stats.nbinom`` after conjugate updates computed by hand, and
the offline segment marginal against the chain rule, i.e. the product of
one-step negative-binomial predictives, which shares no code with the
closed form. ``behavior`` tests pin detection on synthetic rate changes, the
streaming bound, and the input checks.
"""

from functools import partial

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

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


def rate_change(seed, rates=(2.0, 9.0, 2.0), length=60):
    gen = torch.Generator().manual_seed(seed)
    return torch.cat(
        [torch.poisson(torch.full((length,), rate), generator=gen) for rate in rates]
    )


def chain_rule_log_marginal(counts, alpha, beta):
    """log p(x_1..x_n) as a sum of one-step predictives (scipy nbinom)."""
    total = 0.0
    for x in counts:
        total += nbinom.logpmf(x, alpha, beta / (beta + 1.0))
        alpha, beta = alpha + x, beta + 1.0
    return total


@pytest.mark.math
def test_online_predictive_is_negative_binomial():
    counts = [3, 0, 7, 2, 5]
    alpha0, beta0 = 1.5, 0.4
    model = online_likelihoods.Poisson(alpha=alpha0, beta=beta0, device="cpu")
    for x in counts:
        model.update_theta(torch.tensor(float(x)))
    # After k updates, run length r (0..k) has seen the last r counts.
    x_new = 4
    log_probs = model.pdf(torch.tensor(float(x_new))).double().numpy()
    for r in range(len(counts) + 1):
        seen = counts[len(counts) - r :]
        alpha, beta = alpha0 + sum(seen), beta0 + r
        expected = nbinom.logpmf(x_new, alpha, beta / (beta + 1.0))
        assert log_probs[r] == pytest.approx(expected, abs=1e-5)


@pytest.mark.math
@pytest.mark.parametrize("alpha0, beta0", [(1.0, 1.0), (2.5, 0.1), (0.5, 3.0)])
def test_offline_marginal_matches_the_chain_rule(alpha0, beta0):
    counts = rate_change(0, rates=(3.0, 12.0), length=15).double()
    model = offline_likelihoods.Poisson(device="cpu", alpha0=alpha0, beta0=beta0)
    array = counts.numpy()
    for t, s in [(0, 1), (0, 15), (5, 25), (14, 30)]:
        expected = chain_rule_log_marginal(array[t:s], alpha0, beta0)
        assert model.pdf(counts, t, s) == pytest.approx(expected, abs=1e-9)
    rows = model.pdf_rows(counts, 3).numpy()
    for j, value in enumerate(rows):
        expected = chain_rule_log_marginal(array[3 : 3 + 1 + j], alpha0, beta0)
        assert value == pytest.approx(expected, abs=1e-9)


@pytest.mark.math
def test_offline_multivariate_sums_independent_dimensions():
    gen = torch.Generator().manual_seed(1)
    counts = torch.poisson(torch.full((20, 2), 5.0), generator=gen).double()
    model = offline_likelihoods.Poisson(device="cpu", alpha0=2.0, beta0=0.5)
    expected = sum(
        chain_rule_log_marginal(counts[4:17, d].numpy(), 2.0, 0.5) for d in range(2)
    )
    assert model.pdf(counts, 4, 17) == pytest.approx(expected, abs=1e-9)


def found_exactly(found, true, tolerance=3):
    return len(found) == len(true) and all(
        abs(a - b) <= tolerance for a, b in zip(found, true)
    )


# Detection is checked as a rate over 20 draws rather than on one lucky
# seed: rate 2 -> 9 -> 2, 60 points per segment. Measured when written:
# online (lag-10 probability > 0.5) 19/20, offline 20/20. The MAP-path rule
# (get_map_changepoints) is not used: it adds a spurious start on about a
# quarter of the draws, the known twitchiness of that rule.
SEEDS = range(20)


@pytest.mark.behavior
def test_online_detects_rate_changes():
    hits = 0
    for seed in SEEDS:
        R, _ = online_changepoint_detection(
            rate_change(seed),
            partial(constant_hazard, 100, device="cpu"),
            online_likelihoods.Poisson(alpha=1.0, beta=0.1, device="cpu"),
            device="cpu",
        )
        probs = changepoint_probabilities(R, lag=10)
        found = (torch.where(probs[1:] > 0.5)[0] + 1).tolist()
        hits += found_exactly(found, [60, 120])
    assert hits >= 18


@pytest.mark.behavior
def test_offline_detects_rate_changes():
    hits = 0
    for seed in SEEDS:
        counts = rate_change(seed)
        _, _, log_pcp = offline_changepoint_detection(
            counts,
            partial(const_prior, p=1 / (len(counts) + 1)),
            offline_likelihoods.Poisson(device="cpu", alpha0=1.0, beta0=0.1),
            device="cpu",
        )
        found = torch.where(torch.exp(log_pcp).sum(0) > 0.5)[0].tolist()
        hits += found_exactly(found, [59, 119])
    assert hits >= 19


@pytest.mark.behavior
def test_streaming_with_a_bound():
    counts = rate_change(4, rates=(2.0, 9.0), length=300)
    detector = OnlineChangepointDetector(
        partial(constant_hazard, 200, device="cpu"),
        online_likelihoods.Poisson(alpha=1.0, beta=0.1, device="cpu"),
        max_run_length=100,
    )
    map_start = {}
    for x in counts:
        detector.update(x)
        map_start[detector.t] = detector.t - detector.map_run_length
    assert detector.likelihood_model.alpha.numel() == 101
    # 40 points after the change (run length below the bound) the MAP
    # segment starts at the change. The lag-10 probability is not used: on
    # this draw its mass splits between starts 300 and 301.
    assert abs(map_start[340] - 300) <= 3


@pytest.mark.behavior
@pytest.mark.parametrize("bad", [-1.0, 2.5])
def test_non_counts_are_rejected(bad):
    online = online_likelihoods.Poisson(device="cpu")
    with pytest.raises(ValueError, match="non-negative integer"):
        online.pdf(torch.tensor(bad))
    offline = offline_likelihoods.Poisson(device="cpu")
    data = torch.tensor([1.0, 3.0, bad, 2.0])
    with pytest.raises(ValueError, match="non-negative integer"):
        offline.pdf(data, 0, 4)


@pytest.mark.behavior
def test_integer_tensors_are_accepted():
    counts = torch.tensor([1, 0, 4, 2, 3])
    offline = offline_likelihoods.Poisson(device="cpu")
    assert np.isfinite(offline.pdf(counts, 0, 5))
    online = online_likelihoods.Poisson(device="cpu")
    assert torch.isfinite(online.pdf(counts[0])).all()


@pytest.mark.behavior
@pytest.mark.parametrize("alpha, beta", [(0.0, 1.0), (1.0, -1.0)])
def test_priors_must_be_positive(alpha, beta):
    with pytest.raises(ValueError, match="positive"):
        online_likelihoods.Poisson(alpha=alpha, beta=beta, device="cpu")
    with pytest.raises(ValueError, match="positive"):
        offline_likelihoods.Poisson(device="cpu", alpha0=alpha, beta0=beta)
