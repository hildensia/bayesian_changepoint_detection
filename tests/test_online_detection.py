"""
Behavioural and reference tests for online (BOCPD) changepoint detection.

The run-length posterior ``R`` is checked against a small, independent NumPy
implementation of Adams & MacKay (2007) written directly from the paper's
recursion with ``scipy.stats.t`` predictive densities. The detection outputs
(``map_run_lengths``, ``get_map_changepoints``, ``changepoint_probabilities``)
are checked on synthetic series with known changepoints, including a series
with no changepoint at all.
"""

from functools import partial

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal, norm
from scipy.stats import t as student_t

from bayesian_changepoint_detection.bayesian_models import (
    changepoint_probabilities,
    get_map_changepoints,
    online_changepoint_detection,
)
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.online_likelihoods import MultivariateT, StudentT


def reference_bocpd(x, lam, alpha0, beta0, kappa0, mu0):
    """Adams & MacKay recursion in NumPy with Normal-Gamma StudentT predictive.

    Returns R with R[r, t] = P(run length r | x_0 .. x_{t-1}), same layout as
    the library. Written independently of the library code.
    """
    T = len(x)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    kappa = np.array([kappa0])
    mu = np.array([mu0])
    H = 1.0 / lam
    for t in range(T):
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = student_t.pdf(x[t], df, loc=mu, scale=scale)  # one per run length
        growth = R[: t + 1, t] * pred * (1 - H)
        cp = np.sum(R[: t + 1, t] * pred * H)
        R[1 : t + 2, t + 1] = growth
        R[0, t + 1] = cp
        R[:, t + 1] /= R[:, t + 1].sum()
        # posterior update, then prepend the prior for run length 0
        mu_n = (kappa * mu + x[t]) / (kappa + 1)
        kappa_n = kappa + 1
        alpha_n = alpha + 0.5
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_n])
        kappa = np.concatenate([[kappa0], kappa_n])
        alpha = np.concatenate([[alpha0], alpha_n])
        beta = np.concatenate([[beta0], beta_n])
    return R


@pytest.mark.parametrize("seed", [0, 1])
def test_run_length_posterior_matches_numpy_reference(seed):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 40)])
    expected = reference_bocpd(x, lam=50, alpha0=0.1, beta0=0.01, kappa0=1.0, mu0=0.0)

    R, map_run_lengths = online_changepoint_detection(
        torch.tensor(x, dtype=torch.float32),
        partial(constant_hazard, 50, device="cpu"),
        StudentT(0.1, 0.01, 1.0, 0.0, device="cpu"),
        device="cpu",
    )
    assert R.shape == expected.shape
    # float32 recursion over 80 steps vs float64 reference
    assert np.allclose(R.numpy(), expected, atol=2e-4)
    assert np.array_equal(map_run_lengths.numpy(), expected.argmax(axis=0))


def test_columns_are_normalized_and_run_length_zero_equals_hazard():
    """P(r_t = 0 | x_1..t) is the hazard under a constant hazard: a known
    property of the recursion, and the reason it is not a detection signal."""
    torch.manual_seed(0)
    data = torch.cat([torch.randn(50), torch.randn(50) + 5])
    R, _ = online_changepoint_detection(
        data,
        partial(constant_hazard, 40, device="cpu"),
        StudentT(0.1, 0.01, 1.0, 0.0, device="cpu"),
        device="cpu",
    )
    assert torch.allclose(R.sum(dim=0), torch.ones(R.shape[1]), atol=1e-5)
    assert torch.allclose(R[0, 1:], torch.full((R.shape[1] - 1,), 1 / 40), atol=1e-5)


def test_single_mean_shift_is_detected_at_the_right_place():
    torch.manual_seed(0)
    data = torch.cat([torch.randn(80), torch.randn(80) + 5])
    R, map_run_lengths = online_changepoint_detection(
        data,
        partial(constant_hazard, 100, device="cpu"),
        StudentT(0.1, 0.01, 1.0, 0.0, device="cpu"),
        device="cpu",
    )
    # MAP run length grows to 80 and resets to 1 right after the shift.
    assert map_run_lengths[80] == 80
    assert map_run_lengths[81] == 1
    assert get_map_changepoints(R).tolist() == [80]

    probs = changepoint_probabilities(R, lag=10)
    assert probs.shape == (len(data) + 1 - 10,)
    assert probs[80] > 0.85
    assert probs[1:80].max() < 0.1 and probs[81:].max() < 0.1


def test_no_changepoint_series_reports_nothing():
    torch.manual_seed(3)
    data = torch.randn(200)
    R, _ = online_changepoint_detection(
        data,
        partial(constant_hazard, 100, device="cpu"),
        StudentT(0.1, 0.01, 1.0, 0.0, device="cpu"),
        device="cpu",
    )
    assert get_map_changepoints(R).numel() == 0
    # Position 0 is trivially a segment start; everything after it must be
    # far below any sensible threshold.
    assert changepoint_probabilities(R, lag=10)[1:].max() < 0.2


def test_multiple_changes_with_min_separation():
    torch.manual_seed(0)
    data = torch.cat(
        [
            torch.randn(60),
            torch.randn(60) + 4,
            torch.randn(60),
            torch.randn(60) - 4,
        ]
    )
    R, _ = online_changepoint_detection(
        data,
        partial(constant_hazard, 50, device="cpu"),
        StudentT(0.1, 0.01, 1.0, 0.0, device="cpu"),
        device="cpu",
    )
    found = get_map_changepoints(R, min_separation=10).tolist()
    assert len(found) == 3
    for truth, got in zip([60, 120, 180], found):
        assert abs(got - truth) <= 3


def test_changepoint_probabilities_validates_lag():
    R = torch.eye(5)
    with pytest.raises(ValueError):
        changepoint_probabilities(R, lag=5)
    assert changepoint_probabilities(R, lag=0).shape == (5,)


def test_get_map_changepoints_threshold_is_deprecated():
    R = torch.eye(4)
    with pytest.warns(DeprecationWarning):
        get_map_changepoints(R, threshold=0.5)


def test_multivariate():
    """Ten-dimensional mean shifts at t=50, 100, 150 (from the original suite)."""
    np.random.seed(seed=34)
    dataset = np.vstack(
        (
            multivariate_normal.rvs([0] * 10, size=50),
            multivariate_normal.rvs([4] * 10, size=50),
            multivariate_normal.rvs([0] * 10, size=50),
            multivariate_normal.rvs([-4] * 10, size=50),
        )
    )
    R, map_run_lengths = online_changepoint_detection(
        dataset,
        partial(constant_hazard, 50, device="cpu"),
        MultivariateT(dims=10, device="cpu"),
        device="cpu",
    )
    found = get_map_changepoints(R, min_separation=10).tolist()
    assert len(found) == 3
    for truth, got in zip([50, 100, 150], found):
        assert abs(got - truth) <= 5


def test_univariate():
    """Mean shift at t=50; the MAP run length must drop sharply there
    (the assertion the original test.py made before the 1.0 migration)."""
    np.random.seed(seed=34)
    dataset = np.hstack((norm.rvs(0, size=50), norm.rvs(2, size=50)))
    R, maxes = online_changepoint_detection(
        dataset,
        partial(constant_hazard, 20, device="cpu"),
        StudentT(0.1, 0.01, 1, 0, device="cpu"),
        device="cpu",
    )
    assert maxes[50] - maxes[51] > 40


def test_input_validation():
    """Same contract as the offline detector and viterbi_changepoints: an
    empty series and non-finite values raise instead of yielding a 1x1
    posterior or a non-finite R with MAP run length 0 (1.1.0 behaviour)."""
    hazard = partial(constant_hazard, 50, device="cpu")
    with pytest.raises(ValueError, match="at least one observation"):
        online_changepoint_detection(
            torch.zeros(0), hazard, StudentT(device="cpu"), device="cpu"
        )
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="NaN or Inf"):
            online_changepoint_detection(
                torch.tensor([1.0, bad, 2.0]),
                hazard,
                StudentT(device="cpu"),
                device="cpu",
            )
    with pytest.raises(ValueError, match="NaN or Inf"):
        online_changepoint_detection(
            torch.tensor([[1.0, 2.0], [float("nan"), 0.0]]),
            hazard,
            MultivariateT(dims=2, device="cpu"),
            device="cpu",
        )


def test_accepts_lists_arrays_and_other_dtypes():
    """ensure_tensor coerces lists, NumPy arrays, float64 and integer input;
    the recursion runs in float32 either way."""
    hazard = partial(constant_hazard, 50, device="cpu")
    series = [0.1, 0.2, 5.0, 5.1, 5.2]
    outs = []
    for data in (series, np.array(series), torch.tensor(series, dtype=torch.float64)):
        R, m = online_changepoint_detection(
            data, hazard, StudentT(device="cpu"), device="cpu"
        )
        assert R.dtype == torch.float32 and R.shape == (6, 6)
        outs.append(R)
    assert torch.allclose(outs[0], outs[1]) and torch.allclose(
        outs[0], outs[2], atol=1e-6
    )
    R_int, _ = online_changepoint_detection(
        torch.tensor([1, 2, 3, 40, 41]), hazard, StudentT(device="cpu"), device="cpu"
    )
    assert torch.isfinite(R_int).all()
