"""
Tests for ``OnlineChangepointDetector``, the incremental online detector.

The ``math`` tests compare it with ``reference_mv_bocpd`` in
``tests/_reference_bocpd.py``, a NumPy/scipy implementation written from
Adams & MacKay (2007) and Murphy (2007) that shares no code with the
library, with and without the run-length bound. The ``behaviour`` tests pin
its agreement with ``online_changepoint_detection`` (the same recursion
through a second code path), the bound, and the input checks.
"""

from functools import partial

import numpy as np
import pytest
import torch

from bayesian_changepoint_detection import (
    OnlineChangepointDetector,
    changepoint_probabilities,
    constant_hazard,
    online_changepoint_detection,
)
from bayesian_changepoint_detection.online_likelihoods import (
    BaseLikelihood,
    MultivariateT,
    StudentT,
)
from tests._reference_bocpd import reference_mv_bocpd

HAZARD = partial(constant_hazard, 40, device="cpu")


def mean_shift(seed, n=60, shift=3.0, dims=None):
    gen = torch.Generator().manual_seed(seed)
    shape = (n,) if dims is None else (n, dims)
    first = torch.randn(shape, generator=gen)
    second = torch.randn(shape, generator=gen) + shift
    return torch.cat([first, second])


def stream(detector, data):
    """Feed ``data`` and collect the posterior after every observation."""
    return [detector.update(x) for x in data]


def reference(X, max_run_length=None):
    dims = X.shape[1]
    return reference_mv_bocpd(
        X.numpy().astype(np.float64),
        lam=40,
        dof0=dims + 1,
        kappa0=1.0,
        mu0=np.zeros(dims),
        W0=np.eye(dims) / (dims + 1),  # the library default
        max_run_length=max_run_length,
    )


@pytest.mark.math
def test_matches_the_numpy_reference():
    X = mean_shift(0, n=30, dims=2)
    expected = reference(X)
    detector = OnlineChangepointDetector(HAZARD, MultivariateT(dims=2, device="cpu"))
    for t, posterior in enumerate(stream(detector, X)):
        column = expected[: t + 2, t + 1]
        assert np.abs(posterior.numpy() - column).max() < 1e-5


@pytest.mark.math
@pytest.mark.parametrize("max_run_length", [1, 5, 20])
def test_bounded_run_length_matches_the_truncated_reference(max_run_length):
    X = mean_shift(1, n=30, dims=2)
    expected = reference(X, max_run_length=max_run_length)
    detector = OnlineChangepointDetector(
        HAZARD, MultivariateT(dims=2, device="cpu"), max_run_length=max_run_length
    )
    for t, posterior in enumerate(stream(detector, X)):
        keep = min(t + 2, max_run_length + 1)
        assert posterior.numel() == keep
        assert np.abs(posterior.numpy() - expected[:keep, t + 1]).max() < 1e-5
        assert np.all(expected[keep:, t + 1] == 0.0)


@pytest.mark.behaviour
@pytest.mark.parametrize("dims", [None, 3], ids=["univariate", "multivariate"])
def test_agrees_with_the_batch_detector(dims):
    # Same recursion; the batch version normalizes over the zero-padded
    # column of R, so the two differ by float32 rounding only.
    data = mean_shift(2, dims=dims)
    make = (
        (lambda: StudentT(device="cpu"))
        if dims is None
        else (lambda: MultivariateT(dims=dims, device="cpu"))
    )
    R, map_run_lengths = online_changepoint_detection(
        data, HAZARD, make(), device="cpu"
    )
    detector = OnlineChangepointDetector(HAZARD, make())
    for t, x in enumerate(data):
        posterior = detector.update(x)
        assert torch.allclose(posterior, R[: t + 2, t + 1], rtol=1e-5, atol=1e-6)
        assert detector.map_run_length == int(map_run_lengths[t + 1])
    assert detector.t == len(data)
    lagged = changepoint_probabilities(R, lag=10)
    assert detector.changepoint_probability(10) == pytest.approx(
        float(lagged[len(data) - 10]), abs=1e-6
    )


@pytest.mark.behaviour
def test_a_bound_above_the_stream_length_changes_nothing():
    data = mean_shift(3)
    exact = OnlineChangepointDetector(HAZARD, StudentT(device="cpu"))
    bounded = OnlineChangepointDetector(
        HAZARD, StudentT(device="cpu"), max_run_length=len(data)
    )
    for x in data:
        assert torch.equal(exact.update(x), bounded.update(x))


@pytest.mark.behaviour
def test_bounded_memory_on_a_long_stream():
    # 2 000 observations, bound 50: the posterior and every parameter vector
    # stay at 51 entries, and the change at 1 000 is still found.
    gen = torch.Generator().manual_seed(4)
    data = torch.cat(
        [torch.randn(1000, generator=gen), torch.randn(1000, generator=gen) + 3]
    )
    likelihood = StudentT(device="cpu")
    detector = OnlineChangepointDetector(
        partial(constant_hazard, 500, device="cpu"), likelihood, max_run_length=50
    )
    starts = set()
    for x in data:
        posterior = detector.update(x)
        assert posterior.numel() <= 51
        assert posterior.sum().item() == pytest.approx(1.0, abs=1e-5)
        if detector.t > 10 and detector.changepoint_probability(10) > 0.5:
            starts.add(detector.t - 10)
    assert {
        name: getattr(likelihood, name).shape[0]
        for name in likelihood._run_length_state
    } == dict.fromkeys(likelihood._run_length_state, 51)
    assert starts == {1000}


@pytest.mark.behaviour
def test_rejected_observations_leave_the_state_alone():
    detector = OnlineChangepointDetector(HAZARD, StudentT(device="cpu"))
    for x in [0.1, -0.3, 0.2]:
        detector.update(x)
    before = detector.run_length_posterior
    bad = [float("nan"), float("inf"), torch.randn(2), torch.tensor(1 + 1j)]
    for x in bad:
        with pytest.raises(ValueError):
            detector.update(x)
    assert detector.t == 3
    assert torch.equal(detector.run_length_posterior, before)


@pytest.mark.behaviour
def test_multivariate_observations_must_have_the_model_dimension():
    detector = OnlineChangepointDetector(HAZARD, MultivariateT(dims=3, device="cpu"))
    with pytest.raises(ValueError, match=r"shape \[3\]"):
        detector.update(torch.randn(2))
    detector.update(torch.randn(3))
    one_dim = OnlineChangepointDetector(HAZARD, MultivariateT(dims=1, device="cpu"))
    one_dim.update(0.5)  # a scalar is taken as a length-1 vector
    assert one_dim.t == 1


@pytest.mark.behaviour
def test_changepoint_probability_range():
    detector = OnlineChangepointDetector(
        HAZARD, StudentT(device="cpu"), max_run_length=5
    )
    with pytest.raises(ValueError, match="lag"):
        detector.changepoint_probability(1)  # nothing seen yet
    for x in range(10):
        detector.update(float(x))
    assert 0.0 <= detector.changepoint_probability(5) <= 1.0
    with pytest.raises(ValueError, match="lag"):
        detector.changepoint_probability(6)  # beyond the bound
    with pytest.raises(ValueError, match="lag"):
        detector.changepoint_probability(-1)


@pytest.mark.behaviour
@pytest.mark.parametrize(
    "bound, error",
    [(0, ValueError), (-3, ValueError), (2.5, TypeError), (True, TypeError)],
)
def test_invalid_bounds_are_rejected(bound, error):
    with pytest.raises(error):
        OnlineChangepointDetector(HAZARD, StudentT(device="cpu"), max_run_length=bound)


class PdfOnlyLikelihood(BaseLikelihood):
    """A third-party likelihood with no ``_run_length_state``."""

    def pdf(self, data):
        return torch.zeros(self.t + 1)

    def update_theta(self, data, **kwargs):
        self.t += 1


@pytest.mark.behaviour
def test_a_bound_needs_a_prunable_likelihood():
    with pytest.raises(ValueError, match="_run_length_state"):
        OnlineChangepointDetector(
            HAZARD, PdfOnlyLikelihood(device="cpu"), max_run_length=5
        )
    # Without a bound it works as before.
    detector = OnlineChangepointDetector(HAZARD, PdfOnlyLikelihood(device="cpu"))
    detector.update(1.0)
    detector.update(2.0)
    assert detector.t == 2
    with pytest.raises(NotImplementedError):
        PdfOnlyLikelihood(device="cpu").prune(1)
