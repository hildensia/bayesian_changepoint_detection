"""
The input contract shared by the three detectors.

``online_changepoint_detection``, ``offline_changepoint_detection`` and
``viterbi_changepoints`` accept ``[T]`` or ``[T, D]`` data that is
non-empty, real and finite, and whose per-observation dimension matches the
likelihood's ``dims`` when it declares one. Anything else raises
``ValueError`` at the boundary. Before this contract some of these inputs
failed deep inside a likelihood (``IndexError`` for a 0-d tensor), and some
ran to completion on the wrong model: offline ``MultivariateT(dims=3)``
accepted transposed ``[3, T]`` data as three observations of dimension T.
"""

from functools import partial

import pytest
import torch

from bayesian_changepoint_detection import (
    const_prior,
    constant_hazard,
    offline_changepoint_detection,
    offline_likelihoods,
    online_changepoint_detection,
    online_likelihoods,
    viterbi_changepoints,
)

pytestmark = pytest.mark.behaviour

HAZARD = partial(constant_hazard, 50, device="cpu")
PRIOR = partial(const_prior, p=1 / 100)


def run_online(data, likelihood):
    return online_changepoint_detection(data, HAZARD, likelihood, device="cpu")


def run_viterbi(data, likelihood):
    return viterbi_changepoints(data, HAZARD, likelihood, device="cpu")


def run_offline(data, likelihood):
    return offline_changepoint_detection(data, PRIOR, likelihood, device="cpu")


# (detector, univariate likelihood factory, multivariate likelihood factory)
DETECTORS = {
    "online": (
        run_online,
        lambda: online_likelihoods.StudentT(device="cpu"),
        lambda d: online_likelihoods.MultivariateT(dims=d, device="cpu"),
    ),
    "viterbi": (
        run_viterbi,
        lambda: online_likelihoods.StudentT(device="cpu"),
        lambda d: online_likelihoods.MultivariateT(dims=d, device="cpu"),
    ),
    "offline": (
        run_offline,
        lambda: offline_likelihoods.StudentT(device="cpu"),
        lambda d: offline_likelihoods.MultivariateT(dims=d, device="cpu"),
    ),
}


@pytest.fixture(params=sorted(DETECTORS))
def detector(request):
    return DETECTORS[request.param]


@pytest.mark.parametrize(
    "data",
    [torch.tensor(1.0), torch.zeros(10, 2, 1)],
    ids=["0-d", "3-d"],
)
def test_data_must_be_one_or_two_dimensional(detector, data):
    run, univariate, _ = detector
    with pytest.raises(ValueError, match=r"shape \[T\] or \[T, D\]"):
        run(data, univariate())


def test_complex_data_is_rejected(detector):
    run, univariate, _ = detector
    data = torch.randn(20).to(torch.complex64)
    with pytest.raises(ValueError, match="must be real"):
        run(data, univariate())


def test_transposed_multivariate_data_is_reported(detector):
    run, _, multivariate = detector
    data = torch.randn(40, 3)
    with pytest.raises(ValueError, match=r"pass data\.T"):
        run(data.T, multivariate(3))


def test_dimension_mismatch_is_rejected(detector):
    run, _, multivariate = detector
    with pytest.raises(ValueError, match="expects 3-dimensional observations"):
        run(torch.randn(40), multivariate(3))
    with pytest.raises(ValueError, match="expects 3-dimensional observations"):
        run(torch.randn(40, 2), multivariate(3))


def test_the_callers_data_is_not_modified(detector):
    run, univariate, multivariate = detector
    torch.manual_seed(0)
    for data, likelihood in [
        (torch.randn(30), univariate()),
        (torch.randn(30, 2), multivariate(2)),
    ]:
        before = data.clone()
        run(data, likelihood)
        assert torch.equal(data, before)


def test_matching_dimensions_still_run(detector):
    run, univariate, multivariate = detector
    torch.manual_seed(1)
    run(torch.randn(25), univariate())
    run(torch.randn(25, 2), multivariate(2))


def test_univariate_series_fits_a_one_dimensional_multivariate_model(detector):
    # [T] data and dims=1: each observation is taken as a length-1 vector.
    run, _, multivariate = detector
    torch.manual_seed(3)
    run(torch.randn(25), multivariate(1))


class RecordingStudentT(online_likelihoods.StudentT):
    """Online StudentT that records every ``to`` call."""

    def __init__(self):
        super().__init__(device="cpu")
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return super().to(device)


@pytest.mark.parametrize("run", [run_online, run_viterbi], ids=["online", "viterbi"])
def test_a_rejected_input_does_not_move_the_model(run):
    # Validation comes before likelihood_model.to(device), so a caller's
    # model is not moved to another device by a call that then fails.
    likelihood = RecordingStudentT()
    with pytest.raises(ValueError):
        run(torch.tensor(1.0), likelihood)
    assert likelihood.moves == []
    run(torch.randn(5), likelihood)
    assert likelihood.moves  # the check above is not vacuous


class TestOfflineMultivariateTDims:
    def test_explicit_dims_that_disagree_with_the_data_raise_in_pdf(self):
        # The same check guards direct calls, which bypass the detector.
        likelihood = offline_likelihoods.MultivariateT(dims=3, device="cpu")
        with pytest.raises(ValueError, match=r"MultivariateT\(dims=3\)"):
            likelihood.pdf(torch.randn(20, 2), 0, 10)

    def test_dims_none_adapts_to_each_series(self):
        # dims=None must not be pinned to the first series' dimension: with
        # the explicit-dims check, a pinned value would reject the second.
        likelihood = offline_likelihoods.MultivariateT(device="cpu")
        torch.manual_seed(2)
        run_offline(torch.randn(20, 2), likelihood)
        run_offline(torch.randn(20, 4), likelihood)
        assert likelihood.dims is None
