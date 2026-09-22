"""
Consistency between the scalar ``pdf`` API and the vectorized ``pdf_rows``.

``pdf_rows(data, t)[j]`` must equal ``pdf(data, t, t + 1 + j)`` for every
likelihood, and the ``BaseLikelihood.pdf_rows`` fallback must produce the same
values for third-party subclasses that only implement ``pdf``.
"""

import pytest
import torch

from bayesian_changepoint_detection.offline_likelihoods import (
    BaseLikelihood,
    FullCovarianceLikelihood,
    IndependentFeaturesLikelihood,
    MultivariateT,
    StudentT,
)

pytestmark = pytest.mark.behavior


@pytest.fixture
def univariate_data():
    generator = torch.Generator().manual_seed(11)
    return torch.randn(40, generator=generator, dtype=torch.float64)


@pytest.fixture
def multivariate_data():
    generator = torch.Generator().manual_seed(12)
    return torch.randn(40, 3, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize("t", [0, 7, 38])
def test_studentt_rows_match_pdf(univariate_data, t):
    likelihood = StudentT(device="cpu")
    rows = likelihood.pdf_rows(univariate_data, t)
    n = univariate_data.shape[0]
    assert rows.shape == (n - t,)
    for j in range(n - t):
        assert rows[j].item() == pytest.approx(
            likelihood.pdf(univariate_data, t, t + 1 + j), abs=1e-9
        )


@pytest.mark.parametrize(
    "likelihood_cls",
    [IndependentFeaturesLikelihood, FullCovarianceLikelihood, MultivariateT],
)
@pytest.mark.parametrize("t", [0, 5, 20])
def test_multivariate_rows_match_pdf(multivariate_data, likelihood_cls, t):
    likelihood = likelihood_cls(device="cpu")
    rows = likelihood.pdf_rows(multivariate_data, t)
    n = multivariate_data.shape[0]
    assert rows.shape == (n - t,)
    for j in range(n - t):
        value = likelihood.pdf(multivariate_data, t, t + 1 + j)
        if torch.isfinite(rows[j]):
            assert rows[j].item() == pytest.approx(value, abs=1e-8)
        else:
            assert not torch.isfinite(torch.tensor(value))


def test_base_class_fallback_loops_over_pdf(univariate_data):
    class MeanLikelihood(BaseLikelihood):
        """Minimal third-party likelihood implementing only pdf."""

        def pdf(self, data, t, s):
            return -float(data[t:s].mean().abs()) * (s - t)

    likelihood = MeanLikelihood(device="cpu")
    rows = likelihood.pdf_rows(univariate_data, 3)
    n = univariate_data.shape[0]
    assert rows.shape == (n - 3,)
    for j in range(n - 3):
        assert rows[j].item() == pytest.approx(
            likelihood.pdf(univariate_data, 3, 4 + j), abs=1e-9
        )


def test_setup_recomputes_when_data_changes():
    generator = torch.Generator().manual_seed(13)
    first = torch.randn(30, generator=generator, dtype=torch.float64)
    second = torch.randn(30, generator=generator, dtype=torch.float64)

    likelihood = StudentT(device="cpu")
    value_first = likelihood.pdf(first, 0, 30)
    value_second = likelihood.pdf(second, 0, 30)
    assert value_first != value_second

    # Returning to the first dataset must give the original value again
    assert likelihood.pdf(first, 0, 30) == pytest.approx(value_first, abs=1e-12)


@pytest.mark.parametrize(
    "cls",
    [
        StudentT,
        IndependentFeaturesLikelihood,
        FullCovarianceLikelihood,
        MultivariateT,
    ],
)
def test_in_place_mutation_invalidates_cached_statistics(cls):
    """Mutating the same tensor in place must not return stale likelihoods."""
    torch.manual_seed(3)
    data = torch.randn(40, 2, dtype=torch.float64)
    model = cls(device="cpu")
    before = model.pdf(data, 5, 25)

    data[10:20] += 50.0  # same storage, same data_ptr, new contents
    after = model.pdf(data, 5, 25)

    fresh = cls(device="cpu").pdf(data.clone(), 5, 25)
    assert after != before
    assert after == pytest.approx(fresh, rel=1e-12)


@pytest.mark.parametrize(
    "cls",
    [
        IndependentFeaturesLikelihood,
        FullCovarianceLikelihood,
    ],
)
def test_univariate_input_has_finite_likelihoods(cls):
    """Length-one segments have zero variance; the prior floor must keep
    the log likelihood finite so Q does not become nan (regression)."""
    torch.manual_seed(0)
    data = torch.cat([torch.randn(30), torch.randn(30) + 4.0])
    model = cls(device="cpu")
    rows = model.pdf_rows(data, 5)
    assert torch.isfinite(rows).all()

    from functools import partial

    from bayesian_changepoint_detection import (
        const_prior,
        offline_changepoint_detection,
    )

    Q, _, Pcp = offline_changepoint_detection(
        data, partial(const_prior, p=1 / 61), model, device="cpu"
    )
    assert torch.isfinite(Q).all()
    cp = torch.exp(Pcp).sum(0)
    assert abs(int(cp.argmax()) - 29) <= 2


def test_float32_input_hits_the_statistics_cache():
    """Passing the same float32 tensor twice must not recompute statistics."""
    torch.manual_seed(1)
    data = torch.randn(50, 2, dtype=torch.float32)
    model = StudentT(device="cpu")
    calls = []
    original = model._compute_stats
    model._compute_stats = lambda d: (calls.append(1), original(d))
    model.pdf_rows(data, 0)
    model.pdf_rows(data, 3)
    assert len(calls) == 1


def test_detection_passes_caller_data_unchanged_to_third_party_pdf():
    """offline_changepoint_detection must not reshape or recast the data it
    hands to a likelihood that only implements the scalar pdf API."""
    from functools import partial

    from bayesian_changepoint_detection import (
        const_prior,
        offline_changepoint_detection,
    )

    seen = []

    class Recording(BaseLikelihood):
        def pdf(self, data, t, s):
            seen.append((tuple(data.shape), data.dtype))
            return -float(s - t)

    data = torch.randn(12, dtype=torch.float32)
    offline_changepoint_detection(
        data,
        partial(const_prior, p=1 / 13),
        Recording(device="cpu"),
        device="cpu",
    )
    assert seen and all(shape == (12,) for shape, _ in seen)
    assert all(dtype == torch.float32 for _, dtype in seen)


def test_offline_studentt_keeps_positional_device_argument():
    """StudentT('cpu') was valid before the prior hyperparameters existed."""
    model = StudentT("cpu")
    assert model.device == torch.device("cpu")
    assert model.alpha0 == 1.0
    with pytest.raises(TypeError):
        StudentT(0.1, 0.01, 1.0, 0.0)  # priors are keyword-only


def test_inference_mode_tensors_are_accepted():
    """Inference-mode tensors have no version counter; they must still work."""
    torch.manual_seed(0)
    model = StudentT(device="cpu")
    with torch.inference_mode():
        data = torch.randn(30, dtype=torch.float64)
        value = model.pdf(data, 3, 20)
    assert torch.isfinite(torch.tensor(value))


def test_statistics_cache_is_invalidated_when_model_device_changes():
    torch.manual_seed(0)
    data = torch.randn(30, dtype=torch.float64)
    model = StudentT(device="cpu")
    model.pdf_rows(data, 0)
    prepared_before = model._prepared
    model.device = torch.device("cpu")  # same device: cache must survive
    model.pdf_rows(data, 0)
    assert model._prepared is prepared_before
    # The key records the device the statistics were prepared on, so a
    # driver that moves the model (e.g. the MPS -> CPU fallback) misses it.
    assert model._stats_key[-1] == torch.device("cpu")


def test_impossible_entries_stay_minus_inf_in_changepoint_matrix():
    """-inf log probabilities must not be clamped to finite extrema."""
    from bayesian_changepoint_detection.bayesian_models import _nan_to_neg_inf

    x = torch.tensor([0.0, float("nan"), float("-inf"), float("inf")])
    y = _nan_to_neg_inf(x)
    assert y[0] == 0.0
    assert y[1] == float("-inf")
    assert y[2] == float("-inf")
    assert y[3] == float("inf")


def test_prepare_data_casts_before_moving():
    """A float64 CPU tensor given to an MPS model must be cast to float32
    on the CPU side first (MPS rejects float64 transfers)."""
    model = StudentT(device="cpu")
    prepared = model._prepare_data(torch.randn(10, dtype=torch.float32))
    assert prepared.dtype == torch.float64 and prepared.shape == (10, 1)
    if torch.backends.mps.is_available():
        model = StudentT(device="mps")
        prepared = model._prepare_data(torch.randn(10, dtype=torch.float64))
        assert prepared.dtype == torch.float32
        assert prepared.device.type == "mps"
