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


@pytest.mark.parametrize("cls", [
    StudentT,
    IndependentFeaturesLikelihood,
    FullCovarianceLikelihood,
    MultivariateT,
])
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
