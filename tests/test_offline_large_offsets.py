"""
Offline marginals stay exact for data far from zero (issue #55).

Each offline likelihood is compared with its closed form evaluated on exact
rational sufficient statistics (``tests/_exact_marginals.py``) at data
offsets up to 1e10, with the prior mean at 0 and at the data. Before the
prefix sums were centered, the errors at an offset of 1e8 were 44 nats
(StudentT, prior mean at the data), 17 (MultivariateT), 16
(FullCovariance) and 1.6 (IndependentFeatures).
"""

import pytest
import torch

from bayesian_changepoint_detection import offline_likelihoods as off
from tests import _exact_marginals as exact

pytestmark = pytest.mark.math

OFFSETS = [1e4, 1e6, 1e8, 1e10]
SEGMENTS = [(0, 60), (10, 40), (55, 57)]


def series(offset, dims=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    shape = (60,) if dims is None else (60, dims)
    return torch.randn(shape, generator=gen, dtype=torch.float64) + offset


def assert_close(value, expected):
    # Relative to the size of the value: float64 cannot do better when the
    # marginal itself is ~1e17 (prior mean at 0, data at 1e10).
    assert abs(value - expected) <= 1e-12 * max(1.0, abs(expected))


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("prior_at_data", [False, True], ids=["mu0=0", "mu0=data"])
def test_student_t(offset, prior_at_data):
    data = series(offset)
    mu0 = offset if prior_at_data else 0.0
    model = off.StudentT(device="cpu", mu0=mu0)
    for t, s in SEGMENTS:
        expected = exact.student_t(data[t:s].tolist(), 1.0, 1.0, 1.0, mu0)
        assert_close(model.pdf(data, t, s), expected)
        assert_close(model.pdf_rows(data, t)[s - t - 1].item(), expected)


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("prior_at_data", [False, True], ids=["mu0=0", "mu0=data"])
def test_multivariate_t(offset, prior_at_data):
    data = series(offset, dims=2, seed=1)
    mu0 = [offset, offset] if prior_at_data else [0.0, 0.0]
    model = off.MultivariateT(device="cpu", mu0=torch.tensor(mu0, dtype=torch.float64))
    psi0 = [[3.0, 0.0], [0.0, 3.0]]  # the default: dof0 * I with dof0 = d + 1
    for t, s in SEGMENTS:
        expected = exact.multivariate_t(data[t:s].tolist(), 1.0, 3.0, mu0, psi0)
        assert_close(model.pdf(data, t, s), expected)


@pytest.mark.parametrize("offset", OFFSETS)
def test_independent_features(offset):
    data = series(offset, dims=3, seed=2)
    model = off.IndependentFeaturesLikelihood(device="cpu")
    for t, s in SEGMENTS:
        assert_close(
            model.pdf(data, t, s), exact.independent_features(data[t:s].tolist())
        )


@pytest.mark.parametrize("offset", OFFSETS)
def test_full_covariance(offset):
    data = series(offset, dims=2, seed=3)
    model = off.FullCovarianceLikelihood(device="cpu")
    for t, s in SEGMENTS:
        assert_close(model.pdf(data, t, s), exact.full_covariance(data[t:s].tolist()))


def test_per_dimension_offsets():
    # Dimensions at very different locations: the flattened prior variance
    # is dominated by the spread of the dimension means.
    gen = torch.Generator().manual_seed(4)
    data = torch.randn(40, 3, generator=gen, dtype=torch.float64)
    data += torch.tensor([1e8, -1e8, 5.0], dtype=torch.float64)
    for model, reference in [
        (off.IndependentFeaturesLikelihood(device="cpu"), exact.independent_features),
        (off.FullCovarianceLikelihood(device="cpu"), exact.full_covariance),
    ]:
        for t, s in [(0, 40), (5, 25)]:
            assert_close(model.pdf(data, t, s), reference(data[t:s].tolist()))


def test_small_offsets_agree_with_the_reference_too():
    # Guards the reference itself: at offset 0 it must match the library,
    # whose formulas are proven against scipy elsewhere.
    data = series(0.0, dims=2, seed=5)
    model = off.MultivariateT(device="cpu")
    psi0 = [[3.0, 0.0], [0.0, 3.0]]
    expected = exact.multivariate_t(data[3:30].tolist(), 1.0, 3.0, [0.0, 0.0], psi0)
    assert_close(model.pdf(data, 3, 30), expected)
