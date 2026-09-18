"""
Offline (Fearnhead 2006) recursion checked against an exhaustive reference.

``reference_offline_posterior`` in ``_reference_bocpd.py`` enumerates every
segmentation of a short series and sums their joint probabilities directly
from the model definition, using the library's own segment log-likelihoods
``P``. Agreement therefore tests the prior handling and the dynamic
programme, not the likelihoods (those are pinned elsewhere). Versions up to
1.0.x agreed only for ``const_prior``: the first changepoint row used
``g(length - 1)``, later rows paired ``g`` with the wrong length, and ``G``
included a length-0 term.
"""

from functools import partial

import numpy as np
import pytest
import torch

from bayesian_changepoint_detection import offline_changepoint_detection
from bayesian_changepoint_detection.offline_likelihoods import (
    IndependentFeaturesLikelihood,
    StudentT,
)
from bayesian_changepoint_detection.priors import (
    const_prior,
    geometric_prior,
    negative_binomial_prior,
)
from tests._reference_bocpd import reference_offline_posterior


def _priors(n):
    return {
        "const": partial(const_prior, p=1 / (n + 1)),
        "geometric": partial(geometric_prior, p=0.3),
        "negative_binomial": partial(negative_binomial_prior, k=2, p=0.4),
    }


def _series(n, seed, dims):
    torch.manual_seed(seed)
    if dims == 1:
        return torch.cat([torch.randn(n // 2), torch.randn(n - n // 2) + 3])
    shift = torch.tensor([0.0, 2.0]) * (torch.arange(n) >= n // 2).float().unsqueeze(1)
    return torch.randn(n, dims) + shift


@pytest.mark.parametrize("prior_name", ["const", "geometric", "negative_binomial"])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_exhaustive_enumeration_univariate(prior_name, n, seed):
    prior = _priors(n)[prior_name]
    data = _series(n, seed, 1)
    Q, P, Pcp = offline_changepoint_detection(
        data, prior, StudentT(device="cpu"), device="cpu"
    )
    log_q0, cp_ref, pcp_ref = reference_offline_posterior(
        P.numpy(), lambda length: float(prior(length))
    )
    assert abs(Q[0].item() - log_q0) < 1e-9
    if n > 1:
        cp = torch.exp(Pcp).sum(0).numpy()
        assert np.allclose(cp, cp_ref, atol=1e-9)
        finite = np.isfinite(pcp_ref)
        assert np.allclose(Pcp.numpy()[finite], pcp_ref[finite], atol=1e-9)
        assert np.all(np.isneginf(Pcp.numpy()[~finite]))


@pytest.mark.parametrize("prior_name", ["const", "geometric", "negative_binomial"])
@pytest.mark.parametrize("n", [2, 4, 7])
def test_matches_exhaustive_enumeration_multivariate(prior_name, n):
    prior = _priors(n)[prior_name]
    data = _series(n, seed=3, dims=2)
    Q, P, Pcp = offline_changepoint_detection(
        data,
        prior,
        IndependentFeaturesLikelihood(device="cpu"),
        device="cpu",
    )
    log_q0, cp_ref, _ = reference_offline_posterior(
        P.numpy(), lambda length: float(prior(length))
    )
    assert abs(Q[0].item() - log_q0) < 1e-9
    assert np.allclose(torch.exp(Pcp).sum(0).numpy(), cp_ref, atol=1e-9)


def test_geometric_prior_is_usable_end_to_end():
    """1.0.x raised ValueError here because the recursion evaluated the prior
    at length 0, which geometric_prior rejected."""
    torch.manual_seed(0)
    data = torch.cat([torch.randn(30), torch.randn(30) + 4])
    Q, P, Pcp = offline_changepoint_detection(
        data, partial(geometric_prior, p=1 / 30), StudentT(device="cpu"), device="cpu"
    )
    cp = torch.exp(Pcp).sum(0)
    assert torch.isfinite(Q).all()
    # data[29] = 2.3 sits between the two means, so the posterior may put
    # the boundary one point early; the constant prior does the same.
    assert abs(int(torch.argmax(cp)) - 29) <= 1


def test_legacy_truncation_is_off_by_default_and_was_wrong():
    """With multivariate likelihoods the legacy rule (cut the sum at the
    first term 40 nats below the running sum) fired inside the true segment
    and discarded the dominant term; on this 10-D series it returned
    changepoint "probabilities" around 1e31. The default is the exact sum."""
    from scipy.stats import multivariate_normal

    from bayesian_changepoint_detection.offline_likelihoods import MultivariateT

    np.random.seed(34)
    data = torch.tensor(
        np.vstack(
            [
                multivariate_normal.rvs([0] * 10, size=50),
                multivariate_normal.rvs([4] * 10, size=50),
                multivariate_normal.rvs([0] * 10, size=50),
                multivariate_normal.rvs([-4] * 10, size=50),
            ]
        )
    )
    prior = partial(const_prior, p=1 / (len(data) + 1))
    _, _, Pcp = offline_changepoint_detection(
        data, prior, MultivariateT(device="cpu"), device="cpu"
    )
    probs = torch.exp(Pcp).sum(0)
    assert probs.max() <= 1 + 1e-9
    assert torch.exp(Pcp[0]).sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.where(probs > 0.5)[0].tolist() == [49, 99, 149]

    # The bug, reproduced on request. How deep the dip gets depends on the
    # prior: with the pre-1.1.1 default Psi0 = I (dof0 times tighter than
    # the current unit-covariance default) the legacy cut discards the
    # dominant term on this series; with the current default it happens
    # not to on this series, which is luck, not safety.
    with pytest.warns(DeprecationWarning, match="truncate"):
        _, _, Pcp_legacy = offline_changepoint_detection(
            data,
            prior,
            MultivariateT(Psi0=torch.eye(10, dtype=torch.float64), device="cpu"),
            truncate=-40.0,
            device="cpu",
        )
    assert torch.exp(Pcp_legacy).sum(0).max() > 1e6


class TestEdgeCases:
    def _run(self, data, prior=None):
        prior = prior or partial(const_prior, p=1 / (max(len(data), 1) + 1))
        return offline_changepoint_detection(
            data, prior, StudentT(device="cpu"), device="cpu"
        )

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="at least one observation"):
            self._run(torch.zeros(0))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_raise(self, bad):
        with pytest.raises(ValueError, match="NaN or Inf"):
            self._run(torch.tensor([1.0, bad, 2.0, 3.0]))

    def test_single_point(self):
        Q, P, Pcp = self._run(torch.tensor([1.5]))
        assert Q.shape == (1,) and P.shape == (1, 1) and Pcp.shape == (0, 0)
        assert torch.isfinite(Q).all() and Q[0] == P[0, 0]

    def test_two_points(self):
        Q, P, Pcp = self._run(torch.tensor([0.0, 10.0]))
        assert Pcp.shape == (1, 1)
        cp = torch.exp(Pcp[0, 0]).item()
        assert 0.0 <= cp <= 1.0

    def test_constant_series_is_finite_and_quiet(self):
        Q, P, Pcp = self._run(torch.ones(25))
        assert torch.isfinite(Q).all()
        cp = torch.exp(Pcp).sum(0)
        assert torch.isfinite(cp).all() and cp.max() < 0.5

    def test_prior_mass_above_one_raises_instead_of_nan(self):
        """const_prior(p=0.25) on 20 points puts mass 4.75 on lengths 1..19;
        1.0.x returned Q = nan and all-zero changepoint probabilities."""
        with pytest.raises(ValueError, match="total mass"):
            self._run(torch.randn(20), partial(const_prior, p=0.25))

    def test_changepoint_probabilities_are_bounded(self):
        torch.manual_seed(4)
        data = torch.cat([torch.randn(15), torch.randn(15) + 5])
        _, _, Pcp = self._run(data, partial(geometric_prior, p=0.1))
        cp = torch.exp(Pcp).sum(0)
        assert (cp >= 0).all() and (cp <= 1 + 1e-9).all()
        assert int(torch.argmax(cp)) == 14
