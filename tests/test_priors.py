"""
Tests for prior probability distributions.
"""

import math

import numpy as np
import pytest
import torch
from scipy.stats import geom, nbinom

from bayesian_changepoint_detection.priors import (
    const_prior,
    geometric_prior,
    negative_binomial_prior,
)


class TestConstPrior:
    """Test constant prior function."""

    @pytest.mark.math
    def test_single_timepoint(self):
        """Test constant prior for single time point."""
        log_prob = const_prior(5, p=0.1)
        expected = np.log(0.1)
        assert abs(log_prob - expected) < 1e-6

    @pytest.mark.math
    def test_multiple_timepoints(self):
        """Test constant prior for multiple time points."""
        t = torch.arange(10)
        log_probs = const_prior(t, p=0.2)

        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == (10,)

        # All values should be the same
        expected = torch.log(torch.tensor(0.2))
        assert torch.allclose(log_probs, expected.expand_as(log_probs))

    @pytest.mark.behaviour
    def test_probability_validation(self):
        """Test probability parameter validation."""
        # Valid probabilities
        const_prior(1, p=0.1)
        const_prior(1, p=1.0)

        # Invalid probabilities
        with pytest.raises(ValueError):
            const_prior(1, p=0.0)

        with pytest.raises(ValueError):
            const_prior(1, p=1.1)

        with pytest.raises(ValueError):
            const_prior(1, p=-0.1)


class TestGeometricPrior:
    """geometric_prior(t, p) = (1 - p)^(t - 1) p for t >= 1 (trials to first success)."""

    @pytest.mark.math
    @pytest.mark.parametrize("p", [0.1, 0.25, 0.9])
    def test_closed_form(self, p):
        for t in range(1, 30):
            expected = (t - 1) * math.log1p(-p) + math.log(p)
            assert abs(geometric_prior(t, p=p) - expected) < 1e-5, t

    @pytest.mark.math
    def test_matches_scipy_geom(self):
        t = torch.arange(1, 40)
        got = geometric_prior(t, p=0.3, device="cpu").numpy()
        assert np.allclose(got, geom(0.3).logpmf(t.numpy()), atol=1e-5)

    @pytest.mark.math
    def test_is_a_probability_mass_function(self):
        probs = torch.exp(geometric_prior(torch.arange(1, 5000), p=0.05, device="cpu"))
        assert abs(probs.sum().item() - 1.0) < 1e-4
        assert probs[0] > probs[-1]

    @pytest.mark.math
    def test_p_equal_one_is_a_point_mass_at_length_one(self):
        assert geometric_prior(1, p=1.0) == 0.0
        assert geometric_prior(2, p=1.0) == float("-inf")

    @pytest.mark.math
    def test_impossible_lengths_are_log_zero(self):
        """Length 0 (or negative) has probability 0, i.e. log prob -inf, the
        same convention negative_binomial_prior uses for t < k. Versions
        1.0.x raised instead, which made the prior unusable with
        offline_changepoint_detection."""
        assert geometric_prior(0, p=0.1) == float("-inf")
        assert geometric_prior(-1, p=0.1) == float("-inf")
        out = geometric_prior(torch.tensor([0, 1, 2]), p=0.1, device="cpu")
        assert out[0] == float("-inf") and torch.isfinite(out[1:]).all()

    @pytest.mark.behaviour
    def test_probability_validation_geometric(self):
        with pytest.raises(ValueError):
            geometric_prior(1, p=0.0)
        with pytest.raises(ValueError):
            geometric_prior(1, p=1.1)


class TestNegativeBinomialPrior:
    """negative_binomial_prior(t, k, p) = C(t-1, k-1) p^k (1-p)^(t-k) for t >= k."""

    @pytest.mark.math
    @pytest.mark.parametrize("k,p", [(1, 0.25), (2, 0.25), (3, 0.5), (5, 0.9)])
    def test_closed_form(self, k, p):
        for t in range(k, 40):
            expected = (
                math.log(math.comb(t - 1, k - 1))
                + k * math.log(p)
                + (t - k) * math.log1p(-p)
            )
            assert abs(negative_binomial_prior(t, k=k, p=p) - expected) < 1e-5, t

    @pytest.mark.math
    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_matches_scipy_nbinom(self, k):
        t = torch.arange(k, 60)
        got = negative_binomial_prior(t, k=k, p=0.3, device="cpu").numpy()
        assert np.allclose(got, nbinom(k, 0.3).logpmf(t.numpy() - k), atol=1e-5)

    @pytest.mark.math
    def test_reduces_to_geometric_for_k_1(self):
        """Versions 1.0.x had p and 1 - p swapped, so this did not hold."""
        t = torch.arange(1, 30)
        nb = negative_binomial_prior(t, k=1, p=0.3, device="cpu")
        ge = geometric_prior(t, p=0.3, device="cpu")
        assert torch.allclose(nb, ge, atol=1e-5)

    @pytest.mark.math
    def test_is_a_probability_mass_function(self):
        probs = torch.exp(
            negative_binomial_prior(torch.arange(1, 5000), k=3, p=0.05, device="cpu")
        )
        assert abs(probs.sum().item() - 1.0) < 1e-4

    @pytest.mark.math
    def test_impossible_cases(self):
        assert negative_binomial_prior(1, k=2, p=0.1) == float("-inf")
        out = negative_binomial_prior(
            torch.tensor([1, 2, 3, 4]), k=3, p=0.1, device="cpu"
        )
        assert out[0] == float("-inf") and out[1] == float("-inf")
        assert torch.isfinite(out[2:]).all()

    @pytest.mark.math
    def test_p_equal_one(self):
        assert negative_binomial_prior(2, k=2, p=1.0) == 0.0
        assert negative_binomial_prior(3, k=2, p=1.0) == float("-inf")

    @pytest.mark.behaviour
    def test_output_device_and_dtype(self):
        out = negative_binomial_prior(torch.arange(1, 5), k=2, p=0.3, device="cpu")
        assert out.device.type == "cpu" and out.dtype == torch.float32

    @pytest.mark.behaviour
    def test_parameter_validation_nb(self):
        negative_binomial_prior(5, k=1, p=0.1)
        negative_binomial_prior(5, k=3, p=0.9)
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=0, p=0.1)
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=-1, p=0.1)
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=1, p=0.0)
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=1, p=1.1)


@pytest.mark.behaviour
class TestPriorDeviceHandling:
    """Test device handling for priors."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self):
        """Test that priors work correctly with different devices."""
        t_cpu = torch.arange(1, 6)
        t_cuda = t_cpu.cuda()

        # Test const_prior
        log_probs_cpu = const_prior(t_cpu, p=0.1, device="cpu")
        log_probs_cuda = const_prior(t_cuda, p=0.1, device="cuda")

        assert log_probs_cpu.device.type == "cpu"
        assert log_probs_cuda.device.type == "cuda"
        assert torch.allclose(log_probs_cpu, log_probs_cuda.cpu())

        # Test geometric_prior
        geom_cpu = geometric_prior(t_cpu, p=0.2, device="cpu")
        geom_cuda = geometric_prior(t_cuda, p=0.2, device="cuda")

        assert geom_cpu.device.type == "cpu"
        assert geom_cuda.device.type == "cuda"
        assert torch.allclose(geom_cpu, geom_cuda.cpu())

        # Test negative_binomial_prior
        nb_cpu = negative_binomial_prior(t_cpu, k=2, p=0.3, device="cpu")
        nb_cuda = negative_binomial_prior(t_cuda, k=2, p=0.3, device="cuda")

        assert nb_cpu.device.type == "cpu"
        assert nb_cuda.device.type == "cuda"
        assert torch.allclose(nb_cpu, nb_cuda.cpu(), equal_nan=True)
