"""
Tests for online likelihood functions.
"""

import pytest
import torch

from bayesian_changepoint_detection.online_likelihoods import MultivariateT, StudentT

pytestmark = pytest.mark.behavior


class TestStudentT:
    """Test univariate Student's t-distribution likelihood."""

    def test_initialization(self):
        """Test StudentT initialization."""
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)

        assert likelihood.alpha0 == 0.1
        assert likelihood.beta0 == 0.01
        assert likelihood.kappa0 == 1
        assert likelihood.mu0 == 0
        assert likelihood.t == 0

    def test_pdf_single_observation(self):
        """Test PDF computation for single observation."""
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        data = torch.tensor(1.0)

        log_probs = likelihood.pdf(data)

        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == (1,)
        assert torch.isfinite(log_probs).all()
        assert likelihood.t == 1

    def test_pdf_multiple_observations(self):
        """Test PDF computation for multiple observations."""
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)

        # First observation
        data1 = torch.tensor(1.0)
        log_probs1 = likelihood.pdf(data1)
        likelihood.update_theta(data1)

        # Second observation
        data2 = torch.tensor(2.0)
        log_probs2 = likelihood.pdf(data2)

        assert log_probs1.shape == (1,)
        assert log_probs2.shape == (2,)
        assert likelihood.t == 2

    def test_update_theta(self):
        """Test parameter updates."""
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        data = torch.tensor(1.0)

        # Store initial parameters
        initial_alpha = likelihood.alpha.clone()
        initial_beta = likelihood.beta.clone()
        initial_kappa = likelihood.kappa.clone()
        initial_mu = likelihood.mu.clone()

        likelihood.update_theta(data)

        # Parameters should have grown
        assert likelihood.alpha.shape[0] == 2
        assert likelihood.beta.shape[0] == 2
        assert likelihood.kappa.shape[0] == 2
        assert likelihood.mu.shape[0] == 2

        # First elements should be initial parameters
        assert torch.allclose(likelihood.alpha[0:1], initial_alpha)
        assert torch.allclose(likelihood.beta[0:1], initial_beta)
        assert torch.allclose(likelihood.kappa[0:1], initial_kappa)
        assert torch.allclose(likelihood.mu[0:1], initial_mu)

    def test_scalar_input_validation(self):
        """Test input validation for scalar data."""
        likelihood = StudentT()

        # Should work with scalar
        data = torch.tensor(1.0)
        log_probs = likelihood.pdf(data)
        assert log_probs.shape == (1,)

        # Should fail with vector
        with pytest.raises(ValueError, match="scalar input"):
            vector_data = torch.tensor([1.0, 2.0])
            likelihood.pdf(vector_data)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_computation(self):
        """Test computation on GPU."""
        likelihood = StudentT(device="cuda")
        data = torch.tensor(1.0, device="cuda")

        log_probs = likelihood.pdf(data)
        assert log_probs.device.type == "cuda"

        likelihood.update_theta(data)
        assert likelihood.alpha.device.type == "cuda"

    def test_inference_mode_construction(self):
        """A model built under inference mode still updates outside it."""
        with torch.inference_mode():
            model = StudentT(alpha=0.1, beta=0.01, kappa=1.0, mu=0.0, device="cpu")
        data = torch.tensor(1.25)
        log_probs = model.pdf(data)
        model.update_theta(data)
        assert log_probs.shape == (1,)
        assert torch.isfinite(log_probs).all()
        assert model.alpha.shape == (2,)
        assert model._prior.shape == (4, 1)
        model.update_theta(torch.tensor(-0.5))
        assert model.alpha.shape == (3,)
        assert torch.isfinite(model.mu).all()

    def test_prior_column_after_first_observation_at_large_offset(self):
        """Run length 0 is the centered prior, including at a large offset."""
        offset = 1.0e8
        mu0, alpha0, beta0, kappa0 = 5.0, 0.1, 0.01, 1.0
        model = StudentT(alpha=alpha0, beta=beta0, kappa=kappa0, mu=mu0, device="cpu")
        assert "_prior" not in vars(model)

        model.update_theta(torch.tensor(offset, dtype=torch.float64))

        expected = torch.tensor(
            [[mu0 - offset], [kappa0], [alpha0], [beta0]],
            dtype=torch.float32,
        )
        assert model._prior.shape == (4, 1)
        assert model._prior.dtype == torch.float32
        assert model._prior.device == torch.device("cpu")
        assert torch.equal(model._prior, expected)
        assert torch.equal(model._mu_c[:1], model._prior[0])
        assert torch.equal(model.kappa[:1], model._prior[1])
        assert torch.equal(model.alpha[:1], model._prior[2])
        assert torch.equal(model.beta[:1], model._prior[3])

    def test_to_before_and_after_first_observation(self, monkeypatch):
        """``to()`` runs with no prior column, then moves that column."""
        seen = []
        real_to = torch.Tensor.to

        def record(tensor, *args, **kwargs):
            result = real_to(tensor, *args, **kwargs)
            seen.append(result)
            return result

        monkeypatch.setattr(torch.Tensor, "to", record)

        model = StudentT(alpha=0.2, beta=0.3, kappa=1.5, mu=4.0, device="cpu")
        # CPU-only runs have no second device. Point the model elsewhere so
        # ``to()`` takes the move path instead of returning immediately.
        model.device = torch.device("meta")
        seen.clear()
        returned = model.to("cpu")
        assert returned is model
        assert model.device == torch.device("cpu")
        assert "_prior" not in vars(model)
        assert any(tensor is model.alpha for tensor in seen)
        assert any(tensor is model.beta for tensor in seen)
        assert any(tensor is model.kappa for tensor in seen)
        assert any(tensor is model._mu_c for tensor in seen)

        model.update_theta(torch.tensor(1.0e8, dtype=torch.float64))
        prior = model._prior.clone()
        seen.clear()
        model.device = torch.device("meta")
        model.to("cpu")
        assert any(tensor is model._prior for tensor in seen)
        assert model._prior.device == torch.device("cpu")
        assert torch.equal(model._prior, prior)
        assert model.alpha.device == torch.device("cpu")
        assert model._mu_c.device == torch.device("cpu")

    def test_subclass_alpha0_is_read_at_the_first_observation(self):
        """A subclass that sets ``alpha0`` after ``super().__init__`` is used."""

        class RaisedAlpha(StudentT):
            def __init__(self):
                super().__init__(alpha=0.1, beta=0.01, kappa=1.0, mu=0.0, device="cpu")
                self.alpha0 = 4.0

        model = RaisedAlpha()
        model.update_theta(torch.tensor(1.5))
        assert model.alpha[0].item() == pytest.approx(4.0)
        assert model.beta[0].item() == pytest.approx(0.01)
        assert model.kappa[0].item() == pytest.approx(1.0)
        assert model._prior[2, 0].item() == pytest.approx(4.0)


class TestMultivariateT:
    """Test multivariate Student's t-distribution likelihood."""

    def test_initialization(self):
        """Test MultivariateT initialization."""
        dims = 3
        likelihood = MultivariateT(dims=dims)

        assert likelihood.dims == dims
        assert likelihood.dof0 == dims + 1
        assert likelihood.kappa0 == 1.0
        assert likelihood.mu0.shape == (dims,)
        assert likelihood.scale0.shape == (dims, dims)
        assert likelihood.t == 0

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        dims = 2
        custom_mu = torch.tensor([1.0, 2.0])
        custom_scale = torch.eye(2) * 2.0

        likelihood = MultivariateT(
            dims=dims, dof=10, kappa=2.0, mu=custom_mu, scale=custom_scale
        )

        assert likelihood.dof0 == 10
        assert likelihood.kappa0 == 2.0
        # the model moves its parameters to its own device (e.g. MPS/CUDA),
        # so compare on the device the model chose
        assert torch.allclose(likelihood.mu0.cpu(), custom_mu)
        assert torch.allclose(likelihood.scale0.cpu(), custom_scale)

    def test_pdf_single_observation(self):
        """Test PDF computation for single observation."""
        dims = 3
        likelihood = MultivariateT(dims=dims)
        data = torch.randn(dims)

        log_probs = likelihood.pdf(data)

        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == (1,)
        assert torch.isfinite(log_probs).all()
        assert likelihood.t == 1

    def test_pdf_multiple_observations(self):
        """Test PDF computation for multiple observations."""
        dims = 2
        likelihood = MultivariateT(dims=dims)

        # First observation
        data1 = torch.randn(dims)
        log_probs1 = likelihood.pdf(data1)
        likelihood.update_theta(data1)

        # Second observation
        data2 = torch.randn(dims)
        log_probs2 = likelihood.pdf(data2)

        assert log_probs1.shape == (1,)
        assert log_probs2.shape == (2,)
        assert likelihood.t == 2

    def test_update_theta(self):
        """Test parameter updates for multivariate case."""
        dims = 2
        likelihood = MultivariateT(dims=dims)
        data = torch.randn(dims)

        assert likelihood.mu.shape == (1, dims)
        assert likelihood.scale.shape == (1, dims, dims)

        likelihood.update_theta(data)

        # Parameters should have grown
        assert likelihood.mu.shape == (2, dims)
        assert likelihood.scale.shape == (2, dims, dims)
        assert likelihood.dof.shape == (2,)
        assert likelihood.kappa.shape == (2,)

        # First elements should be initial parameters
        assert torch.allclose(likelihood.mu[0], likelihood.mu0)
        assert torch.allclose(likelihood.scale[0], likelihood.scale0)

    def test_input_shape_validation(self):
        """Test input shape validation."""
        dims = 3
        likelihood = MultivariateT(dims=dims)

        # Should work with correct shape
        data = torch.randn(dims)
        log_probs = likelihood.pdf(data)
        assert log_probs.shape == (1,)

        # Should fail with wrong shape
        with pytest.raises(ValueError, match="Expected data shape"):
            wrong_data = torch.randn(dims + 1)
            likelihood.pdf(wrong_data)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_computation_multivariate(self):
        """Test multivariate computation on GPU."""
        dims = 2
        likelihood = MultivariateT(dims=dims, device="cuda")
        data = torch.randn(dims, device="cuda")

        log_probs = likelihood.pdf(data)
        assert log_probs.device.type == "cuda"

        likelihood.update_theta(data)
        assert likelihood.mu.device.type == "cuda"
        assert likelihood.scale.device.type == "cuda"


class TestConsistencyWithOriginal:
    """Test consistency with the original implementation behavior."""

    def test_studentt_parameter_evolution(self):
        """Test that parameters evolve as expected."""
        likelihood = StudentT(alpha=1.0, beta=1.0, kappa=1.0, mu=0.0)

        # Sequence of observations
        observations = [1.0, 2.0, -1.0, 0.5]

        for i, obs in enumerate(observations):
            data = torch.tensor(obs)
            log_prob = likelihood.pdf(data)

            # Should have i+1 run lengths
            assert log_prob.shape == (i + 1,)
            assert torch.isfinite(log_prob).all()

            likelihood.update_theta(data)

            # Parameters should grow
            assert likelihood.alpha.shape[0] == i + 2
            assert likelihood.beta.shape[0] == i + 2
            assert likelihood.kappa.shape[0] == i + 2
            assert likelihood.mu.shape[0] == i + 2

    def test_multivariate_parameter_evolution(self):
        """Test multivariate parameter evolution."""
        dims = 2
        likelihood = MultivariateT(dims=dims, dof=dims + 1, kappa=1.0)

        # Sequence of observations
        observations = [
            torch.tensor([1.0, 0.5]),
            torch.tensor([-0.5, 1.5]),
            torch.tensor([0.0, 0.0]),
        ]

        for i, data in enumerate(observations):
            log_prob = likelihood.pdf(data)

            # Should have i+1 run lengths
            assert log_prob.shape == (i + 1,)
            assert torch.isfinite(log_prob).all()

            likelihood.update_theta(data)

            # Parameters should grow
            assert likelihood.mu.shape == (i + 2, dims)
            assert likelihood.scale.shape == (i + 2, dims, dims)
            assert likelihood.dof.shape == (i + 2,)
            assert likelihood.kappa.shape == (i + 2,)
