"""
Integration tests for the complete refactored system.
"""

import pytest
import torch
from functools import partial

from bayesian_changepoint_detection import (
    changepoint_probabilities,
    get_map_changepoints,
    online_changepoint_detection,
    offline_changepoint_detection,
    get_device,
)
from bayesian_changepoint_detection.online_likelihoods import StudentT, MultivariateT
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.generate_data import (
    generate_mean_shift_example,
    generate_multivariate_normal_time_series
)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_basic_online_detection_flow(self):
        """Test the complete online detection workflow."""
        # Generate test data
        partition, data = generate_mean_shift_example(
            num_segments=3, segment_length=50, device='cpu'
        )
        
        # Set up detection
        hazard_func = partial(constant_hazard, 100)
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device='cpu')
        
        # Run detection
        R, map_run_lengths = online_changepoint_detection(
            data.squeeze(), hazard_func, likelihood, device='cpu'
        )
        
        # Verify outputs
        assert isinstance(R, torch.Tensor)
        assert isinstance(map_run_lengths, torch.Tensor)
        assert R.shape[1] == len(data) + 1  # T+1 time steps
        assert len(map_run_lengths) == len(data) + 1
        assert torch.all(R >= 0)  # Probabilities should be non-negative
        assert map_run_lengths.dtype == torch.long
        assert torch.all(map_run_lengths >= 0)
    
    def test_basic_offline_detection_flow(self):
        """Test the complete offline detection workflow."""
        # Generate test data
        partition, data = generate_mean_shift_example(
            num_segments=3, segment_length=50, device='cpu'
        )
        
        # Set up detection
        prior_func = partial(const_prior, p=1/(len(data)+1))
        likelihood = OfflineStudentT(device='cpu')
        
        # Run detection
        Q, P, Pcp = offline_changepoint_detection(
            data.squeeze(), prior_func, likelihood, device='cpu'
        )
        
        # Verify outputs
        assert isinstance(Q, torch.Tensor)
        assert isinstance(P, torch.Tensor)
        assert isinstance(Pcp, torch.Tensor)
        assert len(Q) == len(data)
        assert P.shape == (len(data), len(data))
        assert Pcp.shape == (len(data) - 1, len(data) - 1)
    
    def test_multivariate_online_detection(self):
        """Test multivariate online detection."""
        # Generate multivariate test data
        partition, data = generate_multivariate_normal_time_series(
            num_segments=2, dims=3, min_length=30, max_length=50, device='cpu'
        )
        
        # Set up detection
        hazard_func = partial(constant_hazard, 80)
        likelihood = MultivariateT(dims=3, device='cpu')
        
        # Run detection
        R, map_run_lengths = online_changepoint_detection(
            data, hazard_func, likelihood, device='cpu'
        )
        
        # Verify outputs
        assert isinstance(R, torch.Tensor)
        assert isinstance(map_run_lengths, torch.Tensor)
        assert R.shape[1] == data.shape[0] + 1
        assert len(map_run_lengths) == data.shape[0] + 1
    
    def test_device_consistency(self):
        """Test that device handling is consistent throughout."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Generate data on CPU
        partition, data = generate_mean_shift_example(
            num_segments=2, segment_length=30, device='cpu'
        )
        
        # Run detection on GPU
        hazard_func = partial(constant_hazard, 60)
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device='cuda')
        
        R, map_run_lengths = online_changepoint_detection(
            data.squeeze(), hazard_func, likelihood, device='cuda'
        )
        
        # Verify results are on GPU
        assert R.device.type == 'cuda'
        assert map_run_lengths.device.type == 'cuda'
    
    def test_backward_compatibility(self):
        """Test that the refactored code maintains backward compatibility."""
        # This test ensures the new API can handle the same patterns as the old code
        
        # Generate test data using new function
        partition, data = generate_mean_shift_example(
            num_segments=3, segment_length=40, device='cpu'
        )
        
        # Use the new API in a way similar to the old API
        hazard_func = partial(constant_hazard, 80)
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        
        # This should work without specifying device explicitly
        R, map_run_lengths = online_changepoint_detection(
            data.squeeze(), hazard_func, likelihood
        )
        
        # Should produce reasonable results
        assert torch.isfinite(R).all()
        assert torch.all(map_run_lengths.cpu() <= torch.arange(len(map_run_lengths)))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameter values."""
        # Generate data with small variance
        data = torch.randn(100) * 0.01
        
        # Use extreme parameter values
        hazard_func = partial(constant_hazard, 1000)  # Very low hazard
        likelihood = StudentT(alpha=0.001, beta=0.001, kappa=0.1, mu=0)
        
        # Should not produce NaN or infinite values
        R, map_run_lengths = online_changepoint_detection(
            data, hazard_func, likelihood
        )
        
        assert torch.isfinite(R).all()
        assert torch.all(map_run_lengths >= 0)
    
    def test_empty_and_small_data(self):
        """Test handling of edge cases with small datasets."""
        # Very small dataset
        data = torch.tensor([1.0, 2.0])
        
        hazard_func = partial(constant_hazard, 10)
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        
        # Should handle gracefully
        R, map_run_lengths = online_changepoint_detection(
            data, hazard_func, likelihood
        )
        
        assert len(map_run_lengths) == 3  # len(data) + 1
        assert torch.isfinite(R).all()
        assert torch.all(map_run_lengths >= 0)
    
    def test_performance_scaling(self):
        """Test that performance scales reasonably with data size."""
        import time
        
        sizes = [50, 100, 200]
        times = []
        
        for size in sizes:
            # Generate data
            partition, data = generate_mean_shift_example(
                num_segments=2, segment_length=size//2, device='cpu'
            )
            
            # Set up detection
            hazard_func = partial(constant_hazard, size)
            likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
            
            # Time the detection
            start_time = time.time()
            R, map_run_lengths = online_changepoint_detection(
                data.squeeze(), hazard_func, likelihood
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Performance should not degrade catastrophically
        # (This is a rough check - exact scaling depends on implementation details)
        assert times[-1] < times[0] * 20  # Should not be more than 20x slower for 4x data


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance-focused integration tests."""
    
    def test_large_dataset_online(self):
        """Test online detection on a larger dataset."""
        # Generate larger dataset
        partition, data = generate_mean_shift_example(
            num_segments=5, segment_length=200, device='cpu'
        )
        
        hazard_func = partial(constant_hazard, 500)
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        
        # Should complete without memory issues
        R, map_run_lengths = online_changepoint_detection(
            data.squeeze(), hazard_func, likelihood
        )
        
        assert torch.isfinite(R).all()
        assert torch.all(map_run_lengths >= 0)
    
    def test_large_dataset_multivariate(self):
        """Test multivariate detection on larger dataset."""
        # Generate larger multivariate dataset
        partition, data = generate_multivariate_normal_time_series(
            num_segments=3, dims=5, min_length=100, max_length=150, device='cpu'
        )
        
        hazard_func = partial(constant_hazard, 300)
        likelihood = MultivariateT(dims=5)
        
        # Should complete without memory issues
        R, map_run_lengths = online_changepoint_detection(
            data, hazard_func, likelihood
        )
        
        assert torch.isfinite(R).all()
        assert torch.all(map_run_lengths >= 0)


class TestRegressionAgainstOriginal:
    """Tests to ensure refactored code produces similar results to original."""
    
    def test_simple_detection_regression(self):
        """Test that we get reasonable changepoint detection on known data."""
        # Create data with obvious changepoints
        segment1 = torch.zeros(50)
        segment2 = torch.ones(50) * 5  # Clear shift
        segment3 = torch.zeros(50)
        data = torch.cat([segment1, segment2, segment3])
        
        # Add small amount of noise
        torch.manual_seed(0)
        data += torch.randn_like(data) * 0.1
        
        # Run detection with more sensitive parameters
        hazard_func = partial(constant_hazard, 50)  # More frequent changepoints expected
        likelihood = StudentT(alpha=0.01, beta=0.01, kappa=1, mu=0)  # More sensitive
        
        R, map_run_lengths = online_changepoint_detection(
            data, hazard_func, likelihood
        )
        
        # Should detect changepoints around positions 50 and 100: the MAP
        # run length resets there, and the lagged probability peaks there.
        detected = get_map_changepoints(R, min_separation=10).tolist()
        probs = changepoint_probabilities(R, lag=5)
        high_prob_indices = (torch.where(probs[1:] > 0.5)[0] + 1).tolist()
        
        # Check if we detected changepoints near positions 50 and 100
        detected_near_50 = any(45 <= idx <= 55 for idx in detected + high_prob_indices)
        detected_near_100 = any(95 <= idx <= 105 for idx in detected + high_prob_indices)
        
        # Should detect at least one of the changepoints
        assert detected_near_50 and detected_near_100, (
            f"Expected changepoints near 50 and 100; MAP starts {detected}, "
            f"lag-5 probability > 0.5 at {high_prob_indices}"
        )
        
