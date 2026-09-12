#!/usr/bin/env python3
"""
Basic usage example for PyTorch-based Bayesian changepoint detection.

This example demonstrates the basic functionality of the refactored library
using both online and offline changepoint detection methods.
"""

import torch
from functools import partial
import matplotlib.pyplot as plt

# Import the refactored modules
from bayesian_changepoint_detection import (
    online_changepoint_detection,
    offline_changepoint_detection,
    changepoint_probabilities,
    get_map_changepoints,
    get_device,
    get_device_info,
)
from bayesian_changepoint_detection.online_likelihoods import StudentT
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.generate_data import generate_mean_shift_example


def main():
    """Run basic usage examples."""
    print("=" * 60)
    print("Bayesian Changepoint Detection - Basic Usage Example")
    print("=" * 60)
    
    # Display device information
    device = get_device()
    device_info = get_device_info()
    print(f"Using device: {device}")
    print(f"Available devices: {device_info['devices']}")
    print(f"CUDA available: {device_info['cuda_available']}")
    print()
    
    # Generate synthetic data with known changepoints
    print("Generating synthetic data...")
    torch.manual_seed(42)  # For reproducibility
    
    # Create a simple step function with 4 segments
    partition, data = generate_mean_shift_example(
        num_segments=4, 
        segment_length=100, 
        shift_magnitude=3.0,
        noise_std=1.0,
        device=device
    )
    
    print(f"Generated {len(data)} data points in {len(partition)} segments")
    print(f"True segment lengths: {partition.tolist()}")
    print(f"True changepoints at: {torch.cumsum(partition, 0)[:-1].tolist()}")
    print()
    
    # Example 1: Online Changepoint Detection
    print("=" * 40)
    print("Online Changepoint Detection")
    print("=" * 40)
    
    # Set up online detection
    hazard_func = partial(constant_hazard, 250)  # Expected run length = 250
    online_likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device=device)
    
    print("Running online changepoint detection...")
    R, map_run_lengths = online_changepoint_detection(
        data.squeeze(), hazard_func, online_likelihood, device=device
    )
    
    # Segment starts implied by the MAP run-length path
    detected_online = get_map_changepoints(R, min_separation=10)
    print(f"Detected changepoints (MAP run-length path): {detected_online.tolist()}")
    
    # Calibrated probability per position, judged `lag` observations later
    lag = 10
    changepoint_probs = torch.zeros(len(data) + 1, device=R.device)
    changepoint_probs[: len(data) + 1 - lag] = changepoint_probabilities(R, lag=lag)
    threshold = 0.5
    peaks = (torch.where(changepoint_probs[1:] > threshold)[0] + 1).tolist()
    print(f"Positions with P(change) > {threshold} at lag {lag}: {peaks}")
    print()
    
    # Example 2: Offline Changepoint Detection
    print("=" * 40)
    print("Offline Changepoint Detection")
    print("=" * 40)
    
    # Set up offline detection
    prior_func = partial(const_prior, p=1/(len(data)+1))
    offline_likelihood = OfflineStudentT(device=device)
    
    print("Running offline changepoint detection...")
    Q, P, Pcp = offline_changepoint_detection(
        data.squeeze(), prior_func, offline_likelihood, device=device
    )
    
    # Get changepoint probabilities
    changepoint_probs_offline = torch.exp(Pcp).sum(0)
    detected_offline = torch.where(changepoint_probs_offline > 0.1)[0]
    print(f"Detected changepoints (threshold=0.1): {detected_offline.tolist()}")
    
    # Find peaks for offline detection
    offline_peaks = []
    for i in range(len(changepoint_probs_offline)):
        if i == 0 or i == len(changepoint_probs_offline) - 1:
            continue
        if (changepoint_probs_offline[i] > changepoint_probs_offline[i-1] and 
            changepoint_probs_offline[i] > changepoint_probs_offline[i+1] and 
            changepoint_probs_offline[i] > 0.01):
            offline_peaks.append(i)
    
    print(f"Detected peaks in offline changepoint probabilities: {offline_peaks}")
    print()
    
    # Visualization (if matplotlib is available)
    try:
        print("Creating visualization...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot original data
        axes[0].plot(data.cpu().numpy(), 'b-', linewidth=1)
        axes[0].set_title('Original Time Series Data')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Mark true changepoints
        true_changepoints = torch.cumsum(partition, 0)[:-1]
        for cp in true_changepoints:
            axes[0].axvline(cp.item(), color='red', linestyle='--', alpha=0.7, label='True changepoint')
        if len(true_changepoints) > 0:
            axes[0].legend()
        
        # Plot online detection results
        axes[1].plot(changepoint_probs.cpu().numpy(), 'g-', linewidth=2)  # lag-10 P(change at t)
        axes[1].set_title('Online Changepoint Detection Probabilities')
        axes[1].set_ylabel('Probability')
        axes[1].grid(True, alpha=0.3)
        
        # Mark detected peaks
        for peak in peaks:
            axes[1].axvline(peak, color='orange', linestyle=':', alpha=0.8)
        
        # Plot offline detection results
        axes[2].plot(changepoint_probs_offline.cpu().numpy(), 'purple', linewidth=2)
        axes[2].set_title('Offline Changepoint Detection Probabilities')
        axes[2].set_ylabel('Probability')
        axes[2].set_xlabel('Time')
        axes[2].grid(True, alpha=0.3)
        
        # Mark detected peaks
        for peak in offline_peaks:
            axes[2].axvline(peak, color='orange', linestyle=':', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('changepoint_detection_example.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'changepoint_detection_example.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    
    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()