#!/usr/bin/env python3
"""
Simple example demonstrating basic usage of bayesian_changepoint_detection.

This example shows:
1. Online changepoint detection with synthetic data
2. Offline changepoint detection with synthetic data
3. Basic visualization of results
"""

import torch
import matplotlib.pyplot as plt
from functools import partial

from bayesian_changepoint_detection import (
    online_changepoint_detection,
    offline_changepoint_detection,
    constant_hazard,
    const_prior,
    StudentT
)
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT

def create_synthetic_data():
    """Create synthetic data with known changepoints."""
    torch.manual_seed(42)
    
    # Create data with obvious changepoints at positions 50 and 100
    segment1 = torch.randn(50) + 0      # mean=0, std=1
    segment2 = torch.randn(50) + 3      # mean=3, std=1  
    segment3 = torch.randn(50) + 0      # mean=0, std=1
    
    data = torch.cat([segment1, segment2, segment3])
    true_changepoints = [50, 100]
    
    return data, true_changepoints

def run_online_detection(data):
    """Run online changepoint detection."""
    print("Running online changepoint detection...")
    
    # Set up hazard function (prior over changepoint locations)
    hazard_func = partial(constant_hazard, 250)  # Expected run length of 250
    
    # Set up likelihood model
    likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
    
    # Run detection
    run_length_probs, changepoint_probs = online_changepoint_detection(
        data, hazard_func, likelihood
    )
    
    print(f"✓ Online detection completed")
    print(f"  Max changepoint probability: {changepoint_probs.max().item():.4f}")
    
    return run_length_probs, changepoint_probs

def run_offline_detection(data):
    """Run offline changepoint detection."""
    print("Running offline changepoint detection...")
    
    # Set up prior function
    prior_func = partial(const_prior, p=1/(len(data)+1))
    
    # Set up likelihood model
    likelihood = OfflineStudentT()
    
    # Run detection
    Q, P, changepoint_log_probs = offline_changepoint_detection(
        data, prior_func, likelihood
    )
    
    # Get changepoint probabilities
    changepoint_probs = torch.exp(changepoint_log_probs).sum(0)
    
    print(f"✓ Offline detection completed")
    print(f"  Max changepoint probability: {changepoint_probs.max().item():.4f}")
    
    return Q, P, changepoint_probs

def plot_results(data, online_probs, offline_probs, true_changepoints):
    """Plot the results."""
    print("Creating visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Data
    plt.subplot(3, 1, 1)
    plt.plot(data.cpu().numpy(), 'b-', alpha=0.7, label='Data')
    for cp in true_changepoints:
        plt.axvline(x=cp, color='r', linestyle='--', alpha=0.8, label='True changepoint' if cp == true_changepoints[0] else '')
    plt.title('Synthetic Data with Known Changepoints')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Online detection results
    plt.subplot(3, 1, 2)
    plt.plot(online_probs.cpu().numpy(), 'g-', label='Online changepoint probability')
    for cp in true_changepoints:
        plt.axvline(x=cp, color='r', linestyle='--', alpha=0.8)
    plt.title('Online Changepoint Detection Results')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Offline detection results
    plt.subplot(3, 1, 3)
    plt.plot(offline_probs.cpu().numpy(), 'm-', label='Offline changepoint probability')
    for cp in true_changepoints:
        plt.axvline(x=cp, color='r', linestyle='--', alpha=0.8)
    plt.title('Offline Changepoint Detection Results')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('changepoint_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved as 'changepoint_detection_results.png'")

def main():
    """Main function."""
    print("=" * 60)
    print("Bayesian Changepoint Detection - Simple Example")
    print("=" * 60)
    
    # Create synthetic data
    data, true_changepoints = create_synthetic_data()
    print(f"Generated data with {len(data)} points")
    print(f"True changepoints at: {true_changepoints}")
    
    # Run online detection
    run_length_probs, online_changepoint_probs = run_online_detection(data)
    
    # Run offline detection
    Q, P, offline_changepoint_probs = run_offline_detection(data)
    
    # Find detected changepoints (simple peak detection)
    online_peaks = torch.where(online_changepoint_probs > 0.01)[0]
    offline_peaks = torch.where(offline_changepoint_probs > 0.01)[0]
    
    print(f"\nDetected changepoints:")
    print(f"  Online method: {online_peaks.tolist()[:5]}...")  # Show first 5
    print(f"  Offline method: {offline_peaks.tolist()[:5]}...")  # Show first 5
    
    # Create visualization
    try:
        plot_results(data, online_changepoint_probs, offline_changepoint_probs, true_changepoints)
    except ImportError:
        print("⚠ matplotlib not available for plotting")
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()