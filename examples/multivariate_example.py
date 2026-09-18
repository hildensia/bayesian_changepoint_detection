#!/usr/bin/env python3
"""
Multivariate changepoint detection example.

This example demonstrates changepoint detection for multivariate time series
using the PyTorch-based implementation.
"""

from functools import partial

import matplotlib.pyplot as plt
import torch

# Import the refactored modules
from bayesian_changepoint_detection import (
    changepoint_probabilities,
    get_device,
    get_map_changepoints,
    online_changepoint_detection,
)
from bayesian_changepoint_detection.generate_data import (
    generate_correlation_change_example,
    generate_multivariate_normal_time_series,
)
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.online_likelihoods import MultivariateT


def correlation_change_example():
    """Demonstrate detection of correlation changes."""
    print("=" * 50)
    print("Correlation Change Detection Example")
    print("=" * 50)

    device = get_device()
    print(f"Using device: {device}")

    # Generate Xiang & Murphy's motivating example
    print("Generating correlation change example...")
    partition, data = generate_correlation_change_example(
        min_length=100, max_length=150, seed=42, device=device
    )

    print(f"Generated {data.shape[0]} data points with {data.shape[1]} dimensions")
    print(f"Segment lengths: {partition.tolist()}")
    print(f"True changepoints at: {torch.cumsum(partition, 0)[:-1].tolist()}")
    print()

    # Set up multivariate online detection
    hazard_func = partial(constant_hazard, 200)  # Expected run length = 200
    likelihood = MultivariateT(dims=2, device=device)

    print("Running multivariate changepoint detection...")
    R, map_run_lengths = online_changepoint_detection(
        data, hazard_func, likelihood, device=device
    )

    # Segment starts implied by the MAP run-length path
    detected = get_map_changepoints(R, min_separation=10)
    print(f"Detected changepoints (MAP run-length path): {detected.tolist()}")

    # Calibrated probability per position, judged `lag` observations later
    lag = 10
    changepoint_probs = torch.zeros(data.shape[0] + 1, device=R.device)
    changepoint_probs[: data.shape[0] + 1 - lag] = changepoint_probabilities(R, lag=lag)
    peaks = (torch.where(changepoint_probs[1:] > 0.5)[0] + 1).tolist()
    print(f"Positions with P(change) > 0.5 at lag {lag}: {peaks}")

    # Visualization
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot multivariate data
        data_np = data.cpu().numpy()
        axes[0].plot(data_np[:, 0], "b-", label="Dimension 1", linewidth=1)
        axes[0].plot(data_np[:, 1], "r-", label="Dimension 2", linewidth=1)
        axes[0].set_title("Multivariate Time Series (Correlation Changes)")
        axes[0].set_ylabel("Value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mark true changepoints
        true_changepoints = torch.cumsum(partition, 0)[:-1]
        for cp in true_changepoints:
            axes[0].axvline(cp.item(), color="green", linestyle="--", alpha=0.7)

        # Plot correlation over time (approximate)
        window_size = 50
        correlations = []
        for i in range(window_size, len(data)):
            window_data = data[i - window_size : i]
            corr = torch.corrcoef(window_data.T)[0, 1].item()
            correlations.append(corr)

        axes[1].plot(range(window_size, len(data)), correlations, "purple", linewidth=2)
        axes[1].set_title(f"Rolling Correlation (window={window_size})")
        axes[1].set_ylabel("Correlation")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color="black", linestyle="-", alpha=0.3)

        # Mark true changepoints
        for cp in true_changepoints:
            axes[1].axvline(cp.item(), color="green", linestyle="--", alpha=0.7)

        # Plot changepoint probabilities
        axes[2].plot(changepoint_probs.cpu().numpy(), "orange", linewidth=2)
        axes[2].set_title("Changepoint Detection Probabilities")
        axes[2].set_ylabel("Probability")
        axes[2].set_xlabel("Time")
        axes[2].grid(True, alpha=0.3)

        # Mark detected peaks
        for peak in peaks:
            axes[2].axvline(peak, color="red", linestyle=":", alpha=0.8)

        plt.tight_layout()
        plt.savefig(
            "multivariate_changepoint_example.png", dpi=150, bbox_inches="tight"
        )
        print("Visualization saved as 'multivariate_changepoint_example.png'")

    except ImportError:
        print("Matplotlib not available - skipping visualization")

    print()


def general_multivariate_example():
    """Demonstrate detection in general multivariate data."""
    print("=" * 50)
    print("General Multivariate Detection Example")
    print("=" * 50)

    device = get_device()

    # Generate multivariate data with mean and covariance changes
    print("Generating multivariate time series...")
    partition, data = generate_multivariate_normal_time_series(
        num_segments=3, dims=4, min_length=80, max_length=120, seed=123, device=device
    )

    print(f"Generated {data.shape[0]} data points with {data.shape[1]} dimensions")
    print(f"Segment lengths: {partition.tolist()}")
    print(f"True changepoints at: {torch.cumsum(partition, 0)[:-1].tolist()}")
    print()

    # Set up multivariate online detection
    hazard_func = partial(constant_hazard, 150)
    likelihood = MultivariateT(dims=4, device=device)

    print("Running multivariate changepoint detection...")
    R, map_run_lengths = online_changepoint_detection(
        data, hazard_func, likelihood, device=device
    )

    # Segment starts implied by the MAP run-length path
    detected = get_map_changepoints(R, min_separation=10)
    print(f"Detected changepoints (MAP run-length path): {detected.tolist()}")

    # Calibrated probability per position, judged `lag` observations later
    lag = 10
    changepoint_probs = torch.zeros(data.shape[0] + 1, device=R.device)
    changepoint_probs[: data.shape[0] + 1 - lag] = changepoint_probabilities(R, lag=lag)
    peaks = (torch.where(changepoint_probs[1:] > 0.5)[0] + 1).tolist()
    print(f"Positions with P(change) > 0.5 at lag {lag}: {peaks}")

    # Visualization
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot first two dimensions of multivariate data
        data_np = data.cpu().numpy()
        for i in range(min(4, data.shape[1])):
            axes[0].plot(
                data_np[:, i], label=f"Dimension {i + 1}", linewidth=1, alpha=0.8
            )

        axes[0].set_title("Multivariate Time Series (4D)")
        axes[0].set_ylabel("Value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mark true changepoints
        true_changepoints = torch.cumsum(partition, 0)[:-1]
        for cp in true_changepoints:
            axes[0].axvline(cp.item(), color="green", linestyle="--", alpha=0.7)

        # Plot changepoint probabilities
        axes[1].plot(changepoint_probs.cpu().numpy(), "red", linewidth=2)
        axes[1].set_title("Changepoint Detection Probabilities")
        axes[1].set_ylabel("Probability")
        axes[1].set_xlabel("Time")
        axes[1].grid(True, alpha=0.3)

        # Mark detected peaks
        for peak in peaks:
            axes[1].axvline(peak, color="orange", linestyle=":", alpha=0.8)

        plt.tight_layout()
        plt.savefig("general_multivariate_example.png", dpi=150, bbox_inches="tight")
        print("Visualization saved as 'general_multivariate_example.png'")

    except ImportError:
        print("Matplotlib not available - skipping visualization")

    print()


def main():
    """Run multivariate examples."""
    print("Multivariate Bayesian Changepoint Detection Examples")
    print("=" * 60)

    # Run correlation change example
    correlation_change_example()

    # Run general multivariate example
    general_multivariate_example()

    print("=" * 60)
    print("All multivariate examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
