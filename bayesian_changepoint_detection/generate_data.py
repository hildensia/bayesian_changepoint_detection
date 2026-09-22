"""
Data generation utilities for testing changepoint detection algorithms.

This module provides functions to generate synthetic time series data with
known changepoints for testing and benchmarking changepoint detection methods.
"""

from typing import Optional, Union

import torch

from .device import ensure_tensor, get_device


def generate_normal_time_series(
    num_segments: int,
    min_length: int = 50,
    max_length: int = 1000,
    seed: Optional[int] = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate univariate time series with changepoints in mean and variance.

    Creates a time series consisting of multiple segments, each with different
    Gaussian parameters (mean and variance).

    Parameters
    ----------
    num_segments : int
        Number of segments to generate.
    min_length : int, optional
        Minimum length of each segment (default: 50).
    max_length : int, optional
        Maximum length of each segment (default: 1000).
    seed : int or None, optional
        Random seed for reproducibility (default: 42).
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Returns
    -------
    partition : torch.Tensor
        Length of each segment. Shape: [num_segments].
    data : torch.Tensor
        Generated time series data. Shape: [T, 1] where T is total length.

    Examples
    --------
    >>> partition, data = generate_normal_time_series(3, 50, 200, seed=42)
    >>> print(f"Generated {len(data)} data points in {len(partition)} segments")
    >>> print(f"Segment lengths: {partition}")

    Notes
    -----
    Each segment has:
    - Mean sampled from Normal(0, 10²)
    - Standard deviation sampled from |Normal(0, 1)| + 0.1 (kept away from 0)
    """
    device = get_device(device)

    if seed is not None:
        torch.manual_seed(seed)

    # Generate segment lengths
    partition = torch.randint(
        min_length, max_length + 1, (num_segments,), device=device, dtype=torch.long
    )

    # Generate data for each segment
    data_segments = []

    for segment_length in partition:
        # Random mean and variance for this segment
        mean = torch.randn(1, device=device) * 10
        std = torch.abs(torch.randn(1, device=device)) + 0.1  # Ensure positive

        # Generate segment data
        segment_data = torch.normal(
            mean.expand(segment_length), std.expand(segment_length)
        )
        data_segments.append(segment_data)

    # Concatenate all segments
    data = torch.cat(data_segments).unsqueeze(1)  # Shape: [T, 1]

    return partition, data


def generate_multivariate_normal_time_series(
    num_segments: int,
    dims: int,
    min_length: int = 50,
    max_length: int = 1000,
    seed: Optional[int] = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multivariate time series with changepoints in mean and covariance.

    Creates a multivariate time series with segments having different
    Gaussian parameters (mean vectors and covariance matrices).

    Parameters
    ----------
    num_segments : int
        Number of segments to generate.
    dims : int
        Dimensionality of the time series.
    min_length : int, optional
        Minimum length of each segment (default: 50).
    max_length : int, optional
        Maximum length of each segment (default: 1000).
    seed : int or None, optional
        Random seed for reproducibility (default: 42).
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Returns
    -------
    partition : torch.Tensor
        Length of each segment. Shape: [num_segments].
    data : torch.Tensor
        Generated time series data. Shape: [T, dims] where T is total length.

    Examples
    --------
    >>> partition, data = generate_multivariate_normal_time_series(3, 5, seed=42)
    >>> print(f"Generated {data.shape[0]} time points with {data.shape[1]} dimensions")
    >>> print(f"Segment lengths: {partition}")

    Notes
    -----
    Each segment has:
    - Mean vector sampled from Normal(0, 10²) for each dimension
    - Covariance matrix generated as A @ A.T where A ~ Normal(0, 1)
    """
    device = get_device(device)

    if seed is not None:
        torch.manual_seed(seed)

    # Generate segment lengths
    partition = torch.randint(
        min_length, max_length + 1, (num_segments,), device=device, dtype=torch.long
    )

    # Generate data for each segment
    data_segments = []

    for segment_length in partition:
        # Random mean vector for this segment
        mean = torch.randn(dims, device=device) * 10

        # Generate random positive definite covariance matrix
        A = torch.randn(dims, dims, device=device)
        cov = torch.matmul(A, A.T)

        # Ensure numerical stability
        cov = cov + 1e-6 * torch.eye(dims, device=device)

        # Generate segment data using multivariate normal
        try:
            mvn = torch.distributions.MultivariateNormal(mean, cov)
            segment_data = mvn.sample((segment_length,))
        except RuntimeError:
            # Fallback: use independent normals if covariance is problematic
            std = torch.sqrt(torch.diag(cov))
            segment_data = torch.normal(
                mean.unsqueeze(0).expand(segment_length, -1),
                std.unsqueeze(0).expand(segment_length, -1),
            )

        data_segments.append(segment_data)

    # Concatenate all segments
    data = torch.cat(data_segments, dim=0)  # Shape: [T, dims]

    return partition, data


def generate_correlation_change_example(
    min_length: int = 50,
    max_length: int = 1000,
    seed: Optional[int] = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate the motivating example from Xiang & Murphy (2007).

    Creates a 2D time series with three segments that have the same mean
    but different correlation structures, demonstrating changepoints that
    are only detectable through covariance changes.

    Parameters
    ----------
    min_length : int, optional
        Minimum length of each segment (default: 50).
    max_length : int, optional
        Maximum length of each segment (default: 1000).
    seed : int or None, optional
        Random seed for reproducibility (default: 42).
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Returns
    -------
    partition : torch.Tensor
        Length of each segment. Shape: [3].
    data : torch.Tensor
        Generated time series data. Shape: [T, 2] where T is total length.

    Examples
    --------
    >>> partition, data = generate_correlation_change_example(seed=42)
    >>> print(f"Generated correlation change example with segments: {partition}")

    Notes
    -----
    The three segments have covariance matrices:
    1. [[1.0, 0.75], [0.75, 1.0]]   - Positive correlation
    2. [[1.0, 0.0],  [0.0, 1.0]]    - No correlation
    3. [[1.0, -0.75], [-0.75, 1.0]] - Negative correlation

    All segments have zero mean, so changepoints are only in correlation.

    References
    ----------
    Xiang, X., & Murphy, K. (2007). Modeling changing dependency structure
    in multivariate time series. ICML, 1055-1062.
    """
    device = get_device(device)

    if seed is not None:
        torch.manual_seed(seed)

    dims = 2
    num_segments = 3

    # Generate segment lengths
    partition = torch.randint(
        min_length, max_length + 1, (num_segments,), device=device, dtype=torch.long
    )

    # Zero mean for all segments
    mu = torch.zeros(dims, device=device)

    # Define the three covariance matrices
    Sigma1 = torch.tensor(
        [[1.0, 0.75], [0.75, 1.0]], device=device
    )  # Positive correlation
    Sigma2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)  # No correlation
    Sigma3 = torch.tensor(
        [[1.0, -0.75], [-0.75, 1.0]], device=device
    )  # Negative correlation

    covariances = [Sigma1, Sigma2, Sigma3]
    data_segments = []

    for segment_length, cov in zip(partition, covariances):
        # Generate segment data
        mvn = torch.distributions.MultivariateNormal(mu, cov)
        segment_data = mvn.sample((segment_length,))
        data_segments.append(segment_data)

    # Concatenate all segments
    data = torch.cat(data_segments, dim=0)  # Shape: [T, 2]

    return partition, data


def generate_mean_shift_example(
    num_segments: int = 4,
    segment_length: int = 100,
    shift_magnitude: float = 3.0,
    noise_std: float = 1.0,
    seed: Optional[int] = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate time series with abrupt mean shifts.

    Creates a time series with segments of equal length but different means,
    useful for testing basic changepoint detection capabilities.

    Parameters
    ----------
    num_segments : int, optional
        Number of segments (default: 4).
    segment_length : int, optional
        Length of each segment (default: 100).
    shift_magnitude : float, optional
        Magnitude of mean shifts between segments (default: 3.0).
    noise_std : float, optional
        Standard deviation of noise (default: 1.0).
    seed : int or None, optional
        Random seed for reproducibility (default: 42).
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Returns
    -------
    partition : torch.Tensor
        Length of each segment. Shape: [num_segments].
    data : torch.Tensor
        Generated time series data. Shape: [T, 1] where T is total length.

    Examples
    --------
    >>> partition, data = generate_mean_shift_example(4, 100, shift_magnitude=2.0)
    >>> print(f"Generated mean shift example: {len(data)} points, {len(partition)} segments")

    Notes
    -----
    Means alternate between 0 and shift_magnitude, creating a step function
    pattern that should be easy to detect.
    """
    device = get_device(device)

    if seed is not None:
        torch.manual_seed(seed)

    # All segments have the same length
    partition = torch.full(
        (num_segments,), segment_length, device=device, dtype=torch.long
    )

    data_segments = []

    for i in range(num_segments):
        # Alternate between 0 and shift_magnitude
        mean = (i % 2) * shift_magnitude

        # Generate segment data
        segment_data = torch.normal(mean, noise_std, (segment_length,), device=device)
        data_segments.append(segment_data)

    # Concatenate all segments
    data = torch.cat(data_segments).unsqueeze(1)  # Shape: [T, 1]

    return partition, data


def generate_variance_change_example(
    num_segments: int = 3,
    segment_length: int = 150,
    variance_levels: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate time series with variance changes but constant mean.

    Creates segments with the same mean but different variances,
    testing the ability to detect heteroscedastic changepoints.

    Parameters
    ----------
    num_segments : int, optional
        Number of segments (default: 3).
    segment_length : int, optional
        Length of each segment (default: 150).
    variance_levels : torch.Tensor or None, optional
        Variance levels for each segment. If None, uses [0.5, 2.0, 0.8].
    seed : int or None, optional
        Random seed for reproducibility (default: 42).
    device : str, torch.device, or None, optional
        Device to place tensors on.

    Returns
    -------
    partition : torch.Tensor
        Length of each segment. Shape: [num_segments].
    data : torch.Tensor
        Generated time series data. Shape: [T, 1] where T is total length.

    Examples
    --------
    >>> partition, data = generate_variance_change_example(3, 100)
    >>> print(f"Generated variance change example with {len(data)} points")

    Notes
    -----
    All segments have zero mean, so changepoints are only detectable through
    variance changes. This tests the algorithm's sensitivity to second-moment changes.
    """
    device = get_device(device)

    if seed is not None:
        torch.manual_seed(seed)

    if variance_levels is None:
        variance_levels = torch.tensor([0.5, 2.0, 0.8], device=device)
    else:
        variance_levels = ensure_tensor(variance_levels, device=device)

    if len(variance_levels) != num_segments:
        raise ValueError(
            f"Number of variance levels ({len(variance_levels)}) must match num_segments ({num_segments})"
        )

    # All segments have the same length
    partition = torch.full(
        (num_segments,), segment_length, device=device, dtype=torch.long
    )

    data_segments = []

    for variance in variance_levels:
        # Zero mean, different variance
        std = torch.sqrt(variance)
        segment_data = torch.normal(0.0, std.item(), (segment_length,), device=device)
        data_segments.append(segment_data)

    # Concatenate all segments
    data = torch.cat(data_segments).unsqueeze(1)  # Shape: [T, 1]

    return partition, data


# Backward compatibility with original function names
def generate_multinormal_time_series(*args, **kwargs):
    """Backward compatibility wrapper for generate_multivariate_normal_time_series."""
    return generate_multivariate_normal_time_series(*args, **kwargs)


def generate_xuan_motivating_example(*args, **kwargs):
    """Backward compatibility wrapper for generate_correlation_change_example."""
    return generate_correlation_change_example(*args, **kwargs)
