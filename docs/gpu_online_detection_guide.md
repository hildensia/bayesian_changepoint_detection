# GPU Acceleration Guide

This guide provides comprehensive examples for using GPU acceleration with the Bayesian Changepoint Detection library.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup and Verification](#setup-and-verification)
3. [Basic GPU Example](#basic-gpu-example)
4. [Performance Comparison](#performance-comparison)
5. [Multivariate Detection](#multivariate-detection)
6. [Offline Detection](#offline-detection)
7. [Visualization](#visualization)
8. [Memory Management](#memory-management)
9. [Best Practices](#best-practices)

## Prerequisites

Before running GPU-accelerated changepoint detection, ensure you have:

- NVIDIA GPU with CUDA support
- CUDA drivers installed
- PyTorch with CUDA support installed
- Sufficient GPU memory for your dataset

## Setup and Verification

First, verify your CUDA setup and import the necessary libraries:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# First, verify CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Import our library
from bayesian_changepoint_detection import (
    online_changepoint_detection,
    offline_changepoint_detection,
    constant_hazard,
    const_prior,
    get_device,
    get_device_info
)
from bayesian_changepoint_detection.online_likelihoods import StudentT, MultivariateT
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT
```

## Basic GPU Example

Here's a simple example to get started with GPU acceleration:

```python
import torch
from functools import partial
from bayesian_changepoint_detection import online_changepoint_detection, constant_hazard
from bayesian_changepoint_detection.online_likelihoods import StudentT

# Generate sample data
torch.manual_seed(42)
data = torch.cat([
    torch.randn(100) + 0,    # First segment: mean=0
    torch.randn(100) + 3,    # Second segment: mean=3
    torch.randn(100) + 0,    # Third segment: mean=0
])

# Set device (automatically selects GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move data to GPU
data_gpu = data.to(device)

# Set up the model on GPU
hazard_func = partial(constant_hazard, 250)
likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device=device)

# Run detection on GPU
run_length_probs, changepoint_probs = online_changepoint_detection(
    data_gpu, hazard_func, likelihood
)

# Find changepoints (threshold at 0.5)
detected = torch.where(changepoint_probs > 0.5)[0].cpu().numpy()
print(f"Detected changepoints at: {detected}")
```

## Generate Test Data

For comprehensive testing, let's create more complex test data:

```python
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate univariate data with multiple changepoints
n_segments = 5
segment_length = 200
total_length = n_segments * segment_length

# Create data with different means and variances
segments = []
true_changepoints = []

for i in range(n_segments):
    # Varying means and standard deviations
    mean = (-1) ** i * (i + 1) * 2  # Alternating means: -2, 4, -6, 8, -10
    std = 0.5 + i * 0.3             # Increasing variance: 0.5, 0.8, 1.1, 1.4, 1.7

    segment = torch.randn(segment_length) * std + mean
    segments.append(segment)

    if i > 0:  # Don't include the start as a changepoint
        true_changepoints.append(i * segment_length)

# Combine segments
data = torch.cat(segments)
print(f"Generated data shape: {data.shape}")
print(f"True changepoints at: {true_changepoints}")

# Generate multivariate data for comparison
dims = 3
mv_segments = []
for i in range(n_segments):
    mean_vector = torch.tensor([(-1)**i * (i+1), i*0.5, (-1)**(i+1) * i])
    cov_scale = 0.5 + i * 0.2
    segment = torch.randn(segment_length, dims) * cov_scale + mean_vector
    mv_segments.append(segment)

mv_data = torch.cat(mv_segments)
print(f"Generated multivariate data shape: {mv_data.shape}")
```

## Performance Comparison

Compare CPU vs GPU performance:

```python
import time

def benchmark_detection(data, device_name='cpu', n_runs=3):
    """Benchmark changepoint detection on specified device."""
    device = torch.device(device_name)

    # Move data to device
    data_device = data.to(device)

    # Setup models
    hazard_func = partial(constant_hazard, 250)
    online_likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device=device)

    times = []

    for run in range(n_runs):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        # Run online detection
        run_length_probs, changepoint_probs = online_changepoint_detection(
            data_device, hazard_func, online_likelihood
        )

        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time, changepoint_probs

# Benchmark on CPU
print("Benchmarking on CPU...")
cpu_time, cpu_std, cpu_results = benchmark_detection(data, 'cpu')
print(f"CPU time: {cpu_time:.3f} ± {cpu_std:.3f} seconds")

# Benchmark on GPU (if available)
if torch.cuda.is_available():
    print("Benchmarking on GPU...")
    gpu_time, gpu_std, gpu_results = benchmark_detection(data, 'cuda')
    print(f"GPU time: {gpu_time:.3f} ± {gpu_std:.3f} seconds")
    speedup = cpu_time / gpu_time
    print(f"GPU speedup: {speedup:.1f}x")
else:
    print("GPU not available, skipping GPU benchmark")
    gpu_results = cpu_results
```

## Multivariate Detection

GPU acceleration is especially beneficial for multivariate data:

```python
# Multivariate detection on GPU
device = get_device()  # Automatically selects best device
print(f"Using device: {device}")

# Move multivariate data to device
mv_data_device = mv_data.to(device)

# Setup multivariate model
hazard_func = partial(constant_hazard, 250)
mv_likelihood = MultivariateT(dims=dims, device=device)

print("Running multivariate changepoint detection...")
start_time = time.time()

mv_run_length_probs, mv_changepoint_probs = online_changepoint_detection(
    mv_data_device, hazard_func, mv_likelihood
)

end_time = time.time()
print(f"Multivariate detection time: {end_time - start_time:.3f} seconds")

# Find detected changepoints (threshold at 0.5)
detected_changepoints = torch.where(mv_changepoint_probs > 0.5)[0].cpu().numpy()
print(f"Detected changepoints: {detected_changepoints}")
```

## Offline Detection

GPU acceleration also works with offline detection methods:

```python
# Offline detection for comparison
print("Running offline changepoint detection...")

# Setup offline model
prior_func = partial(const_prior, p=1/(len(data)+1))
offline_likelihood = OfflineStudentT(device=device)

# Move data to device for offline detection
data_device = data.to(device)

start_time = time.time()
Q, P, changepoint_log_probs = offline_changepoint_detection(
    data_device, prior_func, offline_likelihood
)
end_time = time.time()

print(f"Offline detection time: {end_time - start_time:.3f} seconds")

# Get changepoint probabilities
offline_changepoint_probs = torch.exp(changepoint_log_probs).sum(0)
offline_detected = torch.where(offline_changepoint_probs > 0.5)[0].cpu().numpy()
print(f"Offline detected changepoints: {offline_detected}")
```

## Visualization

Create comprehensive plots to visualize results:

```python
# Plot results (convert back to CPU for plotting)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Original data with true changepoints
axes[0].plot(data.cpu().numpy())
for cp in true_changepoints:
    axes[0].axvline(cp, color='red', linestyle='--', alpha=0.7,
                   label='True changepoint' if cp == true_changepoints[0] else "")
axes[0].set_title('Original Time Series Data')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Online detection probabilities
axes[1].plot(cpu_results.cpu().numpy(), label='Online (CPU)', alpha=0.8)
if torch.cuda.is_available():
    axes[1].plot(gpu_results.cpu().numpy(), label='Online (GPU)', alpha=0.8, linestyle=':')
axes[1].axhline(0.5, color='black', linestyle='-', alpha=0.5, label='Threshold')
for cp in true_changepoints:
    axes[1].axvline(cp, color='red', linestyle='--', alpha=0.7)
axes[1].set_title('Online Changepoint Detection Probabilities')
axes[1].set_ylabel('Probability')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Multivariate detection probabilities
axes[2].plot(mv_changepoint_probs.cpu().numpy(), label='Multivariate', color='green')
axes[2].axhline(0.5, color='black', linestyle='-', alpha=0.5, label='Threshold')
for cp in true_changepoints:
    axes[2].axvline(cp, color='red', linestyle='--', alpha=0.7)
axes[2].set_title('Multivariate Changepoint Detection Probabilities')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Probability')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gpu_changepoint_detection_example.png', dpi=150, bbox_inches='tight')
plt.show()

print("Example completed! Plot saved as 'gpu_changepoint_detection_example.png'")
```

## Memory Management

For large datasets, proper memory management is crucial:

```python
# Monitor GPU memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

    # Clear cache if needed
    torch.cuda.empty_cache()
    print("GPU memory cache cleared")

    # For extremely large datasets, consider processing in chunks
    def chunked_detection(data, chunk_size=10000):
        """Process large datasets in chunks to manage memory."""
        n_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)

        all_probs = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data))
            chunk = data[start_idx:end_idx]

            # Process chunk
            _, chunk_probs = online_changepoint_detection(chunk, hazard_func, online_likelihood)
            all_probs.append(chunk_probs.cpu())  # Move to CPU to save GPU memory

            # Clear GPU cache between chunks
            torch.cuda.empty_cache()

        return torch.cat(all_probs)

    print("Example of chunked processing for large datasets completed")
```

## Best Practices

### When to Use GPU Acceleration

GPU acceleration is most beneficial for:

- **Large datasets** (>1000 time points)
- **Multivariate data** (multiple dimensions)
- **Multiple runs** or hyperparameter tuning
- **Real-time applications** requiring low latency
- **Batch processing** of multiple time series

### Memory Optimization Tips

1. **Use appropriate data types**: Float32 instead of Float64 when precision allows
2. **Clear cache regularly**: Use `torch.cuda.empty_cache()` between operations
3. **Process in chunks**: For very large datasets, process data in smaller chunks
4. **Monitor memory usage**: Use `torch.cuda.memory_allocated()` to track usage

### Performance Tips

1. **Minimize CPU-GPU transfers**: Keep data on GPU throughout the pipeline
2. **Use batch operations**: Process multiple time series simultaneously
3. **Warm up the GPU**: Run a small example first to initialize CUDA kernels
4. **Profile your code**: Use PyTorch profiler to identify bottlenecks

### Device Selection

```python
# Automatic device selection (recommended)
device = get_device()

# Manual device selection
device = torch.device('cuda:0')  # Specific GPU
device = torch.device('cpu')     # Force CPU

# Check device capabilities
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
```

### Error Handling

```python
try:
    # GPU computation
    data_gpu = data.to('cuda')
    result = online_changepoint_detection(data_gpu, hazard_func, likelihood)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU out of memory, falling back to CPU")
        torch.cuda.empty_cache()
        data_cpu = data.to('cpu')
        likelihood_cpu = StudentT(device='cpu')
        result = online_changepoint_detection(data_cpu, hazard_func, likelihood_cpu)
    else:
        raise e
```

## Complete Example Script

Here's a complete script that demonstrates all the concepts above:

```python
#!/usr/bin/env python3
"""
Complete GPU acceleration example for Bayesian changepoint detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial

# Import the library
from bayesian_changepoint_detection import (
    online_changepoint_detection,
    offline_changepoint_detection,
    constant_hazard,
    const_prior,
    get_device,
    get_device_info
)
from bayesian_changepoint_detection.online_likelihoods import StudentT, MultivariateT
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT

def main():
    # Check device availability
    print("=== Device Information ===")
    print(get_device_info())

    device = get_device()
    print(f"Selected device: {device}")

    # Generate test data
    print("\n=== Generating Test Data ===")
    torch.manual_seed(42)

    # Simple univariate data
    data = torch.cat([
        torch.randn(200) + 0,
        torch.randn(200) + 3,
        torch.randn(200) + 0,
        torch.randn(200) + -2,
        torch.randn(200) + 1,
    ])

    print(f"Generated data shape: {data.shape}")

    # Move to device
    data_device = data.to(device)

    # Setup models
    hazard_func = partial(constant_hazard, 250)
    likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device=device)

    # Run detection
    print("\n=== Running Changepoint Detection ===")
    start_time = time.time()

    run_length_probs, changepoint_probs = online_changepoint_detection(
        data_device, hazard_func, likelihood
    )

    end_time = time.time()
    print(f"Detection completed in {end_time - start_time:.3f} seconds")

    # Find changepoints
    detected = torch.where(changepoint_probs > 0.5)[0].cpu().numpy()
    print(f"Detected changepoints at: {detected}")

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(data.cpu().numpy())
    plt.title('Time Series Data')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(changepoint_probs.cpu().numpy())
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
    plt.title('Changepoint Probabilities')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gpu_example_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nExample completed successfully!")
    print("Result plot saved as 'gpu_example_result.png'")

if __name__ == "__main__":
    main()
```

This guide provides everything you need to effectively use GPU acceleration with the Bayesian Changepoint Detection library. For more examples, see the `examples/` directory in the repository.
