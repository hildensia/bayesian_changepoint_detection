#!/usr/bin/env python3
"""
GPU acceleration example for Bayesian changepoint detection.

This example demonstrates the performance benefits of using GPU acceleration
with the PyTorch-based implementation.
"""

import time
from functools import partial

import torch

# Import the refactored modules
from bayesian_changepoint_detection import (
    get_device,
    get_device_info,
    online_changepoint_detection,
)
from bayesian_changepoint_detection.generate_data import (
    generate_mean_shift_example,
    generate_multivariate_normal_time_series,
)
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.online_likelihoods import MultivariateT, StudentT


def benchmark_univariate(data_length=1000, num_runs=3):
    """Benchmark univariate changepoint detection on CPU vs GPU."""
    print(f"Benchmarking Univariate Detection (data length: {data_length})")
    print("-" * 50)

    # Generate test data
    torch.manual_seed(42)
    partition, data = generate_mean_shift_example(
        num_segments=5,
        segment_length=data_length // 5,
        shift_magnitude=2.0,
        device="cpu",  # Start on CPU
    )

    print(f"Generated {len(data)} data points")

    # Benchmark on CPU
    print("Testing CPU performance...")
    cpu_times = []

    for run in range(num_runs):
        # Move data to CPU and create CPU likelihood
        data_cpu = data.to("cpu")
        hazard_func = partial(constant_hazard, 250)
        likelihood_cpu = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cpu")

        start_time = time.time()
        R, map_run_lengths = online_changepoint_detection(
            data_cpu.squeeze(), hazard_func, likelihood_cpu, device="cpu"
        )
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
        print(f"  Run {run + 1}: {cpu_time:.3f}s")

    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    print(f"Average CPU time: {avg_cpu_time:.3f}s")

    # Benchmark on GPU (if available)
    device_info = get_device_info()
    if device_info["cuda_available"]:
        print("\nTesting GPU performance...")
        gpu_times = []

        for run in range(num_runs):
            # Move data to GPU and create GPU likelihood
            data_gpu = data.to("cuda")
            hazard_func = partial(constant_hazard, 250)
            likelihood_gpu = StudentT(
                alpha=0.1, beta=0.01, kappa=1, mu=0, device="cuda"
            )

            # Warm up GPU
            if run == 0:
                R_warmup, _ = online_changepoint_detection(
                    data_gpu.squeeze(), hazard_func, likelihood_gpu, device="cuda"
                )
                torch.cuda.synchronize()  # Ensure GPU is ready

            start_time = time.time()
            R, map_run_lengths = online_changepoint_detection(
                data_gpu.squeeze(), hazard_func, likelihood_gpu, device="cuda"
            )
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = time.time() - start_time
            gpu_times.append(gpu_time)
            print(f"  Run {run + 1}: {gpu_time:.3f}s")

        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        print(f"Average GPU time: {avg_gpu_time:.3f}s")

        speedup = avg_cpu_time / avg_gpu_time
        print(f"\nSpeedup: {speedup:.2f}x")

        # Verify results are consistent
        data_cpu = data.to("cpu")
        data_gpu = data.to("cuda")

        likelihood_cpu = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cpu")
        likelihood_gpu = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cuda")

        R_cpu, cp_cpu = online_changepoint_detection(
            data_cpu.squeeze(), hazard_func, likelihood_cpu, device="cpu"
        )
        R_gpu, cp_gpu = online_changepoint_detection(
            data_gpu.squeeze(), hazard_func, likelihood_gpu, device="cuda"
        )

        # Check if results are close (allowing for small numerical differences)
        cp_gpu_cpu = cp_gpu.to("cpu")
        max_diff = torch.max(torch.abs(cp_cpu - cp_gpu_cpu).float()).item()
        print(f"Maximum difference between CPU and GPU results: {max_diff:.2e}")

    else:
        print("\nGPU not available - skipping GPU benchmark")

    print()


def benchmark_multivariate(dims=5, data_length=500, num_runs=3):
    """Benchmark multivariate changepoint detection on CPU vs GPU."""
    print(f"Benchmarking Multivariate Detection (dims: {dims}, length: {data_length})")
    print("-" * 60)

    # Generate test data
    torch.manual_seed(42)
    partition, data = generate_multivariate_normal_time_series(
        num_segments=3,
        dims=dims,
        min_length=data_length // 3,
        max_length=data_length // 3 + 20,
        device="cpu",
    )

    print(f"Generated {data.shape[0]} x {data.shape[1]} data points")

    # Benchmark on CPU
    print("Testing CPU performance...")
    cpu_times = []

    for run in range(num_runs):
        data_cpu = data.to("cpu")
        hazard_func = partial(constant_hazard, 200)
        likelihood_cpu = MultivariateT(dims=dims, device="cpu")

        start_time = time.time()
        R, map_run_lengths = online_changepoint_detection(
            data_cpu, hazard_func, likelihood_cpu, device="cpu"
        )
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
        print(f"  Run {run + 1}: {cpu_time:.3f}s")

    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    print(f"Average CPU time: {avg_cpu_time:.3f}s")

    # Benchmark on GPU (if available)
    device_info = get_device_info()
    if device_info["cuda_available"]:
        print("\nTesting GPU performance...")
        gpu_times = []

        for run in range(num_runs):
            data_gpu = data.to("cuda")
            hazard_func = partial(constant_hazard, 200)
            likelihood_gpu = MultivariateT(dims=dims, device="cuda")

            # Warm up GPU
            if run == 0:
                R_warmup, _ = online_changepoint_detection(
                    data_gpu, hazard_func, likelihood_gpu, device="cuda"
                )
                torch.cuda.synchronize()

            start_time = time.time()
            R, map_run_lengths = online_changepoint_detection(
                data_gpu, hazard_func, likelihood_gpu, device="cuda"
            )
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            gpu_times.append(gpu_time)
            print(f"  Run {run + 1}: {gpu_time:.3f}s")

        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        print(f"Average GPU time: {avg_gpu_time:.3f}s")

        speedup = avg_cpu_time / avg_gpu_time
        print(f"\nSpeedup: {speedup:.2f}x")

    else:
        print("\nGPU not available - skipping GPU benchmark")

    print()


def memory_usage_demo():
    """Demonstrate memory usage on GPU."""
    device_info = get_device_info()
    if not device_info["cuda_available"]:
        print("GPU not available - skipping memory usage demo")
        return

    print("GPU Memory Usage Demonstration")
    print("-" * 40)

    def print_memory_stats():
        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        cached = torch.cuda.memory_reserved() / (1024**2)  # MB
        print(f"  Allocated: {allocated:.1f} MB, Cached: {cached:.1f} MB")

    print("Initial memory usage:")
    print_memory_stats()

    # Create progressively larger datasets
    sizes = [100, 500, 1000, 2000]

    for size in sizes:
        print(f"\nProcessing dataset of size {size}:")

        # Generate data
        torch.manual_seed(42)
        partition, data = generate_mean_shift_example(
            num_segments=4, segment_length=size // 4, device="cuda"
        )

        print("  After data generation:")
        print_memory_stats()

        # Run detection
        hazard_func = partial(constant_hazard, 250)
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cuda")

        R, map_run_lengths = online_changepoint_detection(
            data.squeeze(), hazard_func, likelihood, device="cuda"
        )

        print("  After detection:")
        print_memory_stats()

        # Clean up
        del data, R, map_run_lengths, likelihood
        torch.cuda.empty_cache()

        print("  After cleanup:")
        print_memory_stats()


def device_switching_demo():
    """Demonstrate switching between devices."""
    print("Device Switching Demonstration")
    print("-" * 35)

    # Generate data
    torch.manual_seed(42)
    partition, data = generate_mean_shift_example(
        num_segments=3,
        segment_length=100,
        device="cpu",  # Start on CPU
    )

    print(f"Initial data device: {data.device}")

    # Process on CPU
    print("\nProcessing on CPU...")
    hazard_func = partial(constant_hazard, 200)
    likelihood_cpu = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cpu")

    start_time = time.time()
    R_cpu, cp_cpu = online_changepoint_detection(
        data.squeeze(), hazard_func, likelihood_cpu, device="cpu"
    )
    cpu_time = time.time() - start_time
    print(f"CPU processing time: {cpu_time:.3f}s")

    # Switch to GPU if available
    device_info = get_device_info()
    if device_info["cuda_available"]:
        print("\nSwitching to GPU...")

        # Move data to GPU
        data_gpu = data.to("cuda")
        print(f"Data moved to: {data_gpu.device}")

        # Create GPU likelihood
        likelihood_gpu = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cuda")

        start_time = time.time()
        R_gpu, cp_gpu = online_changepoint_detection(
            data_gpu.squeeze(), hazard_func, likelihood_gpu, device="cuda"
        )
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU processing time: {gpu_time:.3f}s")

        # Compare results
        cp_gpu_cpu = cp_gpu.to("cpu")
        max_diff = torch.max(torch.abs(cp_cpu - cp_gpu_cpu).float()).item()
        print(f"Maximum difference between devices: {max_diff:.2e}")

        # Move results back to CPU for further processing
        print("\nMoving results back to CPU...")
        final_results = {"R": R_gpu.to("cpu"), "map_run_lengths": cp_gpu.to("cpu")}
        print(f"Results device: {final_results['R'].device}")

    else:
        print("\nGPU not available - staying on CPU")

    print()


def main():
    """Run GPU acceleration examples and benchmarks."""
    print("GPU Acceleration Demo for Bayesian Changepoint Detection")
    print("=" * 65)

    # Display device information
    device = get_device()
    device_info = get_device_info()

    print(f"Default device: {device} (accelerators are opt-in)")
    print(f"device='auto' would pick: {get_device('auto')}")
    print(f"Available devices: {device_info['devices']}")
    print(f"CUDA available: {device_info['cuda_available']}")
    if device_info["cuda_available"]:
        print(f"CUDA device count: {device_info['device_count']}")
        print(f"Current CUDA device: {device_info['current_device']}")
    print()

    # Run benchmarks and demos
    benchmark_univariate(data_length=1000, num_runs=3)
    benchmark_multivariate(dims=3, data_length=300, num_runs=3)

    memory_usage_demo()
    print()

    device_switching_demo()

    print("=" * 65)
    print("GPU acceleration demo completed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
