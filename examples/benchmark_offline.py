"""
Benchmark offline changepoint detection on CPU.

Reproduces the scenario from GitHub issue #47 (1000 data points), plus
smaller sizes, and reports wall-clock seconds per phase.

Run:  python examples/benchmark_offline.py
"""

import time
from functools import partial

import torch

from bayesian_changepoint_detection.bayesian_models import (
    offline_changepoint_detection,
)
from bayesian_changepoint_detection.offline_likelihoods import StudentT
from bayesian_changepoint_detection.priors import const_prior


def make_data(n, segments=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    means = torch.linspace(-4, 6, segments)
    length = n // segments
    parts = [
        torch.randn(length, generator=generator, dtype=torch.float64) + mean
        for mean in means
    ]
    return torch.cat(parts)


def main():
    print(f"torch {torch.__version__}, CPU threads: {torch.get_num_threads()}")
    print(f"{'n':>6} {'seconds':>9}   detected changepoints (prob > 0.5)")
    for n in (250, 500, 1000):
        data = make_data(n)
        prior = partial(const_prior, p=1.0 / (len(data) + 1))
        likelihood = StudentT(device="cpu")

        start = time.perf_counter()
        _, _, Pcp = offline_changepoint_detection(
            data, prior, likelihood, truncate=-40, device="cpu"
        )
        elapsed = time.perf_counter() - start

        probs = torch.exp(Pcp).sum(0)
        detected = torch.where(probs > 0.5)[0].tolist()
        print(f"{n:>6} {elapsed:>9.2f}   {detected}")


if __name__ == "__main__":
    main()
