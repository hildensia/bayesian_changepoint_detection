"""
Run one benchmark case against one version of the library, in its own process.

The driver (``performance.py``) starts this script with ``PYTHONPATH`` pointing
at an exported copy of the version under test, passes the case as a JSON
argument, and reads one JSON line back from stdout. Keeping each case in a
fresh process means one version's imports, caches or thread pools cannot
leak into another's timings, and the peak resident memory is per case.

The adapters below paper over API differences between versions:

- ``v0.4`` is the NumPy implementation (needs numpy, scipy and ``decorator``);
- ``v1.0.0`` and ``v1.1.0`` are the first PyTorch releases;
- ``current`` is whatever checkout the driver points at.
"""

import json
import os
import platform
import resource
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workloads import (  # noqa: E402
    make_data,
    offline_starts,
    online_starts,
    score,
    starts_from_map_run_lengths,
)

STREAMING_SEGMENT = 250


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak if platform.system() == "Darwin" else peak * 1024


def _runner(version, workload, device, n_obs, dims):
    """Return ``run(data) -> starts`` for this version and workload.

    Each call builds fresh likelihood objects: online likelihoods are
    stateful and single-use, and v0.4's offline likelihood caches per data.
    """
    if version == "v0.4":
        import bayesian_changepoint_detection.bayesian_models as bm
        import bayesian_changepoint_detection.hazard_functions as hz
        import bayesian_changepoint_detection.offline_likelihoods as offl
        import bayesian_changepoint_detection.online_likelihoods as onl
        import bayesian_changepoint_detection.priors as pr

        def offline(data):
            if dims != 1:
                raise ValueError("the v0.4 adapter covers univariate offline only")
            likelihood = offl.StudentT()
            x = data[:, None]
            prior = partial(pr.const_prior, p=1.0 / (len(data) + 1))
            _, _, Pcp = bm.offline_changepoint_detection(x, prior, likelihood)
            return offline_starts(np.exp(Pcp).sum(0))

        def online(data):
            if dims == 1:
                likelihood = onl.StudentT(alpha=0.1, beta=0.1, kappa=1, mu=0)
            else:
                likelihood = onl.MultivariateT(dims=dims)
            hazard = partial(hz.constant_hazard, n_obs / 4)
            R, _ = bm.online_changepoint_detection(data, hazard, likelihood)
            return online_starts(R)

        return {"offline": offline, "online": online}[workload]

    import torch

    import bayesian_changepoint_detection as bcd
    from bayesian_changepoint_detection import offline_likelihoods as offl
    from bayesian_changepoint_detection import online_likelihoods as onl

    # 1.0.0 mixes float32 state with float64 input and fails; feed it what
    # its examples used. Later versions take float64 (and keep it).
    dtype = torch.float32 if version == "v1.0.0" else torch.float64

    def offline(data):
        x = torch.as_tensor(data, dtype=dtype)
        if dims == 1:
            likelihood = offl.StudentT(device=device)
        else:
            likelihood = offl.MultivariateT(dims=dims, device=device)
        prior = partial(bcd.const_prior, p=1.0 / (len(data) + 1))
        _, _, Pcp = bcd.offline_changepoint_detection(
            x, prior, likelihood, device=device
        )
        return offline_starts(torch.exp(Pcp).sum(0).cpu().numpy())

    def online(data):
        x = torch.as_tensor(data, dtype=dtype)
        if dims == 1:
            likelihood = onl.StudentT(
                alpha=0.1, beta=0.1, kappa=1.0, mu=0.0, device=device
            )
        else:
            likelihood = onl.MultivariateT(dims=dims, device=device)
        hazard = partial(bcd.constant_hazard, n_obs / 4)
        R, _ = bcd.online_changepoint_detection(x, hazard, likelihood, device=device)
        return online_starts(R.cpu().numpy())

    def streaming(data):
        # Bounded memory: one column of R, at most max_run_length + 1 entries.
        # The bound conditions the posterior on segments no longer than it,
        # so the streaming series has segments (250) well below it (1000).
        x = torch.as_tensor(data, dtype=dtype)
        likelihood = onl.StudentT(alpha=0.1, beta=0.1, kappa=1.0, mu=0.0, device=device)
        detector = bcd.OnlineChangepointDetector(
            partial(bcd.constant_hazard, STREAMING_SEGMENT),
            likelihood,
            max_run_length=1000,
            device=device,
        )
        map_run_lengths = np.empty(len(x), dtype=np.int64)
        for t, value in enumerate(x):
            detector.update(value)
            map_run_lengths[t] = detector.map_run_length
        return starts_from_map_run_lengths(map_run_lengths)

    return {"offline": offline, "online": online, "streaming": streaming}[workload]


def main():
    case = json.loads(sys.argv[1])
    version, workload = case["version"], case["workload"]
    n_obs, dims = case["n"], case["dims"]
    device = case.get("device", "cpu")

    if version == "v1.0.0" and device == "cpu":
        # 1.0.0's constant_hazard ignores the requested device and follows
        # the automatic choice, so on a machine with MPS the online detector
        # crashes even with device="cpu" (fixed in 1.1.0). Hide MPS so its
        # CPU path can be timed; the report says so.
        import torch

        torch.backends.mps.is_available = lambda: False
        case["mps_hidden"] = True

    if case.get("threads"):
        import torch

        torch.set_num_threads(case["threads"])

    run = _runner(version, workload, device, n_obs, dims)
    segment = STREAMING_SEGMENT if workload == "streaming" else None
    data, truth = make_data(
        n_obs, dims, seed=case.get("seed", 0), segment_length=segment
    )
    warmup, _ = make_data(40, dims, seed=12345)

    def synchronize():
        if device != "cpu" and version != "v0.4":
            import torch

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

    run(warmup)  # imports, kernel compilation, allocator warm-up
    baseline_rss = _peak_rss_bytes()

    times, starts = [], []
    budget = case.get("budget", float("inf"))
    for _ in range(case.get("repeats", 5)):
        run(warmup)  # also resets v0.4's per-data cache between timed runs
        synchronize()
        start = time.perf_counter()
        starts = run(data)
        synchronize()
        times.append(time.perf_counter() - start)
        if sum(times) > budget:
            break

    module = sys.modules.get("bayesian_changepoint_detection")
    result = {
        **case,
        "times": times,
        "peak_rss_increase_bytes": max(_peak_rss_bytes() - baseline_rss, 0),
        "detected": [int(s) for s in starts],
        "truth": truth,
        **score(starts, truth),
        "module_file": getattr(module, "__file__", None),
    }
    if version != "v0.4":
        import torch

        result["torch_threads"] = torch.get_num_threads()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
