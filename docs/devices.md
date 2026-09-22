# Devices: CPU, CUDA and Apple MPS

Every likelihood and both detectors take a `device` argument, so the library
runs unchanged on the CPU, on a CUDA GPU or on Apple's MPS backend. This page
says what that argument does, what has been measured, and how to measure on
your own hardware. Every code block below is executed by the test suite
(`tests/test_docs_code_blocks.py`), on the CPU.

**Short version:** run on the CPU unless you have timed the alternative.
On an Apple M-series laptop the CPU beats MPS by 6–30x for the online
detectors, and the offline detector runs on the CPU whenever MPS is selected
(see [What has been measured](#what-has-been-measured)). CUDA has not been
benchmarked ([issue #43](https://github.com/hildensia/bayesian_changepoint_detection/issues/43)).

## How a device is chosen

`get_device(None)` picks the first available of CUDA, MPS, CPU; every
likelihood constructor calls it, so a likelihood built without `device`
lands on the accelerator when there is one.

```python
from bayesian_changepoint_detection import get_device, get_device_info

print(get_device_info())  # cuda_available, mps_available, devices, ...
print(get_device())  # the automatic choice on this machine
print(get_device("cpu"))  # an explicit choice is returned unchanged
```

The two detectors resolve the device differently:

- `online_changepoint_detection(data, hazard, likelihood, device=None)` (and
  `viterbi_changepoints`) uses the **likelihood's** device when `device` is
  omitted. When `device` is given, the likelihood's state, the data and the
  run-length matrix `R` are all moved to it. Either way everything ends up on
  one device, so mixed CPU/GPU inputs cannot collide mid-recursion.
- `offline_changepoint_detection(data, prior, likelihood, device=None)`
  resolves `device` with `get_device` (automatic when omitted), moves the data
  there and points the likelihood at it. The recursion runs in float64. **MPS
  has no float64**, so when MPS is selected, explicitly or automatically, the
  offline detector warns and runs on the CPU.

So the one reliable way to keep a computation on the CPU is to say so on both
the likelihood and the detector:

```python
from functools import partial

import torch

from bayesian_changepoint_detection import (
    const_prior,
    constant_hazard,
    get_map_changepoints,
    offline_changepoint_detection,
    online_changepoint_detection,
)
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT
from bayesian_changepoint_detection.online_likelihoods import StudentT

torch.manual_seed(0)
data = torch.cat([torch.randn(100), torch.randn(100) + 3])

R, map_run_lengths = online_changepoint_detection(
    data, partial(constant_hazard, 100), StudentT(device="cpu"), device="cpu"
)
print("online, segment starts:", get_map_changepoints(R))
print("R lives on", R.device)

Q, P, changepoint_log_probs = offline_changepoint_detection(
    data, partial(const_prior, p=1 / 201), OfflineStudentT(device="cpu"), device="cpu"
)
changepoint_probs = torch.exp(changepoint_log_probs).sum(0)
print("offline, P(change) > 0.5 at:", torch.where(changepoint_probs > 0.5)[0] + 1)
```

To opt into an accelerator, name it. The online detector follows the
likelihood, so setting the device there is enough; the code below stays
correct on a machine without CUDA because it falls back to the CPU.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

likelihood = StudentT(device=device)
R, map_run_lengths = online_changepoint_detection(data, partial(constant_hazard, 100), likelihood)
print("R lives on", R.device)  # same device as the likelihood

# Results move back to the CPU like any tensor.
starts = get_map_changepoints(R).cpu()
print(starts)
```

The MPS and CPU results are not bit-identical (both are float32, the
reductions differ in order); on the 200-point series above `R` agrees to
4e-6 and the segment starts are the same.

## What has been measured

The online recursion is a Python loop over the observations, and each step
works on tensors with at most `t + 1` entries. On an accelerator every step
pays a kernel-launch cost that dwarfs the arithmetic unless the per-step work
is large (high dimension, long series). The numbers in the README FAQ
(["Why is it slow on my laptop with a GPU?"](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/README.md#why-is-it-slow-on-my-laptop-with-a-gpu))
were taken on an Apple M-series laptop with PyTorch 2.14, CPU against MPS:
the CPU is 16x faster on 1 000 univariate points, 6x on 5 000, and 30x on
1 000 points in 10 dimensions. The offline detector never runs on MPS.

Nobody has published CUDA timings for this code. The claims of large GPU
speedups that earlier versions of this documentation made were never
measured and have been removed. If you measure, please report the numbers
on [issue #43](https://github.com/hildensia/bayesian_changepoint_detection/issues/43)
with the hardware, PyTorch version, dtype, series length and dimension.

## Measuring on your own hardware

`examples/gpu_acceleration.py` times the online detector on the CPU and, when
CUDA is available, on the GPU, and checks that the two agree.
`examples/benchmark_offline.py` times the offline detector at several
lengths. For a quick check of your own workload, time it like this. Three
details matter: GPU work is asynchronous, so synchronize before reading the
clock; the first run includes kernel compilation, so discard it; and an
online likelihood object accumulates state as it runs, so build a fresh one
for every run rather than reusing it.

```python
import time


def time_online(data, device, repeats=3):
    hazard = partial(constant_hazard, 250)
    times = []
    for _ in range(repeats + 1):
        likelihood = StudentT(device=device)  # single-use: one per run
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        online_changepoint_detection(data, hazard, likelihood, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times[1:])  # drop the warm-up run


torch.manual_seed(0)
series = torch.cat([torch.randn(150), torch.randn(150) + 3])
print(f"cpu: {time_online(series, 'cpu'):.3f} s")
if torch.cuda.is_available():
    print(f"cuda: {time_online(series, 'cuda'):.3f} s")
```

## Memory

Both detectors keep a `T x T` table, so memory grows with the square of the
series length whatever the device:

- online: `R` is `(T + 1) x (T + 1)` float32, 4 bytes per entry;
- offline: `P` is `T x T` and the changepoint table `(T - 1) x (T - 1)`,
  both float64, 8 bytes per entry, 16 bytes per `T²` together; the
  changepoint step allocates temporaries of up to the same size again, so
  budget for roughly twice that at peak.

```python
for T in (1_000, 10_000, 50_000):
    online_bytes = 4 * (T + 1) ** 2
    offline_bytes = 8 * T**2 + 8 * (T - 1) ** 2
    print(f"T={T:>6}: online R {online_bytes / 1e9:6.2f} GB, offline tables {offline_bytes / 1e9:6.2f} GB")
```

A 10 000-point series needs 0.4 GB for `R` and 1.6 GB for the offline tables;
at 50 000 points the online detector alone needs 10 GB. There is no built-in
chunking: splitting a series and running the detector on each piece changes
the model (the prior restarts at every chunk boundary), so it is not a
transparent memory optimization. For long series use
`OnlineChangepointDetector` instead of `online_changepoint_detection`: it
keeps one column of `R` rather than the whole matrix, and with
`max_run_length=K` its memory is O(K) whatever the length of the stream
(see "Streaming" in the README).
