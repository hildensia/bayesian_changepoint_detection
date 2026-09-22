# 📈 Bayesian Changepoint Detection
<!-- --8<-- [start:intro] -->

Find the points where a time series changes regime, with calibrated posterior
probabilities instead of a threshold. Online (Adams & MacKay 2007) and offline
(Fearnhead 2006) Bayesian changepoint detection on PyTorch tensors, with
conjugate Normal-Gamma and Normal-Wishart likelihoods for univariate and
multivariate series.

[![CI](https://github.com/hildensia/bayesian_changepoint_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/hildensia/bayesian_changepoint_detection/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/bayesian-changepoint.svg)](https://pypi.org/project/bayesian-changepoint/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/LICENSE)

<!-- --8<-- [end:intro] -->

## ✨ Features
<!-- --8<-- [start:features] -->

- 🔭 **Online detection**: the run-length posterior after every observation (Adams & MacKay 2007), for streams and for measuring how quickly a change would have been noticed
- 🔍 **Offline detection**: the exact posterior probability of a changepoint at every position given the whole series (Fearnhead 2006)
- 🎯 **Calibrated outputs**: probabilities you can threshold, MAP segment starts, and the single most probable segmentation (`viterbi_changepoints`)
- 📐 **Conjugate likelihoods**: Student-t predictive for univariate data (unknown mean and variance), multivariate-t for vector data (unknown mean and covariance), independent-features and covariance-only variants, and Gamma-Poisson for counts
- 🧮 **Verified mathematics**: closed forms checked against `scipy` and against exhaustive enumeration of segmentations; every pinned number in the test suite says where it comes from
- ⚡ **Vectorized recursions**: both detectors are O(T²) with the inner work on tensors, not Python loops; 1 000 points offline in under 4 s on a laptop CPU
- 🖥️ **Runs where your tensors are**: CPU, CUDA or Apple MPS through one `device` argument, with measured guidance on when an accelerator is *not* worth it
- 🪶 **One dependency**: `torch`; NumPy, SciPy and Matplotlib are only needed for the tests and examples

<!-- --8<-- [end:features] -->

## 🚀 Quick Start

### Installation
<!-- --8<-- [start:install] -->

```bash
pip install bayesian-changepoint
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv add bayesian-changepoint
```

The import name is `bayesian_changepoint_detection`, whatever the
distribution is called:

```python
import bayesian_changepoint_detection
```

To run the examples and notebooks, add the `plot` extra
(`pip install "bayesian-changepoint[plot]"`, or `".[plot]"` from a clone).

#### Package names

`bayesian-changepoint` is the distribution name from 1.1.0 on. The same
project was published before as **`bayescd`** (0.4, April 2022) and, earlier,
as **`bayesian-changepoint-detection`** (0.2.dev1). Both are frozen at those
releases and neither gets updates; if you have one installed, replace it:

```bash
pip uninstall bayescd bayesian-changepoint-detection
pip install bayesian-changepoint
```

The code is the same project and lives in the same repository; only the name
on PyPI changed. 1.1.0 is a rewrite on PyTorch relative to 0.4 and changes
the online API relative to 1.0.x — the
[CHANGELOG](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/CHANGELOG.md)
lists every breaking change.

### System Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher (installed automatically). For a CUDA build of
  PyTorch, install it first following <https://pytorch.org/get-started/locally/>;
  the CPU build is enough for everything in this README.

<!-- --8<-- [end:install] -->

## 📖 Usage
<!-- --8<-- [start:usage] -->

### Online detection

The online detector processes the series one point at a time and keeps the
posterior over the *run length*, the number of observations since the last
change. Two helpers turn that posterior into changepoints.

```python
from functools import partial

import torch

from bayesian_changepoint_detection import (
    StudentT,
    changepoint_probabilities,
    constant_hazard,
    get_map_changepoints,
    online_changepoint_detection,
)

torch.manual_seed(42)
data = torch.cat([
    torch.randn(50) + 0,  # first segment: mean 0
    torch.randn(50) + 3,  # second segment: mean 3
    torch.randn(50) + 0,  # third segment: mean 0
])

hazard = partial(constant_hazard, 250)  # prior: one change every ~250 points
likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)  # unknown mean and variance

R, map_run_lengths = online_changepoint_detection(data, hazard, likelihood)

# Index of the first point of each new segment on the MAP run-length path.
# min_separation merges starts closer than that many points when the
# posterior hesitates between neighbors.
print(get_map_changepoints(R, min_separation=10))  # tensor([ 50, 100])

# Or a probability per position, judged `lag` observations later.
probs = changepoint_probabilities(R, lag=10)  # probs[t] refers to data index t
print(torch.where(probs[1:] > 0.5)[0] + 1)  # tensor([ 50, 100])
```

`R[r, t]` is `P(run length = r | first t observations)`. Why not simply
threshold `R[0, :]`? Under a constant hazard the posterior probability of run
length 0 is the hazard rate at every step, whatever the data say; the
evidence for a change at `t` shows up in the *following* columns, as mass at
run length `k` in column `t + k`. `changepoint_probabilities` reads exactly
that. `viterbi_changepoints(data, hazard, likelihood)` returns the single
most probable run-length path instead, i.e. the MAP segmentation.

### Streaming

`online_changepoint_detection` needs the whole series and returns the
`(T+1)²` matrix `R`. For a stream of unknown length, feed observations one
at a time to `OnlineChangepointDetector`; it keeps only the current
run-length posterior. With `max_run_length` the memory and the time per
observation stay bounded: run lengths above the bound are dropped and the
posterior renormalized, which is exact until the bound is reached and a
close approximation afterwards when segments are shorter than the bound.

```python
from bayesian_changepoint_detection import OnlineChangepointDetector

detector = OnlineChangepointDetector(
    hazard, StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0), max_run_length=500
)
starts = []
for x in data:  # any iterable: a socket, a file, a generator
    detector.update(x)
    # P(a segment started 10 observations ago), as changepoint_probabilities
    if detector.t > 10 and detector.changepoint_probability(lag=10) > 0.5:
        starts.append(detector.t - 10)
print(starts)  # [50, 100]
```

Without `max_run_length` each posterior equals the corresponding column of
`R` from `online_changepoint_detection` (up to float32 rounding).

### Offline detection

The offline detector sees the whole series and returns, for every position,
the posterior probability that a segment ends there. It is usually sharper
than the online detector; use it for retrospective analysis.

```python
from bayesian_changepoint_detection import const_prior, offline_changepoint_detection
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT

prior = partial(const_prior, p=1 / (len(data) + 1))  # flat prior on segment length
Q, P, changepoint_log_probs = offline_changepoint_detection(data, prior, OfflineStudentT())

changepoint_probs = torch.exp(changepoint_log_probs).sum(0)  # P(a segment ends at t)
print(torch.where(changepoint_probs > 0.5)[0])  # tensor([49, 99])
```

The two detectors use different index conventions: online reports the first
point of the new segment (50), offline the last point of the old one (49).
See the [FAQ](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/README.md#-faq).

### Multivariate data

Pass a `[T, d]` tensor, one row per observation, and a multivariate
likelihood; everything else is the same. All three detectors check their
input first: data must be `[T]` or `[T, d]`, non-empty, real and finite,
and match the likelihood's `dims`. A transposed `[d, T]` tensor is rejected
with a hint rather than read as `d` observations.

```python
from bayesian_changepoint_detection import MultivariateT

dims = 3
mv_data = torch.cat([
    torch.randn(50, dims) + torch.tensor([0.0, 0.0, 0.0]),
    torch.randn(50, dims) + torch.tensor([2.0, -1.0, 1.0]),
    torch.randn(50, dims) + torch.tensor([0.0, 0.0, 0.0]),
])

R, _ = online_changepoint_detection(mv_data, hazard, MultivariateT(dims=dims))
print(get_map_changepoints(R, min_separation=10))  # tensor([ 48, 100])
```

The first start lands two points early on this draw: the lag-10 posterior
puts 0.53 on 48, 0.13 on 49 and 0.26 on 50, and the MAP path takes the
mode. Read `changepoint_probabilities` when the exact position matters.

### Devices

Every likelihood and both detectors take a `device` argument. Selection is
automatic (CUDA, then MPS, then CPU); pass `device="cpu"` to the likelihood
and the detector to opt out. On a laptop the CPU is the faster choice for
the online detector (measured: 6–30x faster than MPS), and the offline
detector always runs on the CPU under MPS because it needs float64. How the
argument is resolved, what has been measured, how to time your own workload
and how much memory the tables need: [docs/devices.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/docs/devices.md).

### API at a glance

| Function | Returns |
|---|---|
| `online_changepoint_detection(data, hazard, likelihood)` | `R` (run-length posterior, `[T+1, T+1]`) and the MAP run length after each point |
| `changepoint_probabilities(R, lag)` | `P(a new segment started at t)`, judged `lag` observations later |
| `get_map_changepoints(R, min_separation=1)` | indices where the MAP run-length path starts a new segment |
| `viterbi_changepoints(data, hazard, likelihood)` | the single most probable run-length path and its segment starts |
| `compute_run_length_posterior(data, hazard, likelihood)` | just `R`, for code that only wants the posterior |
| `OnlineChangepointDetector(hazard, likelihood, max_run_length=None)` | streaming detector: `update(x)`, `run_length_posterior`, `map_run_length`, `changepoint_probability(lag)` |
| `offline_changepoint_detection(data, prior, likelihood)` | `Q` (log evidence), `P` (segment log likelihoods), `Pcp` (log probability of the j-th changepoint at t) |
| `constant_hazard(lam, r)` | hazard `1 / lam` for every run length |
| `const_prior`, `geometric_prior`, `negative_binomial_prior` | log prior on segment length for the offline detector |
| `online_likelihoods.StudentT`, `online_likelihoods.MultivariateT` | online conjugate models (Normal-Gamma, Normal-Wishart) |
| `offline_likelihoods.StudentT`, `MultivariateT`, `IndependentFeaturesLikelihood`, `FullCovarianceLikelihood` | offline segment marginal likelihoods |
| `online_likelihoods.Poisson`, `offline_likelihoods.Poisson` | count data: Gamma-Poisson, negative-binomial predictive |
| `get_device`, `get_device_info`, `to_tensor` | device helpers |

All public functions have NumPy-style docstrings with the formulas and the
paper they come from.

<!-- --8<-- [end:usage] -->

## 🏗️ Architecture

```text
bayesian_changepoint_detection/
├── __init__.py             # Public API and __version__ (from package metadata)
├── bayesian_models.py      # The two detectors, viterbi_changepoints, and the R helpers
├── streaming.py            # OnlineChangepointDetector: the online recursion one observation at a time
├── online_likelihoods.py   # Online StudentT and MultivariateT: per-run-length predictive densities
├── offline_likelihoods.py  # Offline StudentT, MultivariateT, IndependentFeatures, FullCovariance: segment marginals
├── priors.py               # const_prior, geometric_prior, negative_binomial_prior (segment-length priors)
├── hazard_functions.py     # constant_hazard
├── device.py               # get_device, get_device_info, to_tensor, ensure_tensor
└── generate_data.py        # Synthetic series with known changepoints, for tests and examples
```

Supporting directories: `tests/` (the suite, see below), `examples/` (scripts
and two notebooks, run in CI), `docs/` (pages whose code blocks are executed
by the tests).

## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/hildensia/bayesian_changepoint_detection.git
cd bayesian_changepoint_detection

# Install with development dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# ...or, without uv:  python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# Install pre-commit hooks (ruff lint + format on staged files)
pre-commit install
```

### Running Tests

```bash
# Run all tests (about 15 s on a CPU)
pytest

# Only the tests that check the mathematics against independent references
pytest -m math

# Only the tests that pin current behavior (contracts, edge cases, devices, goldens)
pytest -m behavior

# With coverage
pytest --cov=bayesian_changepoint_detection --cov-report=term-missing

# One file
pytest tests/test_online_detection.py -v
```

Every test carries exactly one of the markers `math` and `behavior`;
collection fails otherwise. Tests pass `device="cpu"` explicitly, because
device selection is automatic and the suite is much slower on an accelerator.
The Python blocks in this README and in `docs/` are executed as part of the
suite.

### Code Quality

```bash
# Lint with ruff
ruff check .

# Format code
ruff format .

# Type checking (configured, advisory: not enforced in CI)
mypy bayesian_changepoint_detection
```

`ruff check` and `ruff format --check` are enforced in CI, together with the
test suite on Python 3.9–3.13, the example scripts, and a build job that
installs the wheel into a clean environment. See
[CONTRIBUTING.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/CONTRIBUTING.md) for the workflow and the review process.

### Building

```bash
# Build sdist and wheel
python -m build

# Check the metadata PyPI will see
twine check --strict dist/*
```

## 📊 Example Output

`examples/simple_example.py` runs both detectors on a 150-point series with
changes at 50 and 100 and saves a figure:

```text
============================================================
Bayesian Changepoint Detection - Simple Example
============================================================
Generated data with 150 points
True changepoints at: [50, 100]
Running online changepoint detection...
✓ Online detection completed
  Segment starts on the MAP path: [50, 100]
  Max lag-10 changepoint probability (t > 0): 0.9117
Running offline changepoint detection...
✓ Offline detection completed
  Max changepoint probability: 0.9322

Detected changepoints:
  Online method: [50, 100]...
  Offline method: [49, 99]...
Creating visualization...
✓ Visualization saved as 'changepoint_detection_results.png'

============================================================
✅ Example completed successfully!
============================================================
```

Other scripts in `examples/`: `basic_usage.py` (400 points, four segments,
both detectors), `multivariate_example.py`, `gpu_acceleration.py` (device
selection and CPU/GPU comparison), `benchmark_offline.py` (offline timing at
several lengths), and the notebooks `Example_Code.ipynb` and
`Multivariate_Example.ipynb`. The scripts run in CI on every push.

## ⚡ Performance
<!-- --8<-- [start:performance] -->

Both algorithms are O(T²) in the series length: the offline recursion is
vectorized per start point (one `pdf_rows` call gives the likelihood of every
segment starting there), the online recursion over run lengths at each step.
Memory is also O(T²): the run-length posterior `R` is `(T+1)²` float32, the
offline tables about `16 T²` bytes (see [docs/devices.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/docs/devices.md#memory)).
`OnlineChangepointDetector` with `max_run_length=K` is O(K) in memory and
time per observation instead.

Measured on an Apple M-series laptop, CPU, 4 threads, PyTorch 2.14:

| Workload | Time |
|---|---|
| Offline `StudentT`, 1 000 points, `const_prior`, exact sum | 3.6 s (147 s before the vectorized likelihood of 1.1.0, same changepoints) |
| Online `StudentT`, 1 000 points | 0.16 s |
| Online `StudentT`, 5 000 points | 1.7 s |
| Online `MultivariateT`, 10-D, 1 000 points | 0.56 s |
| `OnlineChangepointDetector`, `StudentT`, 20 000 points, `max_run_length=500` | 2.6 s (128 µs per point, flat) |

Accelerators: see the FAQ; MPS is slower than the CPU on all of these, CUDA
is unmeasured (issue #43). Only measured numbers appear in this README.

<!-- --8<-- [end:performance] -->

## ❓ FAQ
<!-- --8<-- [start:faq] -->

### Which detector should I use, online or offline?

`online_changepoint_detection` (Adams & MacKay 2007) processes the series one
point at a time and, after each point, gives the posterior over how long the
current segment has lasted. Use it for streams, or when you want to know how
quickly a change would have been noticed. `offline_changepoint_detection`
(Fearnhead 2006) sees the whole series and returns the posterior probability
of a changepoint at each position, using data on both sides of it. Use it for
retrospective analysis; it is usually sharper. Both cost O(T²).

### The two detectors report the same change at indices one apart. Why?

Different conventions, both documented in the docstrings:

- Online (`get_map_changepoints`, `changepoint_probabilities`,
  `viterbi_changepoints`): the index of the **first point of the new
  segment**. A series whose first 80 points come from one regime reports 80.
- Offline (`Pcp[j, t]`, and `torch.exp(Pcp).sum(0)[t]`): the probability that
  a segment **ends at `t`**, i.e. the last point of the old regime. The same
  series reports 79.

So `offline index + 1 == online index`.

### Does the scale of my data matter? (issue #34)

Yes. The priors are on the mean and variance of the data, so their
hyperparameters have units, and rescaling the data without rescaling them
changes the model. For the univariate Normal-Gamma model (online `StudentT`
with `alpha, beta, kappa, mu`; offline `StudentT` with `alpha0, beta0,
kappa0, mu0`):

| parameter | meaning | units |
|---|---|---|
| `mu` | prior mean of a segment | data units |
| `kappa` | how many observations the prior mean is worth | none |
| `alpha` | half the number of observations the variance prior is worth | none |
| `beta` | `alpha` times the prior guess of the variance | data units² |

Multiplying the data by `c` is equivalent to using `mu * c` and `beta * c²`
with `kappa` and `alpha` unchanged. With `beta / alpha` far from the actual
within-segment variance, or `mu` far from the data, the first points of
every segment look surprising and the detector over- or under-reacts.

Practical choices: standardize the data (subtract a typical level, divide by
a typical within-segment standard deviation, ideally estimated on a
calibration window rather than on the whole series), or set `mu` to the
expected level and `beta = alpha * expected_variance`. The values in the
examples (`alpha=0.1, beta=0.01, kappa=1, mu=0`) encode "around zero,
variance about 0.1, but I am not sure": with `df = 2 * alpha = 0.2` the
predictive is extremely heavy-tailed, which is why they still work on
roughly unit-scale data.

The multivariate classes work the same way but parametrize the prior on
the covariance differently. Online `MultivariateT` takes `scale`, the
Wishart scale `W` on the *precision*: to encode a prior covariance `C` pass
`scale = inv(C) / dof` (default `I / dof`, unit prior covariance). Offline
`MultivariateT` takes `Psi0`, the inverse-Wishart scale on the *covariance*
side (`Psi0 = inv(W)`): the same prior covariance `C` is `Psi0 = dof0 * C`,
and the default `dof0 * I` is the same unit prior covariance as online.
`mu`/`mu0` are in data units in both.

### How do I make the detector more or less sensitive? (issue #31)

In order of importance:

1. **The hazard, i.e. the expected segment length.** `constant_hazard(lam)`
   puts prior probability `1 / lam` on a change at every step. Larger `lam`
   means fewer detections, more confidence needed, slightly longer delay;
   smaller `lam` means more, earlier, and more false alarms. This is the main
   knob and it is about the data, not the model: set it near the segment
   length you expect.
2. **How much you trust the prior versus the first points of a new segment.**
   `kappa` (for the mean) and `alpha` (for the variance) act as pseudo-counts.
   Small values let a few points establish a new regime quickly; larger values
   make the detector wait for more evidence. `beta` and `mu` should describe
   the data (previous question) rather than be used as sensitivity knobs.
3. **How you read the output.** `changepoint_probabilities(R, lag)` trades
   delay for confidence: a larger `lag` gives a more decisive probability,
   `lag` observations later. `get_map_changepoints(R, min_separation=k)`
   drops starts closer than `k` points to an earlier one, for when the
   posterior hesitates between neighboring points.

Offline, the equivalent of the hazard is the segment-length prior:
`const_prior(p=1/(T+1))` is the flat default; `geometric_prior(p=1/L)`
encodes an expected segment length `L`; `negative_binomial_prior` allows a
peaked length distribution. Leave `truncate` at its default: the sum is exact
and the legacy truncation can drop the dominant term.

### My data are not normally distributed. Can I still use this? (issue #36)

Every likelihood here assumes that **within a segment** the observations are
independent draws from one distribution. The Gaussian ones detect changes in
the mean and/or the (co)variance; `Poisson` detects changes in the rate of
counts:

| likelihood | within-segment model |
|---|---|
| online `StudentT`, offline `StudentT` | i.i.d. Normal, unknown mean and variance (Normal-Gamma prior) |
| online `MultivariateT` | i.i.d. multivariate Normal, unknown mean and covariance (Normal-Wishart) |
| offline `IndependentFeaturesLikelihood` | one Normal-Gamma model per dimension, independent |
| offline `MultivariateT` | i.i.d. multivariate Normal, unknown mean and covariance (Normal-Wishart) |
| online `Poisson`, offline `Poisson` | i.i.d. Poisson counts, unknown rate (Gamma prior); multivariate offline input is independent Poisson dimensions |
| offline `FullCovarianceLikelihood` | multivariate Normal with unknown covariance and **no mean parameter** (mean zero, Xuan & Murphy 2007): it detects covariance changes; segments that differ in mean are misread as scale changes, so use `MultivariateT` when means move |

When the data are not Gaussian the detector still runs, and the question is
what the misspecification does to it:

- **Heavy tails or outliers**: single extreme points look like the start of
  a new segment. The Student-t predictive already tolerates some of this;
  a larger `lam` or `kappa` helps, and so does a transform (log for positive,
  right-skewed quantities such as latencies or prices).
- **Counts**: use `online_likelihoods.Poisson` / `offline_likelihoods.Poisson`
  (non-negative integers only). Counts that vary more than a Poisson allows
  (overdispersion) will show extra changepoints; a square-root or Anscombe
  transform with `StudentT` is the alternative.
- **Bounded data**: a transform (logit for proportions) usually gets you
  close enough.
- **Autocorrelation or slow drift**: the model has no notion of dynamics
  within a segment, so a drift is reported as a sequence of small changes.
  Differencing, or modeling residuals from a trend, is the usual fix.
- **Changes in something other than mean or variance** (e.g. in
  autocorrelation) are not detected.

In short: use it when "piecewise stationary with Gaussian-ish noise" is a
reasonable description after a transform, and check on a segment you trust
that the residuals look plausible.

### Why is it slow on my laptop with a GPU?

Device selection is automatic and prefers CUDA or Apple MPS when present, but
the online recursion is a sequential loop over small tensors, and each step
on an accelerator pays a launch cost. Measured on an Apple M-series laptop
(PyTorch 2.14), CPU against MPS:

| workload | CPU | MPS |
|---|---|---|
| online `StudentT`, 1 000 points | 0.16 s | 2.5 s |
| online `StudentT`, 5 000 points | 1.7 s | 11 s |
| online `MultivariateT`, 10-D, 1 000 points | 0.56 s | 17 s |

The offline detector needs float64 and always runs on the CPU when MPS is
selected. Pass `device="cpu"` to both the likelihood and the detector unless
you have measured otherwise on your hardware; CUDA has not been benchmarked
(issue #43).

<!-- --8<-- [end:faq] -->

## 🤝 Contributing

Contributions are welcome. Please see the [Contributing Guidelines](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/CONTRIBUTING.md)
for the development setup, the conventions (including the rule that a test
pinning a number says where the number comes from) and the review process.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

### Project documentation

| Document | Contents |
| --- | --- |
| [CONTRIBUTING.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/CONTRIBUTING.md) | Development setup, conventions, releasing |
| [CHANGELOG.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/CHANGELOG.md) | Release history, including the numerical changes in 1.1.0 |
| [AGENTS.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/AGENTS.md) | Conventions for AI coding agents: the two `StudentT`s, index conventions, changing the math |
| [SECURITY.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/SECURITY.md) | How to report a vulnerability |
| [CODE_OF_CONDUCT.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/CODE_OF_CONDUCT.md) | Community standards |
| [docs/devices.md](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/docs/devices.md) | CPU, CUDA and MPS: device resolution, measurements, memory |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/LICENSE) file for details.

## 🔗 Related Resources
<!-- --8<-- [start:references] -->

- Ryan P. Adams and David J. C. MacKay (2007). *Bayesian Online Changepoint Detection*. arXiv:0710.3742. <https://arxiv.org/abs/0710.3742> — the online algorithm.
- Paul Fearnhead (2006). *Exact and Efficient Bayesian Inference for Multiple Changepoint Problems*. Statistics and Computing 16(2), 203–213. <https://doi.org/10.1007/s11222-006-8450-8> — the offline algorithm.
- Xiang Xuan and Kevin Murphy (2007). *Modeling Changing Dependency Structure in Multivariate Time Series*. ICML 2007, 1055–1062. <https://doi.org/10.1145/1273496.1273629> — the multivariate likelihoods.
- Kevin P. Murphy (2007). *Conjugate Bayesian analysis of the Gaussian distribution*. Technical note. <https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf> — the Normal-Gamma and Normal-Wishart closed forms used in the likelihoods.

<!-- --8<-- [end:references] -->

## 🙏 Acknowledgements

- **Johannes Kulick** wrote the original NumPy implementation (2014–2022), published as `bayesian-changepoint-detection` and then `bayescd`, and owns this repository.
- **Esteban Carisimo** did the PyTorch rewrite, the vectorized recursions, the verified likelihoods and the current maintenance.

### Citation
<!-- --8<-- [start:citation] -->

If you use this library in your research, please cite it (GitHub's "Cite
this repository" button reads [CITATION.cff](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/CITATION.cff)):

```bibtex
@software{bayesian_changepoint_detection,
  title   = {Bayesian Changepoint Detection: A PyTorch Implementation},
  author  = {Kulick, Johannes and Carisimo, Esteban},
  url     = {https://github.com/hildensia/bayesian_changepoint_detection},
  year    = {2026},
  version = {1.1.0}
}
```

The algorithms are due to Adams & MacKay (2007) and Fearnhead (2006); please
cite those papers as well.

<!-- --8<-- [end:citation] -->
