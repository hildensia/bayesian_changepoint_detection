# Bayesian Changepoint Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern, PyTorch-based library for Bayesian changepoint detection in time series data. This library implements both online and offline methods with GPU acceleration support for high-performance computation.

## Features

- **PyTorch Backend**: Leverages PyTorch for efficient computation and automatic differentiation
- **GPU Acceleration**: Automatic device detection with support for CUDA and Apple Silicon (MPS)
- **Online & Offline Methods**: Sequential and batch changepoint detection algorithms
- **Multiple Distributions**: Support for univariate and multivariate Student's t-distributions
- **Flexible Priors**: Constant, geometric, and negative binomial prior distributions
- **Type Safety**: Full type annotations for better development experience
- **Comprehensive Testing**: Extensive test suite with GPU testing support

## Installation

This package is published on PyPI as **`bayescd`** — the name
`bayesian-changepoint-detection` on PyPI belongs to an unrelated project. The
import name is unaffected:

```bash
pip install bayescd
```

```python
import bayesian_changepoint_detection
```

The sections below cover the supported installation methods with modern Python
package managers. Choose the one that best fits your workflow.

### Method 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. It's the recommended approach for new projects.

#### Install UV
```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### Install the package with UV
```bash
# Create a new virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install bayescd

# Or install directly with auto-managed environment
uv run python -c "import bayesian_changepoint_detection; print('Success!')"
```

#### Development installation with UV
```bash
git clone https://github.com/hildensia/bayesian_changepoint_detection.git
cd bayesian_changepoint_detection

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev]"

# Or install specific dependency groups
uv pip install -e ".[dev,docs,gpu]"
```

### Method 2: Using pip with Virtual Environments

#### Create and activate a virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Install the package
```bash
# Install from PyPI (when available)
pip install bayescd

# Or install from source
git clone https://github.com/hildensia/bayesian_changepoint_detection.git
cd bayesian_changepoint_detection
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Method 3: Using conda/mamba

```bash
# Create conda environment
conda create -n bayesian-cp python=3.9
conda activate bayesian-cp

# Install PyTorch first (recommended for better compatibility)
conda install pytorch torchvision torchaudio -c pytorch

# Install the package
pip install bayescd

# Or from source
git clone https://github.com/hildensia/bayesian_changepoint_detection.git
cd bayesian_changepoint_detection
pip install -e ".[dev]"
```

### Dependency Groups

The package defines several optional dependency groups:

- **`dev`**: Development and test tools (pytest, numpy, scipy, ruff, pre-commit, mypy, etc.)
- **`plot`**: Plotting for the examples and notebooks (matplotlib, seaborn)
- **`docs`**: Documentation generation (sphinx, numpydoc)
- **`gpu`**: GPU support (CUDA-enabled PyTorch)

The library itself depends only on PyTorch.

#### Install specific groups
```bash
# With UV
uv pip install "bayescd[dev,gpu]"

# With pip
pip install "bayescd[dev,gpu]"
```

### GPU Support

For CUDA support, ensure you have CUDA-compatible hardware and drivers, then:

#### Option 1: Install PyTorch with CUDA manually (Recommended)
```bash
# Visit https://pytorch.org/get-started/locally/ for the latest commands
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install the package
pip install bayescd
# or from source:
pip install -e .
```

#### Option 2: Install with GPU extras (May install CPU-only PyTorch)
```bash
# Note: The [gpu] extra attempts to install torch[cuda], but this may not always
# install the GPU version correctly. Option 1 is more reliable.

# UV
uv pip install "bayescd[gpu]"

# pip
pip install "bayescd[gpu]"
```

#### Verify GPU Support
```bash
# Check if PyTorch can see your GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

## GPU/CUDA Acceleration

The library provides GPU acceleration for significant performance improvements. Here's a quick example:

```python
import torch
from functools import partial
from bayesian_changepoint_detection import (
    online_changepoint_detection, constant_hazard, get_map_changepoints,
)
from bayesian_changepoint_detection.online_likelihoods import StudentT

# Automatic device selection (chooses GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate sample data and move to GPU
data = torch.cat([torch.randn(100), torch.randn(100) + 3]).to(device)

# Set up GPU-enabled model
hazard_func = partial(constant_hazard, 250)
likelihood = StudentT(alpha=0.1, beta=0.01, device=device)

# Run detection on GPU
R, map_run_lengths = online_changepoint_detection(data, hazard_func, likelihood)

print("Detected changepoints:", get_map_changepoints(R))
```

**Performance note:** the online recursion is sequential, so an accelerator
only pays off when each step is large (high dimension, long series). On an
Apple M-series laptop the CPU is 6-30x faster than MPS for the cases in the
FAQ below; CUDA is unmeasured (issue #43). Device detection is automatic;
pass `device="cpu"` to opt out.

📖 **For a complete GPU guide with benchmarks, multivariate examples, and memory management tips, see**
- **[docs/gpu_offline_detection_guide.md](docs/gpu_offline_detection_guide.md)**
- **[docs/gpu_online_detection_guide.md](docs/gpu_online_detection_guide.md)**

### Verify Installation

Test your installation:

```python
import torch
from bayesian_changepoint_detection import get_device_info

# Check device availability
print(get_device_info())

# Quick test
from bayesian_changepoint_detection.generate_data import generate_mean_shift_example
partition, data = generate_mean_shift_example(3, 50)
print(f"Generated test data: {data.shape}")
```

Or run one of the examples (they plot, so they need the `plot` extra):

```bash
pip install -e ".[plot]"

# Run example from the project root
PYTHONPATH=. python examples/simple_example.py

# Run the test suite (requires the dev extra)
pip install -e ".[dev]"
pytest
```

### Development Setup

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/hildensia/bayesian_changepoint_detection.git
cd bayesian_changepoint_detection

# Option 1: Using UV (recommended)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev,docs]"

# Option 2: Using pip
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev,docs]"

# Run tests
pytest
# or if pytest is not in PATH:
python -m pytest

# Lint and format (what CI checks)
ruff check .
ruff format .

# Run type checking
mypy bayesian_changepoint_detection
```

### Troubleshooting

#### Common Issues

1. **pytest command not found**
   ```bash
   # Option 1: Use python -m pytest
   python -m pytest

   # Option 2: Ensure pytest is installed
   pip install pytest

   # Option 3: Run just the basic online-detection tests
   python -m pytest tests/test_online_detection.py
   ```

2. **PyTorch installation conflicts**
   ```bash
   # Uninstall and reinstall PyTorch
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio
   ```

2. **CUDA version mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi

   # Install matching PyTorch version from https://pytorch.org/
   ```

3. **Virtual environment issues**
   ```bash
   # Recreate virtual environment
   rm -rf venv  # or .venv
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

4. **Permission errors**
   ```bash
   # Use --user flag if you can't create virtual environments
   pip install --user bayescd
   ```

## Quick Start

### Online Changepoint Detection

```python
import torch
from functools import partial
from bayesian_changepoint_detection import (
    online_changepoint_detection,
    changepoint_probabilities,
    get_map_changepoints,
    constant_hazard,
    StudentT,
)

# Generate sample data
torch.manual_seed(42)
data = torch.cat([
    torch.randn(50) + 0,      # First segment: mean=0
    torch.randn(50) + 3,      # Second segment: mean=3
    torch.randn(50) + 0,      # Third segment: mean=0
])

# Set up the model
hazard_func = partial(constant_hazard, 250)  # Expected run length of 250
likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)

# Run online changepoint detection
R, map_run_lengths = online_changepoint_detection(data, hazard_func, likelihood)

# R[r, t] is P(run length = r | first t observations). The most likely run
# length resets to ~1 right after a change, which is what the detector reads.
# On ambiguous data the MAP path can flip between two nearby starts;
# min_separation merges starts closer than that many observations.
print("Segment starts (MAP run-length path):", get_map_changepoints(R, min_separation=10))

# Or a calibrated probability per position, judged `lag` observations later:
probs = changepoint_probabilities(R, lag=10)     # probs[t] refers to data index t
print("P(change at t) > 0.5 at:", torch.where(probs[1:] > 0.5)[0] + 1)
```

`viterbi_changepoints(data, hazard_func, likelihood)` returns the single most
probable run-length path instead of the posterior, i.e. the MAP segmentation
under the same model; its second value uses the same segment-start convention
as `get_map_changepoints`.

Why not simply threshold `R[0, :]`? Under a constant hazard the posterior
probability of run length 0 is the hazard rate at every step, whatever the
data say; the evidence for a change at `t` shows up in the *following*
columns as mass at run length `k` in column `t + k`. `changepoint_probabilities`
reads exactly that.

### Offline Changepoint Detection

```python
from bayesian_changepoint_detection import (
    offline_changepoint_detection,
    const_prior
)
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT

# Generate sample data (same as above)
data = torch.cat([
    torch.randn(50) + 0,      # First segment: mean=0
    torch.randn(50) + 3,      # Second segment: mean=3
    torch.randn(50) + 0,      # Third segment: mean=0
])

# Use offline method for batch processing
prior_func = partial(const_prior, p=1/(len(data)+1))
likelihood = OfflineStudentT()

Q, P, changepoint_log_probs = offline_changepoint_detection(
    data, prior_func, likelihood
)

# Get changepoint probabilities
changepoint_probs = torch.exp(changepoint_log_probs).sum(0)
```

### GPU Acceleration

```python
# Automatic GPU detection
device = get_device()  # Selects best available device
print(f"Using device: {device}")

# Force specific device
likelihood = StudentT(device='cuda')  # Use GPU
data_gpu = data.to('cuda')

# All computations will run on GPU
R, map_run_lengths = online_changepoint_detection(data_gpu, hazard_func, likelihood)
```

### Multivariate Data

```python
from bayesian_changepoint_detection.online_likelihoods import MultivariateT

# Generate multivariate data
dims = 3
data = torch.cat([
    torch.randn(50, dims) + torch.tensor([0, 0, 0]),
    torch.randn(50, dims) + torch.tensor([2, -1, 1]),
    torch.randn(50, dims) + torch.tensor([0, 0, 0]),
])

# Multivariate likelihood
likelihood = MultivariateT(dims=dims)

# Run detection
R, map_run_lengths = online_changepoint_detection(data, hazard_func, likelihood)
print(get_map_changepoints(R))
```

## Mathematical Background

This library implements Bayesian changepoint detection as described in:

1. **Paul Fearnhead** (2006). "Exact and Efficient Bayesian Inference for Multiple Changepoint Problems." *Statistics and Computing*, 16(2), 203-213.

2. **Ryan P. Adams and David J.C. MacKay** (2007). "Bayesian Online Changepoint Detection." *arXiv preprint arXiv:0710.3742*.

3. **Xuan Xiang and Kevin Murphy** (2007). "Modeling Changing Dependency Structure in Multivariate Time Series." *ICML*, 1055-1062.

### Key Concepts

- **Run Length**: Time since the last changepoint
- **Hazard Function**: Prior probability of a changepoint at each time step
- **Likelihood Model**: Distribution of observations within segments
- **Posterior**: Probability distribution over run lengths given data

## API Reference

### Core Functions

- `online_changepoint_detection()`: Sequential changepoint detection
- `offline_changepoint_detection()`: Batch changepoint detection

### Likelihood Models

- `StudentT`: Univariate Student's t-distribution (unknown mean and variance)
- `MultivariateT`: Multivariate Student's t-distribution

### Prior Distributions

- `const_prior()`: Uniform prior over changepoint locations
- `geometric_prior()`: Geometric distribution for inter-arrival times
- `negative_binomial_prior()`: Generalized geometric distribution

### Hazard Functions

- `constant_hazard()`: Constant probability of changepoint occurrence

### Device Management

- `get_device()`: Automatic device selection
- `to_tensor()`: Convert data to PyTorch tensors
- `get_device_info()`: Get information about available devices

## Performance

Both algorithms are O(T²) in the series length. The offline recursion is
vectorized per start point (one `pdf_rows` call gives the likelihood of every
segment starting there); the online recursion is vectorized over run lengths
at each step.

### Benchmarks

Measured on an Apple M-series laptop, CPU, 4 threads, PyTorch 2.14:

| Workload | Time |
|---|---|
| Offline `StudentT`, 1 000 points, `const_prior`, exact sum | 3.6 s (147 s before the vectorized likelihood of 1.1.0, same changepoints) |
| Online `StudentT`, 1 000 points | 0.16 s |
| Online `StudentT`, 5 000 points | 1.7 s |
| Online `MultivariateT`, 10-D, 1 000 points | 0.56 s |

Accelerators: see the FAQ; MPS is slower than the CPU on all of these, CUDA
is unmeasured.

## Examples

See the `examples/` directory for complete examples:

- `examples/simple_example.py`: online and offline detection on one series, with a figure
- `examples/basic_usage.py`: simple univariate example
- `examples/multivariate_example.py`: multivariate time series
- `examples/gpu_acceleration.py`: device selection and CPU/GPU comparison
- `examples/benchmark_offline.py`: offline detector timing at several lengths
- `examples/Example_Code.ipynb`, `examples/Multivariate_Example.ipynb`: notebook tutorials

The scripts run in CI on every push (headless, `MPLBACKEND=Agg`).

## Development

### Running Tests

#### Basic Tests
```bash
# Run the basic online-detection tests (univariate and multivariate)
python -m pytest tests/test_online_detection.py
```

#### Full Test Suite
```bash
# First, install development dependencies
pip install -e ".[dev]"

# Run all tests in the tests/ directory
pytest tests/
# or if pytest is not in PATH:
python -m pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=bayesian_changepoint_detection
# or:
python -m pytest tests/ --cov=bayesian_changepoint_detection

# Run specific test files
pytest tests/test_device.py
pytest tests/test_online_likelihoods.py

# Run GPU tests only (requires CUDA)
pytest tests/ -m gpu

# Run non-GPU tests only
pytest tests/ -m "not gpu"
```

The full test suite includes:
- Device management tests
- Online and offline likelihood tests
- Prior distribution tests
- Integration tests with regression testing
- GPU computation tests (when CUDA available)

### Code Quality

```bash
# Lint and format (CI runs both; pre-commit install runs them on each commit)
ruff check .
ruff format .

# Type checking (configured, not enforced yet)
mypy bayesian_changepoint_detection
```

## Migration from v0.4

The new PyTorch-based API maintains compatibility while offering performance improvements:

```python
# Old API (still works)
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
Q, P, Pcp = offcd.offline_changepoint_detection(data, prior_func, likelihood_func)

# New PyTorch API (recommended)
from bayesian_changepoint_detection import offline_changepoint_detection
Q, P, Pcp = offline_changepoint_detection(data, prior_func, likelihood)
```

## FAQ

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
   posterior hesitates between neighbouring points.

Offline, the equivalent of the hazard is the segment-length prior:
`const_prior(p=1/(T+1))` is the flat default; `geometric_prior(p=1/L)`
encodes an expected segment length `L`; `negative_binomial_prior` allows a
peaked length distribution. Leave `truncate` at its default: the sum is exact
and the legacy truncation can drop the dominant term.

### My data are not normally distributed. Can I still use this? (issue #36)

Every likelihood here assumes that **within a segment** the observations are
independent and Gaussian, and it detects changes in the mean and/or the
(co)variance of that Gaussian:

| likelihood | within-segment model |
|---|---|
| online `StudentT`, offline `StudentT` | i.i.d. Normal, unknown mean and variance (Normal-Gamma prior) |
| online `MultivariateT` | i.i.d. multivariate Normal, unknown mean and covariance (Normal-Wishart) |
| offline `IndependentFeaturesLikelihood` | one Normal-Gamma model per dimension, independent |
| offline `MultivariateT` | i.i.d. multivariate Normal, unknown mean and covariance (Normal-Wishart) |
| offline `FullCovarianceLikelihood` | multivariate Normal with unknown covariance and **no mean parameter** (mean zero, Xuan & Murphy 2007): it detects covariance changes; segments that differ in mean are misread as scale changes, so use `MultivariateT` when means move |

When the data are not Gaussian the detector still runs, and the question is
what the misspecification does to it:

- **Heavy tails or outliers**: single extreme points look like the start of
  a new segment. The Student-t predictive already tolerates some of this;
  a larger `lam` or `kappa` helps, and so does a transform (log for positive,
  right-skewed quantities such as latencies or prices).
- **Counts or bounded data**: a variance-stabilizing transform (square root
  or Anscombe for counts, logit for proportions) usually gets you close
  enough. A Poisson likelihood is on the roadmap (issue #23).
- **Autocorrelation or slow drift**: the model has no notion of dynamics
  within a segment, so a drift is reported as a sequence of small changes.
  Differencing, or modelling residuals from a trend, is the usual fix.
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

## Contributing

Contributions are welcome! Please see the [Contributing Guide](CONTRIBUTING.md)
for setup, conventions and the review process. The project follows the
[Contributor Covenant](CODE_OF_CONDUCT.md); security problems go through
[SECURITY.md](SECURITY.md), not public issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{bayesian_changepoint_detection,
  title={Bayesian Changepoint Detection: A PyTorch Implementation},
  author={Kulick, Johannes and Carisimo, Esteban},
  url={https://github.com/hildensia/bayesian_changepoint_detection},
  year={2026},
  version={1.1.0}
}
```

## Acknowledgments

- Original implementation by Johannes Kulick
- PyTorch migration and modernization by Esteban Carisimo
- Inspired by the work of Fearnhead, Adams, MacKay, Xiang, and Murphy
