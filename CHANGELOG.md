# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (breaking)

- `online_changepoint_detection` now returns `(R, map_run_lengths)`. The second
  value is the most likely run length after each observation (`argmax` of each
  column of `R`, dtype `long`), which is what versions before 1.0 returned as
  `maxes`. Versions 1.0.x returned the un-normalized `R[0, t]` under the name
  `changepoint_probs`; that quantity scales with the data evidence and, once
  normalized, equals the hazard rate at every step under a constant hazard, so
  it could not detect changepoints (#41, #18). Code that thresholded it must
  switch to `get_map_changepoints` or `changepoint_probabilities`.
- `get_map_changepoints(R)` now reports the data indices at which the MAP
  run-length path implies a new segment started. Its `threshold` argument is
  deprecated and ignored (it thresholded `R[0, :]`); a new `min_separation`
  argument merges nearby starts when the posterior flips between them.
- `online_likelihoods.MultivariateT` default prior: the Wishart scale is now
  `I / dof`, giving unit prior covariance as the documentation always claimed.
  The previous `I` encoded a prior covariance of `I / dof`. Explicit `scale`
  arguments are unaffected.
- `offline_likelihoods.StudentT` gains keyword-only prior hyperparameters
  `alpha0`, `beta0`, `kappa0`, `mu0` (defaults reproduce the previous prior).
- Distribution name on PyPI is `bayescd` (unchanged since 0.4); the
  `bayesian-changepoint-detection` name belongs to an unrelated project.
  `setup.py`, `setup.cfg`, `requirements.txt` and `uv.lock` were removed;
  `pyproject.toml` is the single source of metadata (license as an SPDX
  expression with `license-files`, per PEP 639; setuptools >= 77). `numpy` and `scipy`
  moved to the `dev` extra, `matplotlib`/`seaborn`/`numpy` to a new `plot`
  extra; the library itself depends only on `torch`. Python 3.8 dropped.

### Added

- `viterbi_changepoints` and `compute_run_length_posterior` are exported from
  the package. `viterbi_changepoints` is now a vectorized max-product pass
  that returns the most probable run-length path (the MAP segmentation under
  the BOCPD model), verified against an exhaustive search over segmentations;
  it runs in the same time as the forward pass (about 40x faster than
  before on 160 points). The previous version summed over predecessors in the
  changepoint transition, so its path scores were neither the forward pass
  nor Viterbi, and it had no tests. Input validation as for the other
  detectors.
- `changepoint_probabilities(R, lag)`: `P(a new segment started at t)` judged
  `lag` observations later, i.e. `R[lag, t + lag]`, the quantity the original
  notebook plotted as `R[Nw, Nw:]`.
- Offline likelihoods expose `pdf_rows(data, t)`, returning the log marginal
  likelihood of every segment starting at `t` in one vectorized call, and
  `setup(data)` to precompute sufficient statistics.
- Likelihood models can be moved between devices with `.to(device)`.
- CI workflow (GitHub Actions) running the test suite on Python 3.9–3.13,
  plus a `build` job: sdist and wheel, `twine check --strict`, and a smoke
  test that installs the wheel into a clean environment and runs both
  detectors.
- `AGENTS.md` with repository conventions.
- `CHANGELOG.md` (this file) and `.flake8`.
- `CONTRIBUTING.md` (setup, conventions, the `master` rule, releasing) and an
  explicit `.github/dependabot.yml` (GitHub Actions and pip, weekly, grouped;
  pip uses `increase-if-necessary` so lower bounds are not bumped needlessly).
- Community health files: `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1),
  `SECURITY.md`, `CITATION.cff`, `.github/CODEOWNERS`, issue templates and a
  pull request template.

### Fixed

- Offline recursion (`offline_changepoint_detection`) now indexes the
  segment-length prior correctly (Fearnhead 2006, eq. 2): the first
  changepoint row evaluated `g` at length minus one, the later rows paired
  `g` with the wrong segment length, and the tail term `1 - G` included a
  "length 0" term. None of this was visible with `const_prior`; with the
  geometric and negative binomial priors the changepoint posterior was wrong.
  Verified against an exhaustive enumeration of all segmentations
  (`tests/test_offline_prior_recursion.py`). With `const_prior` the log
  evidence moves by about 0.01 and changepoint locations do not move.
- `geometric_prior` was off by one (`(1-p)^t p` instead of the documented
  `(1-p)^(t-1) p`) and raised at length 0, which made it unusable with the
  offline detector; `negative_binomial_prior` had `p` and `1 - p` swapped, so
  `k = 1` did not reduce to the geometric prior. Both now match
  `scipy.stats.geom` / `scipy.stats.nbinom`; impossible lengths return `-inf`.
- `offline_changepoint_detection` raises a `ValueError` for empty input, for
  `NaN`/`Inf` values, and for a length prior whose mass on lengths `1..T-1`
  exceeds 1 (e.g. `const_prior(p=0.25)` on more than five points), instead
  of returning `nan` evidence and all-zero changepoint probabilities.
- Offline `StudentT` now evaluates the exact Normal-Gamma marginal likelihood
  of a segment (Murphy 2007, eqs. 95–97) instead of scoring each point under
  the posterior of the whole segment. Offline detection is 70–150x faster
  (1000 points: about 3 minutes to under 2 seconds on CPU) and recovers
  changepoints the previous code missed (#47).
- The offline recursion runs in float64 (float32 on MPS, with a CPU fallback
  for the recursion itself).
- Online `MultivariateT` now keeps the inverse Wishart scale `T = W^{-1}`
  as its state (`scale_inv`, Murphy 2007 eq. 255) and evaluates the
  predictive through a Cholesky factor of `T`; no matrix is inverted per
  step. The previous update inverted `W + 1e-6 I` and the result back every
  step, and on a shrinking `W` that regularizer compounds: on 2-D standard
  normal data (`numpy.random.default_rng(0)`), after 500 stationary points
  the posterior scale was 7.6% off and the log predictive 0.06 nats off,
  after 3000 points 67% and 1.0 nats (#59). `scale` is still available as a
  property (`inv(scale_inv)`). About 1.5-2x faster in 10-D.
- Online `MultivariateT` predictive used the Wishart scale `W` where `W^{-1}`
  belongs (Murphy 2007, eq. 258), so the predictive covariance shrank with
  every observation and the run-length posterior collapsed to run lengths
  1–3 even on stationary data. Verified against an independent NumPy/scipy
  implementation of the recursion.
- `constant_hazard` returned a tensor on the auto-selected device instead of
  the caller's, and `online_changepoint_detection` did not move the model
  and data to one device; both crashed on machines with MPS or CUDA even with
  `device="cpu"`.
- `IndependentFeaturesLikelihood` and `FullCovarianceLikelihood` produced
  `nan` for univariate input (zero-variance length-one segments).
- `.idea/` (IDE settings) and two example figures were tracked at the
  repository root; removed, and root-level `*.png` is now ignored.
- Root-level `test.py` was never collected by pytest; its tests now live in
  `tests/test_online_detection.py` with the pre-1.0 assertions restored.

## [1.0.0] — 2025-11-06

- PyTorch rewrite of the library (#46). Not published to PyPI.

## [0.4] — 2022-04-05

- Last NumPy release, published to PyPI as `bayescd`.
