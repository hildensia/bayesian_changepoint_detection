# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `online_likelihoods.Poisson` and `offline_likelihoods.Poisson` for count
  data (issue #23): Poisson counts with a conjugate Gamma prior on the rate,
  negative-binomial predictive online, closed-form segment marginal offline
  (Gelman et al., BDA3, section 2.6). Checked against `scipy.stats.nbinom`
  and against the chain-rule product of predictives. The online class
  supports `max_run_length` in `OnlineChangepointDetector`.
- `OnlineChangepointDetector`, the online detector one observation at a
  time for streams of unknown length (issue #13): `update(x)`,
  `run_length_posterior`, `map_run_length`, `changepoint_probability(lag)`.
  `max_run_length` bounds memory and time per observation by dropping run
  lengths above the bound and renormalizing. Checked against the NumPy
  reference with and without the bound, and against
  `online_changepoint_detection`. Online likelihoods gain `prune(n)`, driven
  by a `_run_length_state` class attribute.
- Documentation site built with MkDocs (Material theme, API reference
  generated from the docstrings by mkdocstrings), checked with
  `mkdocs build --strict` in CI. Its pages include sections of the README,
  so the two stay in sync. The `docs` extra now installs MkDocs instead of
  Sphinx, which had no configuration in the repository. Not published yet.

### Fixed

- The three detectors validate their input in one place, before any work:
  data must be `[T]` or `[T, D]`, non-empty, real, finite and, when the
  likelihood declares `dims`, have that many components per observation.
  Violations raise `ValueError`. Before, a 0-d tensor raised `IndexError`,
  complex data ran, and offline `MultivariateT(dims=3)` silently accepted
  univariate data or a transposed `[3, T]` tensor (read as three
  observations of dimension T). A `[D, T]` tensor now gets a "pass data.T"
  hint. `[T]` data with a `dims=1` multivariate likelihood is taken as
  `[T, 1]` (the online detector used to fail on it). Valid inputs give the
  same results as before.
- Offline `MultivariateT` no longer overwrites `dims=None` with the first
  series' dimension; an explicit `dims` that disagrees with the data raises
  `ValueError` from `pdf` and `pdf_rows` too.

- `constant_hazard(lam, r)` raises `ValueError` for `lam < 1` and for NaN.
  It used to accept any positive `lam`, so `0 < lam < 1` returned a hazard
  `1 / lam` above 1, which is not a probability and made the online
  detector's growth probabilities negative. `lam >= 1` behaves as before.

### Changed

- The test marker `behaviour` is now `behavior` (`pytest -m behavior`), under
  the new rule that the project's identifiers use American English. No
  library names changed; none used British spellings.
- The release workflow uploads from the `pypi` GitHub environment (`testpypi`
  for rehearsals), so each upload is listed under "Deployments" on the
  repository page. Nothing about the published package changes.
- CI measures test coverage on every Python version, fails below 93%, and
  shows the per-module table in the run summary. New tests for the synthetic
  data generators took `generate_data` from 42% to 97% coverage.

## [1.1.0] — 2026-09-22

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
- `offline_likelihoods.MultivariateT` default prior: `Psi0` (the
  covariance-side scale, `inv(W)`) is now `dof0 * I`, unit prior covariance,
  the same prior as the online class; it was `I`, `dof0` times tighter (#75).
  Changepoint locations on the multivariate test series are unchanged; a
  spurious 0.33 bump on the 2-D regression series drops to 0.07. Explicit
  `Psi0` arguments are unaffected.
- `online_likelihoods.MultivariateT` default prior: the Wishart scale is now
  `I / dof`, giving unit prior covariance as the documentation always claimed.
  The previous `I` encoded a prior covariance of `I / dof`. Explicit `scale`
  arguments are unaffected.
- `offline_likelihoods.StudentT` gains keyword-only prior hyperparameters
  `alpha0`, `beta0`, `kappa0`, `mu0` (defaults reproduce the previous prior).
- **Distribution renamed to `bayesian-changepoint`** (`pip install
  bayesian-changepoint`). The import name is unchanged:
  `import bayesian_changepoint_detection`. The same project was published as
  `bayescd` up to 0.4 (April 2022) and as `bayesian-changepoint-detection`
  before that (0.2.dev1); both stay frozen at those releases. Earlier drafts
  of this changelog and of the packaging metadata called
  `bayesian-changepoint-detection` an unrelated project; that was wrong, it is
  this project's own older distribution. `__version__` reads the new
  distribution's metadata and falls back to `bayescd` for an older install.
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
- ruff for linting and formatting (`[tool.ruff]` in `pyproject.toml`, a
  `lint` CI job, `.pre-commit-config.yaml`); black, isort and flake8 are no
  longer used and `.flake8` is gone. The whole tree was formatted once.
- CI job `examples`: runs every example script headless with the `plot`
  extra installed, so examples cannot drift from the API unnoticed.
- CI workflow (GitHub Actions) running the test suite on Python 3.9–3.13,
  plus a `build` job: sdist and wheel, `twine check --strict`, and a smoke
  test that installs the wheel into a clean environment and runs both
  detectors.
- `AGENTS.md` with repository conventions.
- `CHANGELOG.md` (this file).
- README FAQ: online vs offline, the one-index difference between their
  conventions, data scaling and prior units (#34), sensitivity (#31),
  non-Gaussian data (#36), CPU vs accelerator.
- Test kind markers `math` (checked against an independent reference) and
  `behaviour` (pins current behaviour); every test carries exactly one and
  `tests/conftest.py` fails collection otherwise. `pytest -m math` runs the
  87 tests that would fail if the mathematics were wrong.
- Releases publish to PyPI with Trusted Publishing (OpenID Connect): `cd.yml`
  builds the sdist and wheel, checks the version against the tag, uploads with
  `pypa/gh-action-pypi-publish` and attaches both files to the GitHub release.
  No PyPI token or password is stored in the repository; the 2022
  username/password secrets are no longer used. A manual run of the workflow
  can upload to TestPyPI instead.
- README rewritten: what the library computes, a working install,
  online/offline/multivariate
  examples whose printed outputs are the real ones and which run in the test
  suite, how to read `R`, an API table, the package layout, development
  commands, captured example output, measured performance only, the FAQ,
  and full references. Gone: three installation methods and a
  troubleshooting section, the "Migration from v0.4" section (the old module
  path it showed does not exist), and the remaining unmeasured GPU claims.
- `docs/devices.md`: how the `device` argument is resolved by each detector,
  the MPS-to-CPU fallback of the offline detector, what has been measured,
  how to time your own workload, and the memory footprint of the run-length
  and offline tables. It replaces `docs/gpu_offline_detection_guide.md` and
  `docs/gpu_online_detection_guide.md` (2 500 lines written for the PyTorch
  port and never executed: they unpacked the removed `changepoint_probs`
  output, thresholded it at 0.5, linked to a guide that did not exist, and
  recommended GPUs for series above 1 000 points on the strength of speedups
  nobody had measured; #56). A test executes every Python block under
  `docs/` so the prose cannot drift from the API again.
- `maintainers` in `pyproject.toml` lists both maintainers.
- `CONTRIBUTING.md` (setup, conventions, the `master` rule, releasing) and an
  explicit `.github/dependabot.yml` (GitHub Actions and pip, weekly, grouped;
  pip uses `increase-if-necessary` so lower bounds are not bumped needlessly).
- Community health files: `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1),
  `SECURITY.md`, `CITATION.cff`, `.github/CODEOWNERS`, issue templates and a
  pull request template.

### Fixed

- `offline_changepoint_detection` evaluates the sum over segment ends in
  full; `truncate` now defaults to `-inf` and is deprecated. The legacy
  rule (cut at the first term 40 nats below the running sum, inherited
  from the NumPy original) assumed the terms decay after a peak, but for a
  start inside a segment they dip and then rise to the true end; with
  multivariate likelihoods the cut discarded that dominant term and
  returned changepoint "probabilities" around 1e31 (10-D example with three
  changes: `MultivariateT` and `IndependentFeaturesLikelihood` both). Since
  the segment likelihoods are computed for every end in one vectorized call,
  truncation saved no work either. Univariate results are unchanged.
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
- `online_changepoint_detection` raises a `ValueError` for empty input and
  for `NaN`/`Inf` values, like the offline detector and `viterbi_changepoints`;
  it used to return a 1x1 posterior, or a non-finite `R` whose MAP run
  length is 0 at every step.
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
- `examples/example.py` removed: it still used the pre-1.0 NumPy module
  layout (`offline_changepoint_detection` module, `const_prior(l=...)`) and
  had not run since the PyTorch rewrite; `basic_usage.py` and
  `simple_example.py` cover the same ground.
- `.idea/` (IDE settings) and two example figures were tracked at the
  repository root; removed, and root-level `*.png` is now ignored.
- Root-level `test.py` was never collected by pytest; its tests now live in
  `tests/test_online_detection.py` with the pre-1.0 assertions restored.

## [1.0.0] — 2025-11-06

- PyTorch rewrite of the library (#46). Not published to PyPI.

## [0.4] — 2022-04-05

- Last NumPy release, published to PyPI as `bayescd`. (Earlier releases used
  the distribution name `bayesian-changepoint-detection`.)
