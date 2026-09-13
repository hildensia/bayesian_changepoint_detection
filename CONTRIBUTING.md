# Contributing

Thanks for your interest in this library. This guide covers the development
setup, the checks every change must pass, and how a change gets from your
machine into `master`. Repository conventions that are easy to get wrong
(two `StudentT` classes, probability vs. log space, single-use online
likelihood objects, device handling) are in [`AGENTS.md`](AGENTS.md); read
it before touching the algorithms.

## Development setup

The library depends only on `torch`. The test suite also needs `numpy` and
`scipy`, which the `dev` extra provides.

```bash
git clone https://github.com/hildensia/bayesian_changepoint_detection.git
cd bayesian_changepoint_detection
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"                             # or: uv pip install -e ".[dev]"
pytest
```

The full suite takes well under a minute on CPU. On a machine with CUDA or
Apple's MPS it can take minutes, because device selection is automatic and
small tensors are slower on an accelerator; tests therefore pass
`device="cpu"` explicitly unless they are about device handling, and new
tests should do the same. New tests that need a GPU must carry
`@pytest.mark.gpu` and skip themselves when none is present; one existing
test, `test_device_consistency` in `tests/test_integration.py`, predates
the marker and only skips at runtime.

CI runs `pytest tests/` on Python 3.9 to 3.12 with CPU-only PyTorch. No
linter or type checker runs in CI yet; `pyproject.toml` carries black, isort
and mypy settings and `.flake8` the flake8 settings, so running them locally
is welcome but not enforced.

## Conventions

- NumPy-style docstrings on public functions and classes. When a docstring
  states a formula, cite the paper and equation it comes from.
- Every behaviour change gets a test under `tests/` and a line in
  `CHANGELOG.md` under `[Unreleased]`. Tests that pin a numerical result
  say where the expected value comes from (an independent reference
  implementation, a `scipy` evaluation, a paper); a test that compares the
  code against itself proves nothing.
- Changes to the mathematics follow the rules in the "Changing the math"
  section of `AGENTS.md`: name the source, prove the new form against an
  independent path, and say in the PR whether outputs change numerically.
- Keep PRs small and single-purpose. Do not mix refactors with statistical
  fixes.

## Making a change

`master` is not to be pushed to directly. GitHub does not currently enforce
this on the upstream repository, so it is policy, and it applies to
maintainers and to any automated agent acting for them:

- Every change, however small, lands through a pull request opened from a
  fork or a topic branch.
- Never force-push to `master`. If something has to be undone, revert it
  through a pull request.
- A PR is merged only when CI is green on all four `test (3.x)` legs and
  the review is done: request whatever automated code review the
  repository has enabled, or ask a person where none is available, and
  address every comment, either with a fix or with a reply in the thread
  saying why it does not apply.

Step by step:

1. Branch from the current `master`: `git switch -c <type>/<short-name>`
   (`fix/`, `feat/`, `docs/`, `chore/`, `perf/`).
2. Commit in small, coherent steps.
3. Push the branch to your fork and open the PR against
   `hildensia/bayesian_changepoint_detection:master`. Explain what changes,
   why, and how it was verified; for numerical changes include the numbers.
4. Iterate until CI is green and the review is addressed, then merge
   (squash for a single logical change, merge commit when the individual
   commits matter). Delete the branch afterwards.

Merging needs write access to the upstream repository. Branch protection,
repository secrets, PyPI credentials and review-tool settings need the
repository owner.

## Releasing

1. `version` in `pyproject.toml` is the only place the version lives;
   `bayesian_changepoint_detection.__version__` reads it from the installed
   package metadata. Bump it there.
2. Move the `[Unreleased]` section of `CHANGELOG.md` under a new
   `[X.Y.Z] — YYYY-MM-DD` heading.
3. Open a PR with those two changes and merge it.
4. Tag and create the GitHub release **on the upstream repository**, not on
   a fork (the `CD` workflow and its secrets only exist upstream):
   `git tag vX.Y.Z && git push <upstream-remote> vX.Y.Z`, then
   `gh release create vX.Y.Z --repo hildensia/bayesian_changepoint_detection --generate-notes`.
5. The `CD` workflow (`.github/workflows/cd.yml`) builds the distribution,
   runs `twine check`, and uploads to PyPI on the release event. Its upload
   step still uses the username/password secrets from 2022, which PyPI no
   longer accepts; publishing the `bayescd` distribution needs a PyPI API
   token or trusted publishing configured by the project owner. Until that
   is done, expect the upload step to fail; users install from a clone of
   the repository instead (README, "Development installation").

## Reporting issues

Open a GitHub issue with a minimal reproducible example: the data or a
generator with its seed, the likelihood and hazard/prior used, the device,
and the versions of Python and PyTorch. For something security-sensitive,
contact a maintainer privately through their GitHub profile rather than
posting the details in a public issue.
