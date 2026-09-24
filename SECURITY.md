# Security Policy

## Supported versions

Security fixes go into the current development line on `master` and into the
next release. The distribution is `bayesian-changepoint` from 1.1.0 on. Older
tags and the pre-PyTorch releases (0.x, published to PyPI as `bayescd` up to
0.4 and as `bayesian-changepoint-detection` before that) are not maintained.

## Reporting a vulnerability

Please do **not** open a public issue, pull request or discussion for security
problems.

GitHub's private vulnerability reporting is **not enabled** on this repository
(checked 2026-09-18; enabling it needs the repository owner). Until it is,
open an issue titled "Security contact request" that says nothing about the
problem itself, and the maintainer (`@estcarisimo`, the code owner in
[`.github/CODEOWNERS`](.github/CODEOWNERS)) will arrange a private channel.

Please include:

- The library version (`python -c "import bayesian_changepoint_detection as b; print(b.__version__)"`
  or the commit hash) and the Python and PyTorch versions.
- What the issue is and what an attacker can do with it.
- A minimal reproduction: a short script, and a small input file if one is needed.
- Whether you have already disclosed it anywhere else.

## What to expect

The project is maintained by researchers on a volunteer basis, so response
times are best-effort:

| Step | Target |
| ---- | ------ |
| Acknowledgement of your report | within 7 days |
| Initial assessment (confirmed / not a vulnerability / needs more information) | within 14 days |
| Fix merged for a confirmed issue | within 90 days, sooner for anything severe |

We follow coordinated disclosure: the report stays private while a fix is
prepared, you are credited in `CHANGELOG.md` unless you prefer otherwise, and
the issue is made public once the fix is on `master`. If no fix has been merged
90 days after a confirmed report was acknowledged, you are free to disclose it.

## Scope

This is a numerical library: it takes tensors or arrays in, runs Bayesian
changepoint detection with PyTorch, and returns tensors. It has no network
code, does not read files on its own, and does not execute code from its
inputs.

In scope:

- Anything that lets crafted input do more than produce a wrong answer or a
  normal exception (for example code execution or file access through a
  dependency call).
- Unsafe defaults in the public API.
- Vulnerabilities in the GitHub Actions workflows or the release process.

Out of scope:

- Memory or time exhaustion from very long series. Both algorithms allocate
  a matrix of size about T × T for a series of length T (the run-length
  posterior `R` online, `P` offline) and the online docstring states
  the O(T²) time cost, so quadratic growth is by design.
- Vulnerabilities in PyTorch, NumPy or SciPy that do not depend on how this
  library uses them; please report those upstream.

## Dependencies and automation

- Dependabot version updates are configured in `.github/dependabot.yml`
  (GitHub Actions and pip, weekly). Dependabot alerts and security updates
  are repository settings controlled by the owner.
- CI runs the test suite on every push and pull request (`.github/workflows/ci.yml`).
