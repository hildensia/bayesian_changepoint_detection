"""
Prove the closed-form StudentT marginal likelihood is exact.

The log marginal likelihood of a segment factorizes by the chain rule into a
product of one-step posterior predictive densities:

    p(x_1..x_n) = prod_k p(x_k | x_1..x_{k-1})

where each predictive is a Student's t with sequentially updated Normal-Gamma
parameters. That sequential product is computed here with an independent code
path (scipy.stats.t plus the textbook parameter recursion) and compared with
the closed-form expression used by
``offline_likelihoods.StudentT`` (Murphy 2007, "Conjugate Bayesian analysis
of the Gaussian distribution", eq. 95-97). Agreement at ~1e-10 over random
segments and hyperparameters establishes the formula's correctness.
"""

import math

import pytest
import torch

scipy_stats = pytest.importorskip("scipy.stats")

from bayesian_changepoint_detection.offline_likelihoods import StudentT


def sequential_log_marginal(x, alpha0, beta0, kappa0, mu0):
    """Chain-rule reference: sum of one-step predictive log densities."""
    alpha, beta, kappa, mu = alpha0, beta0, kappa0, mu0
    total = 0.0
    for xi in x:
        df = 2.0 * alpha
        scale = math.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        total += float(scipy_stats.t.logpdf(xi, df, loc=mu, scale=scale))
        # Normal-Gamma posterior update with one observation
        beta = beta + kappa * (xi - mu) ** 2 / (2.0 * (kappa + 1.0))
        mu = (kappa * mu + xi) / (kappa + 1.0)
        kappa += 1.0
        alpha += 0.5
    return total


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "alpha0,beta0,kappa0,mu0",
    [
        (1.0, 1.0, 1.0, 0.0),  # the class defaults
        (0.5, 2.0, 3.0, -1.5),
        (2.5, 0.1, 0.2, 4.0),
    ],
)
def test_closed_form_equals_sequential_product(seed, alpha0, beta0, kappa0, mu0):
    generator = torch.Generator().manual_seed(seed)
    data = torch.randn(80, generator=generator, dtype=torch.float64) * 2.0 + 1.0

    likelihood = StudentT(
        alpha0=alpha0, beta0=beta0, kappa0=kappa0, mu0=mu0, device="cpu"
    )

    for t, s in [(0, 1), (0, 5), (10, 40), (0, 80), (63, 80), (25, 26)]:
        closed_form = likelihood.pdf(data, t, s)
        reference = sequential_log_marginal(
            data[t:s].tolist(), alpha0, beta0, kappa0, mu0
        )
        assert closed_form == pytest.approx(reference, abs=1e-9), (
            f"segment [{t}, {s}): closed form {closed_form} != "
            f"sequential product {reference}"
        )


def test_multivariate_input_sums_independent_dimensions():
    generator = torch.Generator().manual_seed(3)
    data = torch.randn(50, 3, generator=generator, dtype=torch.float64)

    likelihood = StudentT(device="cpu")
    combined = likelihood.pdf(data, 5, 35)

    per_dim = sum(
        StudentT(device="cpu").pdf(data[:, d].contiguous(), 5, 35) for d in range(3)
    )
    assert combined == pytest.approx(per_dim, abs=1e-9)
