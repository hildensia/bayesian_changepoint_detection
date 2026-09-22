"""Offline segment marginals with the sufficient statistics in exact
rational arithmetic.

The float data are exact rationals, so sums, scatter matrices and
determinants are computed without rounding (``fractions.Fraction``); only
the final logs and gamma functions round. This isolates the *numerical*
accuracy of the library's prefix-sum pipeline, which the scipy-based tests
cannot probe at large offsets. The formulas are the published closed forms
(Murphy 2007; Xuan & Murphy 2007), written out here independently of the
library's vectorized code.
"""

import math
from fractions import Fraction

from scipy.special import multigammaln

LOG_PI = math.log(math.pi)
LOG_2PI = math.log(2 * math.pi)


def _log(x):
    return math.log(float(x)) if not isinstance(x, Fraction) else _log_fraction(x)


def _log_fraction(x):
    # log of a positive rational without overflowing float conversion
    return math.log(x.numerator) - math.log(x.denominator)


def _det(matrix):
    """Exact determinant of a small square matrix of Fractions."""
    a = [row[:] for row in matrix]
    size, det = len(a), Fraction(1)
    for col in range(size):
        pivot = next(r for r in range(col, size) if a[r][col] != 0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det
        det *= a[col][col]
        for r in range(col + 1, size):
            factor = a[r][col] / a[col][col]
            for c in range(col, size):
                a[r][c] -= factor * a[col][c]
    return det


def _rows(segment):
    """[[Fraction]] rows from a list of floats or a list of lists."""
    return [
        [Fraction(v) for v in (row if isinstance(row, (list, tuple)) else [row])]
        for row in segment
    ]


def student_t(segment, alpha0, beta0, kappa0, mu0):
    """Normal-Gamma marginal, summed over independent dimensions."""
    rows = _rows(segment)
    n, total = len(rows), 0.0
    for j in range(len(rows[0])):
        xs = [row[j] for row in rows]
        mean = sum(xs) / n
        ss = sum((x - mean) ** 2 for x in xs)
        kappa_n = Fraction(kappa0) + n
        beta_n = (
            Fraction(beta0)
            + ss / 2
            + Fraction(kappa0) * n * (mean - Fraction(mu0)) ** 2 / (2 * kappa_n)
        )
        alpha_n = alpha0 + n / 2
        total += (
            math.lgamma(alpha_n)
            - math.lgamma(alpha0)
            + alpha0 * math.log(beta0)
            - alpha_n * _log_fraction(beta_n)
            + 0.5 * (math.log(kappa0) - _log_fraction(kappa_n))
            - n / 2 * LOG_2PI
        )
    return total


def _flat_variance(rows):
    flat = [v for row in rows for v in row]
    mean = sum(flat) / len(flat)
    return sum((v - mean) ** 2 for v in flat) / len(flat)


def independent_features(segment):
    """Xuan & Murphy (2007) section 3.1 with N0 = d, V0 = flattened variance."""
    rows = _rows(segment)
    n, d = len(rows), len(rows[0])
    n0, v0 = d, _flat_variance(rows)
    total = d * (
        -n / 2 * LOG_PI
        + n0 / 2 * _log_fraction(v0)
        - math.lgamma(n0 / 2)
        + math.lgamma((n0 + n) / 2)
    )
    for j in range(d):
        vn = v0 + sum(row[j] ** 2 for row in rows)
        total -= (n0 + n) / 2 * _log_fraction(vn)
    return total


def full_covariance(segment):
    """Xuan & Murphy (2007): zero mean, Wishart with N0 = d, V0 = v0 I."""
    rows = _rows(segment)
    n, d = len(rows), len(rows[0])
    n0, v0 = d, _flat_variance(rows)
    vn = [
        [(v0 if i == j else 0) + sum(row[i] * row[j] for row in rows) for j in range(d)]
        for i in range(d)
    ]
    return (
        -(d * n / 2) * LOG_PI
        + n0 / 2 * d * _log_fraction(v0)
        - multigammaln(n0 / 2, d)
        + multigammaln((n0 + n) / 2, d)
        - (n0 + n) / 2 * _log_fraction(_det(vn))
    )


def multivariate_t(segment, kappa0, dof0, mu0, psi0):
    """Normal-inverse-Wishart marginal (Murphy 2007)."""
    rows = _rows(segment)
    n, d = len(rows), len(rows[0])
    mean = [sum(row[j] for row in rows) / n for j in range(d)]
    diff = [mean[j] - Fraction(mu0[j]) for j in range(d)]
    kappa_n = Fraction(kappa0) + n
    psi_n = [
        [
            Fraction(psi0[i][j])
            + sum((row[i] - mean[i]) * (row[j] - mean[j]) for row in rows)
            + Fraction(kappa0) * n / kappa_n * diff[i] * diff[j]
            for j in range(d)
        ]
        for i in range(d)
    ]
    psi0_exact = [[Fraction(v) for v in row] for row in psi0]
    dof_n = dof0 + n
    return (
        multigammaln(dof_n / 2, d)
        - multigammaln(dof0 / 2, d)
        + dof0 / 2 * _log_fraction(_det(psi0_exact))
        - dof_n / 2 * _log_fraction(_det(psi_n))
        + d / 2 * (math.log(kappa0) - _log_fraction(kappa_n))
        - n * d / 2 * LOG_PI
    )
