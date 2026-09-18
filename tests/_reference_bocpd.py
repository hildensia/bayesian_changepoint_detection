"""Independent NumPy/scipy reference implementations used by the tests."""
import itertools
import math

import numpy as np
from scipy.stats import multivariate_t

def reference_mv_bocpd(X, lam, dof0, kappa0, mu0, W0):
    """Adams & MacKay with Normal-Wishart predictive, Murphy (2007) eqs 255-258.
    Precision Lambda ~ Wishart(W, nu); posterior W_n^{-1} = W^{-1} + kappa/(kappa+1) (x-mu)(x-mu)^T."""
    T, D = X.shape
    R = np.zeros((T + 1, T + 1)); R[0, 0] = 1.0
    mu = [mu0.copy()]; kappa = [kappa0]; nu = [dof0]; Winv = [np.linalg.inv(W0)]
    H = 1.0 / lam
    for t in range(T):
        x = X[t]
        pred = np.empty(t + 1)
        for r in range(t + 1):
            tdof = nu[r] - D + 1
            shape = Winv[r] * (kappa[r] + 1) / (kappa[r] * tdof)
            pred[r] = multivariate_t.pdf(x, loc=mu[r], shape=shape, df=tdof)
        R[1:t + 2, t + 1] = R[:t + 1, t] * pred * (1 - H)
        R[0, t + 1] = np.sum(R[:t + 1, t] * pred * H)
        R[:, t + 1] /= R[:, t + 1].sum()
        new_mu, new_kappa, new_nu, new_Winv = [mu0.copy()], [kappa0], [dof0], [np.linalg.inv(W0)]
        for r in range(t + 1):
            d = x - mu[r]
            new_mu.append((kappa[r] * mu[r] + x) / (kappa[r] + 1))
            new_kappa.append(kappa[r] + 1); new_nu.append(nu[r] + 1)
            new_Winv.append(Winv[r] + kappa[r] / (kappa[r] + 1) * np.outer(d, d))
        mu, kappa, nu, Winv = new_mu, new_kappa, new_nu, new_Winv
    return R


def reference_offline_posterior(P, log_g):
    """Fearnhead (2006) posterior by enumerating every segmentation.

    P[t, s] is the log marginal likelihood of data[t:s+1] (taken from the
    library so that only the prior handling and the recursion are under
    test); log_g(l) is log P(segment length = l) for l >= 1. Segment lengths
    are i.i.d. from g except the last one, which only has to be at least its
    observed length: P(length >= l) = 1 - G(l - 1), G(l) = sum_{i<=l} g(i).

    Returns (log Q[0], P(changepoint at t) for t = 0..n-2, log Pcp[j, t]).
    Exponential in n; use n <= 10.
    """
    n = P.shape[0]
    g = np.array([-np.inf] + [log_g(l) for l in range(1, n + 1)])
    G = np.logaddexp.accumulate(g)

    def log1mexp(a):  # log(1 - exp(a)), a <= 0
        return math.log(-math.expm1(a)) if a < 0 else -np.inf

    total = -np.inf
    cp = np.full(n - 1, -np.inf)
    Pcp = np.full((n - 1, n - 1), -np.inf)
    for bits in itertools.product([0, 1], repeat=n - 1):  # bits[t]: boundary after t
        ends = [t for t in range(n - 1) if bits[t]] + [n - 1]
        start, lp = 0, 0.0
        for k, e in enumerate(ends):
            length = e - start + 1
            lp += P[start, e]
            lp += g[length] if k < len(ends) - 1 else log1mexp(G[length - 1])
            start = e + 1
        total = np.logaddexp(total, lp)
        for j, e in enumerate(ends[:-1]):
            cp[e] = np.logaddexp(cp[e], lp)
            Pcp[j, e] = np.logaddexp(Pcp[j, e], lp)
    return total, np.exp(cp - total), Pcp - total

def reference_map_segmentation(P, T, hazard):
    """MAP segmentation under a constant hazard by enumerating all segmentations.

    P[t, s] is the log marginal likelihood of data[t:s+1] (from the offline
    closed form with the same conjugate prior as the online model). Each of
    the T transitions after an observation is a changepoint (prob ``hazard``)
    or growth (``1 - hazard``); the transition after the last observation is
    growth, which is what a Viterbi pass picks when ``hazard < 0.5``.

    Returns (best log joint probability, sorted list of segment starts > 0).
    Exponential in T; use T <= 10.
    """
    best, best_starts = -np.inf, None
    for bits in itertools.product([0, 1], repeat=T - 1):
        starts = [0] + [t + 1 for t in range(T - 1) if bits[t]]
        ends = starts[1:] + [T]
        lp = sum(P[s, e - 1] for s, e in zip(starts, ends))
        k = len(starts) - 1
        lp += k * math.log(hazard) + (T - k) * math.log1p(-hazard)
        if lp > best:
            best, best_starts = lp, starts[1:]
    return best, best_starts
