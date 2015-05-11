from __future__ import division
import numpy as np
cimport numpy as np
from scipy.special import gammaln
from scipy.misc import comb
from decorator import decorator

try:
    from sselogsumexp import logsumexp
except ImportError:
    from scipy.misc import logsumexp
    print("Use scipy logsumexp().")
else:
    print("Use SSE accelerated logsumexp().")


def offline_changepoint_detection(np.ndarray[double, ndim=1] data,
                                  double truncate=-np.inf):
    """Compute the likelihood of changepoints on data.

    Keyword arguments:
    data -- the time series data
    prior_func -- a function given the likelihood of a changepoint given the
                  distance to the last one
    observation_log_likelihood_function -- a function giving the log likelihood
                                           of a data part
    P -- the likelihoods if pre-computed
    """

    cdef int t, s
    cdef int n = len(data)
    cdef np.ndarray[double, ndim=1] Q = np.zeros((n,))
    cdef np.ndarray[double, ndim=1] g = np.zeros((n,))
    cdef np.ndarray[double, ndim=1] G = np.zeros((n,))
    cdef np.ndarray[double, ndim=2] P = np.ones((n, n)) * -np.inf
    cdef np.ndarray[double, ndim=2] table = np.ones((n+1, n+1)) * np.nan
    cdef double summand, P_next_cp, antiG

    # save everything in log representation
    for t in range(n):
        g[t] = np.log(const_prior(t, len(data)+1))
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t-1], g[t])

    P[n-1, n-1], table = cy_gaussian_obs_log_likelihood(data, n-1, n, table)
    Q[n-1] = P[n-1, n-1]

    for t in reversed(range(n-1)):
        P_next_cp = -np.inf  # == -log(0)
        for s in range(t, n-1):
            P[t, s], table = cy_gaussian_obs_log_likelihood(data, t, s + 1, table)

            # compute recursion
            summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
            P_next_cp = np.logaddexp(P_next_cp, summand)

            # truncate sum to become approx. linear in time (see
            # Fearnhead, 2006, eq. (3))
            if summand - P_next_cp < truncate:
                break

        P[t, n-1], table = cy_gaussian_obs_log_likelihood(data, t, n, table)

        # (1 - G) is numerical stable until G becomes numerically 1
        if G[n-1-t] < -1e-15:  # exp(-1e-15) = .99999...
            antiG = np.log(1 - np.exp(G[n-1-t]))
        else:
            # (1 - G) is approx. -log(G) for G close to 1
            antiG = np.log(-G[n-1-t])

        Q[t] = np.logaddexp(P_next_cp, P[t, n-1] + antiG)

    cdef np.ndarray[double, ndim=2] Pcp = np.ones((n-1, n-1)) * -np.inf
    for t in range(n-1):
        Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
        if np.isnan(Pcp[0, t]):
            Pcp[0, t] = -np.inf
    cdef int j
    for j in range(1, n-1):
        for t in range(j, n-1):
            tmp_cond = Pcp[j-1, j-1:t] + P[j:t+1, t] + Q[t + 1] + g[0:t-j+1] - Q[j:t+1]
            Pcp[j, t] = logsumexp(tmp_cond.astype(np.float32))
            if np.isnan(Pcp[j, t]):
                Pcp[j, t] = -np.inf

    return Q, P, Pcp

#@dynamic_programming
cdef cy_gaussian_obs_log_likelihood(np.ndarray[double, ndim=1] data, int t, int s,
                                np.ndarray[double, ndim=2] table):

    if not np.isnan(table[t, s]):
        return table[t, s], table
    s += 1
    cdef int n = s - t
    cdef double mean = data[t:s].sum(0) / n

    cdef double muT = (n * mean) / (1 + n)
    cdef int nuT = 1 + n
    cdef double alphaT = 1 + n / 2
    cdef double betaT = 1 + 0.5 * ((data[t:s] - mean) ** 2).sum(0) + ((n)/(1 + n)) * (mean**2 / 2)
    scale = (betaT*(nuT + 1))/(alphaT * nuT)

    # splitting the PDF of the student distribution up is /much/ faster.
    # (~ factor 20) using sum over for loop is even more worthwhile
    cdef double prob = np.sum(np.log(1 + (data[t:s] - muT)**2/(nuT * scale)))
    cdef double lgA = gammaln((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) - gammaln(nuT/2)

    return np.sum(n * lgA - (nuT + 1)/2 * prob), table


cdef const_prior(r, l):
    return 1/(l)


def geometric_prior(t, p):
    return p * ((1 - p) ** (t - 1))


def neg_binominal_prior(t, k, p):
    return comb(t - k, k - 1) * p ** k * (1 - p) ** (t - k)
