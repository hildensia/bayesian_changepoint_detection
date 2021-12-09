import numpy as np

try:
    from sselogsumexp import logsumexp
except ImportError:
    from scipy.special import logsumexp

    print("Use scipy logsumexp().")
else:
    print("Use SSE accelerated logsumexp().")


def offline_changepoint_detection(
    data, prior_function, log_likelihood_class, truncate: int = -40
):
    """
    Compute the likelihood of changepoints on data.

    Parameters:
    data    -- the time series data
    truncate  -- the cutoff probability 10^truncate to stop computation for that changepoint log likelihood

    Outputs:
        P  -- the log-likelihood of a datasequence [t, s], given there is no changepoint between t and s
        Q -- the log-likelihood of data
        Pcp --  the log-likelihood that the i-th changepoint is at time step t. To actually get the probility of a changepoint at time step t sum the probabilities.
    """

    # Set up the placeholders for each parameter
    n = len(data)
    Q = np.zeros((n,))
    g = np.zeros((n,))
    G = np.zeros((n,))
    P = np.ones((n, n)) * -np.inf

    # save everything in log representation
    for t in range(n):
        g[t] = prior_function(t)
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t - 1], g[t])

    P[n - 1, n - 1] = log_likelihood_class.pdf(data, t=n - 1, s=n)
    Q[n - 1] = P[n - 1, n - 1]

    for t in reversed(range(n - 1)):
        P_next_cp = -np.inf  # == log(0)
        for s in range(t, n - 1):
            P[t, s] = log_likelihood_class.pdf(data, t=t, s=s + 1)

            # compute recursion
            summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
            P_next_cp = np.logaddexp(P_next_cp, summand)

            # truncate sum to become approx. linear in time (see
            # Fearnhead, 2006, eq. (3))
            if summand - P_next_cp < truncate:
                break

        P[t, n - 1] = log_likelihood_class.pdf(data, t=t, s=n)

        # (1 - G) is numerical stable until G becomes numerically 1
        if G[n - 1 - t] < -1e-15:  # exp(-1e-15) = .99999...
            antiG = np.log(1 - np.exp(G[n - 1 - t]))
        else:
            # (1 - G) is approx. -log(G) for G close to 1
            antiG = np.log(-G[n - 1 - t])

        Q[t] = np.logaddexp(P_next_cp, P[t, n - 1] + antiG)

    Pcp = np.ones((n - 1, n - 1)) * -np.inf
    for t in range(n - 1):
        Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
        if np.isnan(Pcp[0, t]):
            Pcp[0, t] = -np.inf
    for j in range(1, n - 1):
        for t in range(j, n - 1):
            tmp_cond = (
                Pcp[j - 1, j - 1 : t]
                + P[j : t + 1, t]
                + Q[t + 1]
                + g[0 : t - j + 1]
                - Q[j : t + 1]
            )
            Pcp[j, t] = logsumexp(tmp_cond.astype(np.float32))
            if np.isnan(Pcp[j, t]):
                Pcp[j, t] = -np.inf

    return Q, P, Pcp


def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Use online bayesian changepoint detection
    https://scientya.com/bayesian-online-change-point-detection-an-intuitive-understanding-b2d2b9dc165b

    Parameters:
    data    -- the time series data

    Outputs:
        R  -- is the probability at time step t that the last sequence is already s time steps long
        maxes -- the argmax on column axis of matrix R (growth probability value) for each time step
    """
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = R[:, t].argmax()

    return R, maxes
