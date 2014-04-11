from __future__ import division
import numpy as np
from scipy.special import gammaln
from scipy.misc import comb

def offline_changepoint_detection(data, prior_func, observation_log_likelihood_function, P=None):
    """Compute the likelihood of changepoints on data.

    Keyword arguments:
    data -- the time series data
    prior_func -- a function given the likelihood of a changepoint given the
                  distance to the last one
    observation_log_likelihood_function -- a function giving the log likelihood
                                           of a data part
    P -- the likelihoods if pre-computed
    """

    n = len(data)
    Q = np.zeros((n,))
    g = np.zeros((n,))
    G = np.zeros((n,))
    
    if P is None:
        P = np.ones((n,n))  # a log_likelihood won't become one
 
    # save everything in log representation
    for t in range(n):
        g[t] = np.log(prior_func(t))
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t-1], g[t])

    if P[n-1, n-1] == 1:
        P[n-1, n-1] = observation_log_likelihood_function(data, n-1, n)
        
    Q[n-1] = P[n-1, n-1]
   
    for t in reversed(range(n-1)):
        P_next_cp = -np.inf  # == -log(0)
        for s in range(t, n-1):
            # don't recompute log likelihoods already saved
            if P[t, s] == 1:
                P[t, s] = observation_log_likelihood_function(data, t, s + 1)
            
            # compute recursion
            summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
            P_next_cp = np.logaddexp(P_next_cp, summand)
            
            # truncate sum to become approx. linear in time (see Fearnhead, 2006, eq. (3))
            #if summand - P_next_cp < -100:
                #break
                
        if P[t, n-1] == 1:
            P[t, n-1] = observation_log_likelihood_function(data, t, n)

        # (1 - G) is numerical stable until G becomes numerically 1
        if G[n-1-t] < -1e-15:  # exp(-1e-15) = .99999...
            antiG = np.log(1 - np.exp(G[n-1-t])) 
        else:
            # (1 - G) is approx. -log(G) for G close to 1
            antiG = np.log(-G[n-1-t])

        Q[t] = np.logaddexp(P_next_cp, P[t, n-1] + antiG) 
        
    Pcp = np.ones((n-1, n-1)) * -np.inf
    for t in range(n-1):
        if P[0, t] == 1:
            P[0, t] = observation_log_likelihood_function(data, 0, t+1)
        Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0] 
    for j in range(1, n-1):
        for t in range(j, n-1):
            for tau in range(j-1, t):
                if P[tau+1, t] == 1:
                    P[tau+1, t] = observation_log_likelihood_function(data, tau+1, t+1)
                tmp_cond = Pcp[j-1, tau] + P[tau+1, t] + Q[t + 1] + g[t - tau] - Q[tau + 1]
                Pcp[j, t] = np.logaddexp(Pcp[j, t], tmp_cond)

    return Q, P, Pcp

def gaussian_obs_log_likelihood(data, t, s):
    n = s - t
    mean = data[t:s].sum(0) / n
    
    muT = (n * mean) / (1 + n)
    nuT = 1 + n
    alphaT = 1 + n / 2
    betaT = 1 + 0.5 * ((data[t:s] - mean) ** 2).sum(0) + ((n)/(1 + n)) * (mean**2 / 2)
    scale = (betaT*(nuT + 1))/(alphaT * nuT)
    
    # splitting the PDF of the student distribution up is /much/ faster. (~ factor 20)
    # using sum over for loop is even more worthwhile
    prob = np.sum(np.log(1 + (data[t:s] - muT)**2/(nuT * scale)))
    lgA = gammaln((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) - gammaln(nuT/2)
    
    return np.sum(n * lgA - (nuT + 1)/2 * prob)


def const_prior(r, l):
    return 1/(l)

def geometric_prior(t, p):
    return p * ((1 - p) ** (t - 1))

def neg_binominal_prior(t, k, p):
    return comb(t - k, k - 1) * p ** k * (1 - p) ** (t - k)
