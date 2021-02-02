from __future__ import division

import numpy as np
from numpy.linalg import inv
from scipy import stats
from itertools import islice


def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])

        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)

        maxes[t] = R[:, t].argmax()
    return R, maxes


def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)


class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(x=data,
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

class MultivariateT:
    def __init__(self, dims, dof=None, kappa=1, mu=None, scale=None):
        """
        Create a new predictor using the multivariate student T distribution as the posterior predictive.
            This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
             and a Gaussian prior on the mean.
             Implementation based on Haines, T.S., Gaussian Conjugate Prior Cheat Sheet.
        :param dof: The degrees of freedom on the prior distribution of the precision (inverse covariance)
        :param kappa: The number of observations we've already seen
        :param mu: The mean of the prior distribution on the mean
        :param scale: The mean of the prior distribution on the precision
        :param dims: The number of variables
        """
        # We default to the minimum possible degrees of freedom, which is 1 greater than the dimensionality
        if dof is None:
            dof = dims + 1
        # The default mean is all 0s
        if mu is None:
            mu = [0]*dims
        # The default covariance is the identity matrix. The scale is the inverse of that, which is also the identity
        if scale is None:
            scale = np.identity(dims)

        # Track time
        self.t = 0

        # The dimensionality of the dataset (number of variables)
        self.dims = dims

        # Each parameter is a vector of size 1 x t, where t is time. Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def pdf(self, data):
        """
        Returns the probability of the observed data under the current and historical parameters
        :param data: A 1 x D vector of new data
        """
        self.t += 1
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims((self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        ret = np.empty(self.t)
        try:
            # This can't be vectorised due to https://github.com/scipy/scipy/issues/13450
            for i, (df, loc, shape) in islice(enumerate(zip(
                t_dof,
                self.mu,
                inv(expanded * self.scale)
            )), self.t):
                ret[i] = stats.multivariate_t.pdf(
                    x=data,
                    df=df,
                    loc=loc,
                    shape=shape
                )
        except AttributeError:
            raise Exception('You need scipy 1.6.0 or greater to use the multivariate t distribution')
        return ret

    def update_theta(self, data):
        """
        Performs a bayesian update on the prior parameters, given data
        :param data: A 1 x D vector of new data
        """
        centered = data - self.mu

        # We simultaneously update each parameter in the vector, because following figure 1c of the BOCD paper, each
        # parameter for a given t, r is derived from the same parameter for t-1, r-1
        # Then, we add the prior back in as the first element
        self.scale = np.concatenate([
            self.scale[:1],
            inv(
                inv(self.scale)
                + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2)) * (np.expand_dims(centered, 2) @ np.expand_dims(centered, 1))
            )
        ])
        self.mu = np.concatenate([self.mu[:1], (np.expand_dims(self.kappa, 1) * self.mu + data)/np.expand_dims(self.kappa + 1, 1)])
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])

# Plots the run length distributions along with a dataset
def plot(R, data):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(data)
    ax = fig.add_subplot(2, 1, 2, sharex=ax)
    sparsity = 1  # only plot every fifth data for faster display
    ax.pcolor(
        np.array(range(0, len(R[:, 0]), sparsity)),
        np.array(range(0, len(R[:, 0]), sparsity)),
        np.log(R),
        cmap=cm.Greys, vmin=-30, vmax=0
    )
    return fig

