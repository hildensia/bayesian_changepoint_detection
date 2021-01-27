from __future__ import division

import numpy as np
from scipy import stats
from numpy.linalg import inv


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
    def __init__(self, dims, dof=None, kappa=1, mu=None, scale=None, chunksize=1):
        """
        Create a new predictor using the multivariate student T distribution as the posterior.
            This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
             and a Gaussian prior on the mean.
        :param dof: The degrees of freedom on the prior distribution of the precision (inverse covariance)
        :param kappa: TODO
        :param mu: The mean of the prior distribution on the mean
        :param scale: The mean of the prior distribution on the precision
        :param dims: The number of variables
        :param chunksize: The length of array to pre-allocate.
            This should be a best guess as to how long your data will be.
            If you aren't sure or want to save RAM at the expense of speed, leave this as 1
        """
        self.chunksize = chunksize

        if dof is None:
            dof = dims + 1
        if mu is None:
            mu = [0]*dims
        if scale is None:
            scale = np.identity(dims)

        # The number of data points we have seen
        self.n = 0
        self.dims = dims

        # Minor optimisation - rather than re-calculate this each update, we cache it
        self.inv_scale0 = inv(scale)

        # A hack to neaten the code: we store the original parameters here in a list, then immediately expand the array,
        # extended it to a length of chunksize
        self.dof = [dof]
        self.kappa = [kappa]
        self.mu = [mu]
        self.scale = [scale]

        self.expand()

    def expand(self):
        """
        Increases the length of each array by the chunksize
        """
        self.dof = np.hstack((self.dof, np.empty(self.chunksize)))
        self.kappa = np.vstack((self.kappa, np.empty(self.chunksize)))
        self.mu = np.vstack((self.mu, np.empty((self.chunksize, self.dims))))
        self.scale = np.vstack((self.scale, np.empty((self.chunksize, self.dims, self.dims))))

    def pdf(self, data):
        """
        Returns the probability of the observed data under the current parameters
        """
        t_dof = self.dof[self.n] - self.dims + 1
        try:
            return stats.multivariate_t.pdf(
                x=data,
                df=t_dof,
                loc=self.mu[self.n],
                shape=inv((self.kappa[self.n] * t_dof) / (self.kappa[self.n] + 1) * self.scale[self.n])
            )
        except AttributeError:
            raise Exception('You need scipy 1.6.0 or greater to use the multivariate t distribution')

    def update_theta(self, data):
        self.n += 1

        if self.n >= self.kappa.shape[0]:
            self.expand()
        centered = (data - self.mu[0])

        self.dof[self.n] = self.dof[self.n-1] + 1
        self.kappa[self.n] = self.kappa[self.n-1] + 1
        self.mu[self.n] = (self.kappa[0] * self.mu[0] + data) / (self.kappa[0] + 1)
        self.scale[self.n] = inv(self.inv_scale0 + (self.kappa[0] / (self.kappa[0] + 1)) * centered @ centered)
