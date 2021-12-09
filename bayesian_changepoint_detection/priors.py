import numpy as np
import scipy.stats as ss


def const_prior(t, p: float = 0.25):
    """
    Constant prior for every datapoint
    Arguments:
        p - probability of event
    """
    return np.log(p)


def geom_prior(t, p: float = 0.25):
    """
    geometric prior for every datapoint
    Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html for more information on the geometric prior
    Everything reported is in log form.
    Arguments:
        t - number of trials
        p - probability of success
    """
    return np.log(ss.geom.pmf(t, p=p))


def negative_binomial_prior(t, k: int = 1, p: float = 0.25):
    """
    negative binomial prior for the datapoints
     Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html for more information on the geometric prior
    Everything reported is in log form.

    Parameters:
        k - the number of trails until success
        p - the prob of success
    """

    return ss.nbinom.pmf(self.k, t, self.p)
