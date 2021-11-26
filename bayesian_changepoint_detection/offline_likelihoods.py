from abc import ABC, abstractmethod
from decorator import decorator

import numpy as np
import scipy.stats as ss
from scipy.special import gammaln, multigammaln, comb


def _dynamic_programming(f, *args, **kwargs):
    if f.data is None:
        f.data = args[1]

    if not np.array_equal(f.data, args[1]):
        f.cache = {}
        f.data = args[1]

    try:
        f.cache[args[2:4]]
    except KeyError:
        f.cache[args[2:4]] = f(*args, **kwargs)
    return f.cache[args[2:4]]


def dynamic_programming(f):
    f.cache = {}
    f.data = None
    return decorator(_dynamic_programming, f)


class BaseLikelihood(ABC):
    """
    This is an abstract class to serve as a template for future users to mimick
    if they want to add new models for offline bayesian changepoint detection.

    Make sure to override the abstract methods to do which is desired.
    Otherwise you will get an error.
    """

    @abstractmethod
    def pdf(self, data: np.array, t: int, s: int):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class and override this function."
        )


class IndepentFeaturesLikelihood:
    """
    Return the pdf for an independent features model discussed in xuan et al

    Parmeters:
        data - the datapoints to be evaluated (shape: 1 x D vector)
        t - start of data segment
        s - end of data segment
    """

    def pdf(self, data: np.array, t: int, s: int):
        s += 1
        n = s - t
        x = data[t:s]
        if len(x.shape) == 2:
            d = x.shape[1]
        else:
            d = 1
            x = np.atleast_2d(x).T

        N0 = d  # weakest prior we can use to retain proper prior
        V0 = np.var(x)
        Vn = V0 + (x ** 2).sum(0)

        # sum over dimension and return (section 3.1 from Xuan paper):
        return d * (
            -(n / 2) * np.log(np.pi)
            + (N0 / 2) * np.log(V0)
            - gammaln(N0 / 2)
            + gammaln((N0 + n) / 2)
        ) - (((N0 + n) / 2) * np.log(Vn)).sum(0)


class FullCovarianceLikelihood:
    def pdf(self, data: np.ndarray, t: int, s: int):
        """
        Return the pdf function for the covariance model discussed in xuan et al

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
            t - start of data segment
            s - end of data segment
        """
        s += 1
        n = s - t
        x = data[t:s]
        if len(x.shape) == 2:
            dim = x.shape[1]
        else:
            dim = 1
            x = np.atleast_2d(x).T

        N0 = dim  # weakest prior we can use to retain proper prior
        V0 = np.var(x) * np.eye(dim)

        # Improvement over np.outer
        # http://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
        # Vn = V0 + np.array([np.outer(x[i], x[i].T) for i in xrange(x.shape[0])]).sum(0)
        Vn = V0 + np.einsum("ij,ik->jk", x, x)

        # section 3.2 from Xuan paper:
        return (
            -(dim * n / 2) * np.log(np.pi)
            + (N0 / 2) * np.linalg.slogdet(V0)[1]
            - multigammaln(N0 / 2, dim)
            + multigammaln((N0 + n) / 2, dim)
            - ((N0 + n) / 2) * np.linalg.slogdet(Vn)[1]
        )


class StudentT(BaseLikelihood):
    @dynamic_programming
    def pdf(self, data: np.ndarray, t: int, s: int):
        """
        Return the pdf function of the t distribution
        Uses update approach in https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf (page 8, 89)

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
            t - start of data segment
            s - end of data segment
        """
        s += 1
        n = s - t

        mean = data[t:s].sum(0) / n
        muT = (n * mean) / (1 + n)
        nuT = 1 + n
        alphaT = 1 + n / 2

        betaT = (
            1
            + 0.5 * ((data[t:s] - mean) ** 2).sum(0)
            + ((n) / (1 + n)) * (mean ** 2 / 2)
        )
        scale = (betaT * (nuT + 1)) / (alphaT * nuT)

        # splitting the PDF of the student distribution up is /much/ faster.
        # (~ factor 20) using sum over for loop is even more worthwhile
        prob = np.sum(np.log(1 + (data[t:s] - muT) ** 2 / (nuT * scale)))
        lgA = (
            gammaln((nuT + 1) / 2)
            - np.log(np.sqrt(np.pi * nuT * scale))
            - gammaln(nuT / 2)
        )

        return np.sum(n * lgA - (nuT + 1) / 2 * prob)
