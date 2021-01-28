from scipy.stats import multivariate_normal, norm
from functools import partial
import numpy as np
import bayesian_changepoint_detection.online_changepoint_detection as online
import bayesian_changepoint_detection.offline_changepoint_detection as offline


def test_multivariate():
    # 10-dimensional multivariate normal
    dataset = np.vstack((multivariate_normal.rvs([0]*10, size=50), multivariate_normal.rvs([2]*10, size=50)))
    r, maxes = online.online_changepoint_detection(
        dataset,
        partial(online.constant_hazard, 250),
        online.MultivariateT(10)
    )
    print(r)
    # TODO: assertion


def test_univariate():
    # 10-dimensional multivariate normal
    dataset = np.hstack((norm.rvs(0, size=50), norm.rvs(2, size=50)))
    r, maxes = online.online_changepoint_detection(
        dataset,
        partial(online.constant_hazard, 20),
        online.StudentT(0.1, .01, 1, 0)
    )
    print(r)
    # TODO: assertion
