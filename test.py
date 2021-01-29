from scipy.stats import multivariate_normal, norm
from functools import partial
import numpy as np
import bayesian_changepoint_detection.online_changepoint_detection as online


def test_multivariate():
    np.random.seed(seed=34)
    # 10-dimensional multivariate normal, that shifts its mean at t=50
    dataset = np.vstack((multivariate_normal.rvs([0]*10, size=50), multivariate_normal.rvs([2]*10, size=50)))
    r, maxes = online.online_changepoint_detection(
        dataset,
        partial(online.constant_hazard, 250),
        online.MultivariateT(dims=10)
    )
    # Assert that we detected the mean shift
    assert maxes[50] - maxes[51] > 40


def test_univariate():
    np.random.seed(seed=34)
    # 10-dimensional univariate normal
    dataset = np.hstack((norm.rvs(0, size=50), norm.rvs(2, size=50)))
    r, maxes = online.online_changepoint_detection(
        dataset,
        partial(online.constant_hazard, 20),
        online.StudentT(0.1, .01, 1, 0)
    )
    assert maxes[50] - maxes[51] > 40
