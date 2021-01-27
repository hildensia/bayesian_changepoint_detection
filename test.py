from scipy.stats import multivariate_normal
from functools import partial
import numpy as np
import bayesian_changepoint_detection.online_changepoint_detection as online
import bayesian_changepoint_detection.offline_changepoint_detection as offline


def test_multivariate():
    # 10-dimensional multivariate normal
    distr = multivariate_normal(mean=[0]*10)
    dataset = np.vstack((multivariate_normal.rvs([0]*10, size=50), multivariate_normal.rvs([2]*10, size=50)))
    r, maxes = online.online_changepoint_detection(
        dataset,
        partial(offline.const_prior, l=(len(dataset) + 1)),
        online.MultivariateT(10)
    )
    # TODO: assertion
