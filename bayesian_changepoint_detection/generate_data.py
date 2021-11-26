from __future__ import division
import numpy as np


def generate_normal_time_series(num, minl: int=50, maxl: int=1000, seed: int=100):
    np.random.seed(seed)
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn() * 10
        var = np.random.randn() * 1
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return partition, np.atleast_2d(data).T


def generate_multinormal_time_series(num, dim, minl: int=50, maxl: int=1000, seed: int=100):
    np.random.seed(seed)
    data = np.empty((1, dim), dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.standard_normal(dim) * 10
        # Generate a random SPD matrix
        A = np.random.standard_normal((dim, dim))
        var = np.dot(A, A.T)

        tdata = np.random.multivariate_normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return partition, data[1:, :]


def generate_xuan_motivating_example(minl: int=50, maxl: int=1000, seed: int=100):
    np.random.seed(seed)
    dim = 2
    num = 3
    partition = np.random.randint(minl, maxl, num)
    mu = np.zeros(dim)
    Sigma1 = np.asarray([[1.0, 0.75], [0.75, 1.0]])
    data = np.random.multivariate_normal(mu, Sigma1, partition[0])
    Sigma2 = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    data = np.concatenate(
        (data, np.random.multivariate_normal(mu, Sigma2, partition[1]))
    )
    Sigma3 = np.asarray([[1.0, -0.75], [-0.75, 1.0]])
    data = np.concatenate(
        (data, np.random.multivariate_normal(mu, Sigma3, partition[2]))
    )
    return partition, data
