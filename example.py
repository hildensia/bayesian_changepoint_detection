from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import cProfile
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial

import bayesian_changepoint_detection.online_changepoint_detection as oncd
import matplotlib.cm as cm

def generate_normal_time_series(num, minl=50, maxl=1000):
  data = np.array([], dtype=np.float64)
  partition = np.random.randint(minl, maxl, num)
  for p in partition:
    mean = np.random.randn()*10
    var = np.random.randn()*1
    if var < 0:
      var = var * -1
    tdata = np.random.normal(mean, var, p)
    data = np.concatenate((data, tdata))
  return partition, np.atleast_2d(data).T

def generate_multinormal_time_series(num, dim, minl=50, maxl=1000):
  data = np.empty((1,dim), dtype=np.float64)
  partition = np.random.randint(minl, maxl, num)
  for p in partition:
    mean = np.random.standard_normal(dim)*10
    # Generate a random SPD matrix
    A = np.random.standard_normal((dim,dim))
    var = np.dot(A,A.T)

    tdata = np.random.multivariate_normal(mean, var, p)
    data = np.concatenate((data, tdata))
  return partition, data[1:,:]

if __name__ == '__main__':
  show_plot = True
  dim = 4
  #partition, data = generate_normal_time_series(7, 50, 200)
  partition, data = generate_multinormal_time_series(7, dim, 50, 200)
  changes = np.cumsum(partition)

  if show_plot:
    fig, ax = plt.subplots(figsize=[16,12])
    for p in changes:
      ax.plot([p,p],[np.min(data),np.max(data)],'r')
    for d in range(dim):
      ax.plot(data[:,d])
    plt.show()


  #Q, P, Pcp = offcd.offline_changepoint_detection(data,partial(offcd.const_prior, l=(len(data)+1)),offcd.gaussian_obs_log_likelihood, truncate=-20)
  #Q, P, Pcp = offcd.offline_changepoint_detection(data,partial(offcd.const_prior, l=(len(data)+1)),offcd.ifm_obs_log_likelihood, truncate=-20)
  Q, P, Pcp = offcd.offline_changepoint_detection(data,partial(offcd.const_prior, l=(len(data)+1)),offcd.fullcov_obs_log_likelihood, truncate=-20)

  if show_plot:
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(2, 1, 1)
    for p in changes:
      ax.plot([p,p],[np.min(data),np.max(data)],'r')
    for d in range(dim):
      ax.plot(data[:,d])
    ax = fig.add_subplot(2, 1, 2, sharex=ax)
    ax.plot(np.exp(Pcp).sum(0))
    plt.show()

  '''R, maxes = oncd.online_changepoint_detection(data,partial(oncd.constant_hazard, 250), oncd.StudentT(10, .03, 1, 0))

  if show_plot:
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(data)
    for p in changes:
      ax.plot([p,p],[np.min(data),np.max(data)],'r')
    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    sparsity = 5  # only plot every fifth data for faster display
    ax.pcolor(np.array(range(0, len(R[:,0]), sparsity)), \
          np.array(range(0, len(R[:,0]), sparsity)), \
          -np.log(R[0:-1:sparsity, 0:-1:sparsity]), \
          cmap=cm.Greys, vmin=0, vmax=1000)
    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    ax.plot(R[:, 1])
    plt.show()
  '''
