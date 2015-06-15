''' Example from Xiang Xuan's thesis: Section 3.2'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import bayesian_changepoint_detection.offline_changepoint_detection as offcd
import bayesian_changepoint_detection.generate_data as gd
from functools import partial

if __name__ == '__main__':
  show_plot = True

  partition, data = gd.generate_xuan_motivating_example(50,200)
  changes = np.cumsum(partition)

  Q_ifm, P_ifm, Pcp_ifm = offcd.offline_changepoint_detection(data,partial(offcd.const_prior, l=(len(data)+1)),offcd.ifm_obs_log_likelihood,truncate=-20)
  Q_full, P_full, Pcp_full = offcd.offline_changepoint_detection(data,partial(offcd.const_prior, l=(len(data)+1)),offcd.fullcov_obs_log_likelihood, truncate=-20)

  if show_plot:
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(3, 1, 1)
    for p in changes:
      ax.plot([p,p],[np.min(data),np.max(data)],'r')
    for d in range(2):
      ax.plot(data[:,d])
    plt.legend(['Raw data with Original Changepoints'])
    ax1 = fig.add_subplot(3, 1, 2, sharex=ax)
    ax1.plot(np.exp(Pcp_ifm).sum(0))
    plt.legend(['Independent Factor Model'])
    ax2 = fig.add_subplot(3, 1, 3, sharex=ax)
    ax2.plot(np.exp(Pcp_full).sum(0))
    plt.legend(['Full Covariance Model'])
    plt.show()

