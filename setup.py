#!/usr/bin/env python2

from distutils.core import setup
import bayesian_changepoint_detection

setup(name='bayesian_changepoint_detection',
      version=bayesian_changepoint_detection.__version__,
      description='Some Bayesian changepoint detection algorithms',
      author='Johannes Kulick',
      author_email='johannes.kulick@ipvs.uni-stuttgart.de',
      url='http://github.com/hildensia/bayesian_changepoint_detection',
      packages = ['bayesian_changepoint_detection'],
      requires=['scipy', 'numpy']
     )
