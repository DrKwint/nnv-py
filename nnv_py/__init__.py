"""
A library for analyzing deep neural networks with verification techniques
"""
import numpy as np
from nnv_py.constellation import Constellation
from nnv_py.dnn import DNN
from nnv_py._version import version as __version__
from nnv_py.nnv_py import halfspace_gaussian_cdf

__all__ = ['DNN', 'Constellation', 'halfspace_gaussian_cdf']


def halfspace_gaussian_cdfs(coeffs, rhs, mu, sigma):
    return halfspace_gaussian_cdf(coeffs[0].astype(np.float64), rhs,
                                  mu[0].astype(np.float64),
                                  sigma.astype(np.float64))
