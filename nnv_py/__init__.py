"""
A library for analyzing deep neural networks with verification techniques
"""
import numpy as np
from nnv_py.asterism import Asterism
from nnv_py.dnn import DNN
from nnv_py._version import version as __version__
from nnv_py.nnv_py import halfspace_gaussian_cdf
from nnv_py.starset import StarSet

__all__ = ['DNN', 'StarSet', 'Asterism', 'halfspace_gaussian_cdf']


def halfspace_gaussian_cdfs(coeffs, rhs, mu, sigma):
    if not np.any(coeffs) or all(np.isposinf(rhs)):
        return np.array([1.])
    cdfs = []
    for (c, m, r) in zip(coeffs, mu, rhs):
        if any(np.isnan(c)) or any(np.isnan(m)):
            continue
        cdfs.append(
            halfspace_gaussian_cdf(c.astype(np.float64), r,
                                   m.astype(np.float64),
                                   sigma.astype(np.float64)))
    if not cdfs:
        cdfs = [1.]
    return np.array(cdfs)
