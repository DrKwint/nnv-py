"""
Asterism data structure
"""
from os import path
import numpy as np
from numpy.lib.ufunclike import fix
from scipy.stats import norm

from nnv_py.nnv_py import PyAsterism  # pylint: disable=no-name-in-module
from nnv_py.util import gaussian_likelihood

SAMPLE_STD_DEVS = 1.96


# pylint: disable=too-many-instance-attributes
class Asterism:
    """Probabilistic tree of Star Sets for importance sampling

    Caches intermediate progress in building the tree to speed up future sampling calls.
    Changing any of the parameters resets this cache.
    """
    def __init__(self,
                 dnn,
                 mean,
                 scale,
                 input_bounds=None,
                 safe_value=np.inf,
                 cdf_samples=100,
                 max_iters=5,
                 stability_eps=1e-12):
        """
        Args:
            dnn (nnv_py.DNN): The DNN architecture to build on
            mean (ndarray): 1D Gaussian meanation parameter
            scale (ndarray): 2D Gaussian covariance parameter
            input_bounds (ndarray, optional): Lower and upper bounds on the input set
        """
        if dnn is None:
            self.asterism = None
        else:
            self.asterism = PyAsterism(dnn.dnn, input_bounds, mean,
                                       np.diag(scale), safe_value, cdf_samples,
                                       max_iters, stability_eps)
        self._sample_time_limit = None
        self.cached_infeasible_regions = {}

    def clear_cache(self):
        self.cached_infeasible_regions = {}

    def _set_distribution_params(self, mean=None, scale=None):
        if not (mean is None):
            if mean.ndim == 2:
                assert len(mean) == 1
                mean = mean[0]
            self.asterism.set_mean(np.squeeze(mean).astype(np.float64))

        if not (scale is None):
            if scale.ndim == 1:
                scale = np.diag(scale)
            self.asterism.set_scale(scale.astype(np.float64))

    def set_dnn(self,
                dnn,
                mean=None,
                scale=None,
                cdf_samples=100,
                max_iters=5,
                stability_eps=1e-12):
        """Change the DNN architecture to build on, resets the Asterism

        Args:
            dnn (nnv_py.DNN): The DNN architecture to build on
            mean (ndarray, optional): 1D Gaussian meanation parameter
            scale (ndarray, optional): 2D Gaussian covariance parameter
        """
        if self.asterism is None:
            input_bounds = None
        else:
            input_bounds = self.asterism.get_input_bounds()
        if mean is None:
            mean = self.asterism.get_mean()
        if scale is None:
            scale = self.asterism.get_scale()
        if scale.ndim == 1:
            scale = np.diag(scale)
        safe_value = self.asterism.get_safe_value()
        self.asterism = PyAsterism(dnn.dnn, input_bounds, mean, scale,
                                   safe_value, cdf_samples, max_iters,
                                   stability_eps)

    def set_input_bounds(self, fixed_part, mean, scale):
        """Change the parameters of the input Gaussian distribution, partly resets the Asterism

        Args:
            fixed_part (ndarray): 1D prefix of the DNN input
            mean (ndarray, optional): 1D Gaussian mean parameter
            scale (ndarray, optional): 1D Gaussian variance parameter
        """
        if mean.ndim == 2:
            mean = np.squeeze(mean)
        self._set_distribution_params(mean, scale)
        unfixed_part = ((mean - SAMPLE_STD_DEVS * scale).astype(np.float64),
                        (mean + SAMPLE_STD_DEVS * scale).astype(np.float64))
        self.asterism.set_input_bounds(fixed_part, unfixed_part)

    @property
    def max_value(self):
        """Returns maximum allowed output value"""
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        """
        Args:
            max_value (float): Maximum value of DNN output for an input to be accepted
        """
        self._max_value = value

    @property
    def sample_time_limit(self):
        """Returns the time allowed per sample in milliseconds"""
        return self._sample_time_limit

    @sample_time_limit.setter
    def sample_time_limit(self, value):
        """
        Args:
            sample_time_limit (float): the time allowed per sample in milliseconds
        """
        self._sample_time_limit = value

    def set_safe_value(self, value):
        self.asterism.set_safe_value(value)

    def _calculate_single_infeasible_cdf(self, fixed_part, loc, scale):
        sum_cdf = 0.
        for region in self.cached_infeasible_regions[fixed_part]:
            sum_cdf += region.diag_gaussian_cdf(loc, scale)
        return np.array(sum_cdf)

    def calculate_infeasible_cdf(self, fixed_part, loc, scale):
        fixed_part = np.squeeze(fixed_part)
        loc = np.squeeze(fixed_part)
        scale = tuple(scale)
        if len(fixed_part.shape) == 1:
            return self._calculate_single_infeasible_cdf(
                tuple(fixed_part), tuple(loc), scale)
        elif len(fixed_part.shape) == 2:
            return np.stack([
                self._calculate_single_infeasible_cdf(tuple(x), tuple(l),
                                                      scale)
                for (x, l) in zip(fixed_part, loc)
            ])
        else:
            raise Exception()

    def get_samples_and_overapproximated_infeasible_input_regions(
            self,
            num_samples,
            fixed_part,
            mean=None,
            scale=None,
            max_value=None,
            sample_time_limit=None):
        # Check whether a reset is necessary
        if fixed_part is not None or mean is not None or scale is not None:
            fixed_part = np.squeeze(fixed_part).astype(np.float64)
            self.set_input_bounds(fixed_part, mean, scale)
        if max_value:
            self.asterism.set_safe_value(max_value)
        if sample_time_limit:
            self.sample_time_limit = sample_time_limit

        if len(fixed_part.shape) > 1:
            if len(fixed_part) > 2 or fixed_part.shape[0] != 1:
                raise Exception()
            fixed_part = fixed_part[0]
        fixed_part = tuple(fixed_part)
        output = self.asterism.get_samples_and_overapproximated_infeasible_input_regions(
            num_samples,
            num_intermediate_samples=100,
            time_limit=self._sample_time_limit)
        if output is None:
            self.cached_infeasible_regions[fixed_part] = []
            return np.random.normal(self.asterism.get_mean(),
                                    self.asterism.get_scale())
        sample_chunks, self.cached_infeasible_regions[fixed_part] = output
        sample = np.concatenate(sample_chunks)[:num_samples]
        return sample

    # pylint: disable=too-many-arguments
    def bounded_sample_input_with_params(self,
                                         fixed_part,
                                         mean=None,
                                         scale=None,
                                         max_value=None,
                                         sample_time_limit=None):
        """Accept-reject sample with new parameters, resets the Asterism

        Args:
            fixed_part (ndarray): 1D prefix of the DNN input
            mean (ndarray, optional): 1D Gaussian meanation parameter
            scale (ndarray, optional): 2D Gaussian covariance parameter
            max_value (float): Maximum value of DNN output for an input to be accepted
            sample_time_limit (float): the time allowed per sample in milliseconds
        """
        # Check whether a reset is necessary
        if fixed_part is not None or mean is not None or scale is not None:
            fixed_part = np.squeeze(fixed_part).astype(np.float64)
            self.set_input_bounds(fixed_part, mean, scale)
        if max_value:
            self.max_value = max_value
        if sample_time_limit:
            self.sample_time_limit = sample_time_limit
        return self.bounded_sample_input()

    def _unbounded_gaussian_sample(self):
        mean = self.asterism.get_mean()
        scale = self.asterism.get_scale()
        sample = np.random.normal(mean, scale)
        return sample, gaussian_likelihood(sample, mean, np.log(scale)), 0.

    def bounded_sample_input(self):
        """
        Accept-reject sample, modifying the Gaussian distribution by rejecting all samples with
        value greater than `self.max_value`

        Returns:
            ([ndarray], [float]): tuple of samples and corresponding log probabilities
        """
        if self._max_value == np.inf:
            return self._unbounded_gaussian_sample()
        output = self.asterism.bounded_sample_input_multivariate_gaussian(
            self._max_value,
            cdf_samples=250,
            num_samples=100,
            max_iters=5,
            time_limit=self._sample_time_limit,
            stability_eps=1e-10)

        # Handle the case where no safe sample is found
        # Currently this cops out and returns an unbounded sample
        # In the future it should call a provided function, or return `None` which would be handled
        # by the caller
        if output is None:
            return self._unbounded_gaussian_sample()
        sample, path_logp, invalid_cdf_proportion = output
        mean = self.asterism.get_mean()
        scale = self.asterism.get_scale()
        normal_logp = gaussian_likelihood(sample, mean, np.log(scale))
        invalid_logp = np.log(1. - invalid_cdf_proportion)
        if not np.all(np.isfinite(sample)) or not np.all(
                np.isfinite(normal_logp)):
            raise ValueError()
        if (normal_logp - invalid_logp) < -10:
            return self._unbounded_gaussian_sample()
        return sample, normal_logp, invalid_logp + 0.048665171
