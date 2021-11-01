"""
Constellation data structure
"""
import numpy as np
from scipy.stats import norm

from nnv_py.nnv_py import PyConstellation  # pylint: disable=no-name-in-module

SAMPLE_STD_DEVS = 3.5


# pylint: disable=too-many-instance-attributes
class Constellation:
    """Probabilistic tree of Star Sets for importance sampling

    Caches intermediate progress in building the tree to speed up future sampling calls.
    Changing any of the parameters resets this cache.
    """
    def __init__(self, dnn, mean, scale, input_bounds=None):
        """
        Args:
            dnn (nnv_py.DNN): The DNN architecture to build on
            mean (ndarray): 1D Gaussian meanation parameter
            scale (ndarray): 2D Gaussian covariance parameter
            input_bounds (ndarray, optional): Lower and upper bounds on the input set
        """
        self.mean = np.squeeze(mean).astype(np.float64)
        self.scale = np.squeeze(scale).astype(np.float64)
        if dnn is None:
            self.constellation = None
        else:
            self.constellation = PyConstellation(dnn.dnn,
                                                 input_bounds, self.mean,
                                                 np.diag(self.scale))
        self.fixed_part = None
        self.max_value = np.inf
        self._sample_time_limit = np.inf

    def _set_mean(self, mean):
        if mean is None:
            return
        if mean.ndim == 2:
            assert len(mean) == 1
            mean = mean[0]
        self.mean = np.squeeze(mean).astype(np.float64)

    def set_dnn(self, dnn, mean=None, scale=None):
        """Change the DNN architecture to build on, resets the Constellation

        Args:
            dnn (nnv_py.DNN): The DNN architecture to build on
            mean (ndarray, optional): 1D Gaussian meanation parameter
            scale (ndarray, optional): 2D Gaussian covariance parameter
        """
        if mean is not None:
            self.mean = np.squeeze(mean).astype(np.float64)
        if scale is not None:
            self.scale = np.squeeze(scale).astype(np.float64)
        if self.constellation is None:
            bounds = None
        else:
            bounds = self.constellation.get_input_bounds()
        self.constellation = PyConstellation(dnn.dnn, bounds, self.mean,
                                             np.diag(self.scale))

    def set_input_bounds(self, fixed_part, mean=None, scale=None):
        """Change the parameters of the input Gaussian distribution, partly resets the Constellation

        Args:
            fixed_part (ndarray): 1D prefix of the DNN input
            mean (ndarray, optional): 1D Gaussian meanation parameter
            scale (ndarray, optional): 2D Gaussian covariance parameter
        """
        self._set_mean(mean)
        if scale is not None:
            self.scale = np.squeeze(scale).astype(np.float64)
        unfixed_part = (self.mean - SAMPLE_STD_DEVS * scale,
                        self.mean + SAMPLE_STD_DEVS * scale)
        self.constellation.set_input_bounds(fixed_part, unfixed_part)

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

    # pylint: disable=too-many-arguments
    def bounded_sample_input_with_params(self,
                                         fixed_part,
                                         mean=None,
                                         scale=None,
                                         max_value=None,
                                         sample_time_limit=None):
        """Accept-reject sample with new parameters, resets the Constellation

        Args:
            fixed_part (ndarray): 1D prefix of the DNN input
            mean (ndarray, optional): 1D Gaussian meanation parameter
            scale (ndarray, optional): 2D Gaussian covariance parameter
            max_value (float): Maximum value of DNN output for an input to be accepted
            sample_time_limit (float): the time allowed per sample in milliseconds
        """
        # Check whether a reset is necessary
        if fixed_part or mean or scale:
            fixed_part = np.squeeze(fixed_part).astype(np.float64)
            self.set_input_bounds(fixed_part, mean, scale)
        if max_value:
            self.max_value = max_value
        if sample_time_limit:
            self.sample_time_limit = sample_time_limit
        return self.bounded_sample_input()

    def _gaussian_log_prob(self, sample):
        prob = 1.
        for (samp, mean, scale) in zip(sample, self.mean, self.scale):
            dim_prob = norm.pdf(samp, mean, scale)
            if not np.all(np.isfinite(dim_prob)):
                raise ValueError("Bad dim_prob {dim_prob} from {samp} {l} {s}")
            prob *= dim_prob
        if not np.all(np.isfinite(np.log(prob + 1e-12))):
            raise ValueError("Bad logprob from prob {prob}")
        return np.log(prob + 1e-12)

    def _unbounded_gaussian_sample(self):
        sample = np.random.normal(self.mean, self.scale)
        return sample, self._gaussian_log_prob(sample)

    def bounded_sample_input(self):
        """
        Accept-reject sample, modifying the Gaussian distribution by rejecting all samples with
        value greater than `self.max_value`

        Returns:
            ([ndarray], [float]): tuple of samples and corresponding log probabilities
        """
        if self._max_value == np.inf:
            return self._unbounded_gaussian_sample()
        output = self.constellation.bounded_sample_input_multivariate_gaussian(
            self._max_value,
            cdf_samples=30,
            num_samples=1,
            max_iters=2,
            time_limit=self._sample_time_limit)

        # Handle the case where no safe sample is found
        # Currently this cops out and returns an unbounded sample
        # In the future it should call a provided function, or return `None` which would be handled
        # by the caller
        if output is None:
            return self._unbounded_gaussian_sample()
        sample, sample_logp, branch_logp = output
        normal_logp = self._gaussian_log_prob(sample)
        final_logp = (normal_logp - sample_logp) + branch_logp
        if not np.all(np.isfinite(sample)) or not np.all(
                np.isfinite(final_logp)):
            raise ValueError()
        return sample, final_logp