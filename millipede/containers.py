from functools import cached_property

import numpy as np

from .util import stack_namespaces


class SimpleSampleContainer(object):
    """
    Class used to store MCMC samples and compute summary statistics.
    All samples are kept in memory.
    """
    def __init__(self):
        self._samples = []

    def __call__(self, sample):
        self._samples.append(sample)

    @cached_property
    def samples(self):
        samples = stack_namespaces(self._samples)
        del self._samples
        return samples

    @cached_property
    def weights(self):
        weights = self.samples.weight
        return weights / weights.sum()

    @cached_property
    def pip(self):
        return np.dot(self.samples.pip.T, self.weights)

    @cached_property
    def beta(self):
        return np.dot(self.samples.beta.T, self.weights)

    @cached_property
    def beta_std(self):
        return np.sqrt(np.dot(np.square(self.samples.beta.T), self.weights) - np.square(self.beta))

    @cached_property
    def log_nu(self):
        return np.dot(self.samples.log_nu, self.weights).item()

    @cached_property
    def log_nu_std(self):
        return np.sqrt(np.dot(np.square(self.samples.log_nu), self.weights) - self.log_nu ** 2).item()

    @cached_property
    def sigma(self):
        return np.dot(self.samples.sigma, self.weights).item()

    @cached_property
    def sigma_std(self):
        return np.sqrt(np.dot(np.square(self.samples.sigma), self.weights) - self.sigma ** 2).item()

    @cached_property
    def nu(self):
        return np.dot(np.exp(self.samples.log_nu), self.weights).item()

    @cached_property
    def nu_std(self):
        return np.sqrt(np.dot(np.exp(2.0 * self.samples.log_nu), self.weights) - self.nu ** 2).item()

    @cached_property
    def h_alpha(self):
        return np.dot(self.samples.h_alpha, self.weights).item()

    @cached_property
    def h_beta(self):
        return np.dot(self.samples.h_beta, self.weights).item()

    @cached_property
    def h(self):
        return np.dot(self.samples.h_alpha / (self.samples.h_alpha + self.samples.h_beta), self.weights).item()

    @cached_property
    def conditional_beta(self):
        divisor = np.dot(self.samples.gamma.T, self.weights)
        delta = self.beta.shape[0] - divisor.shape[0]
        divisor = np.concatenate([divisor, delta * [1.0]]) if delta > 0 else divisor
        return np.true_divide(self.beta, divisor, where=divisor != 0, out=np.zeros(self.beta.shape))

    @cached_property
    def conditional_beta_std(self):
        divisor = np.dot(self.samples.gamma.T, self.weights)
        delta = self.beta.shape[0] - divisor.shape[0]
        divisor = np.concatenate([divisor, delta * [1.0]]) if delta > 0 else divisor
        beta_sq = np.dot(np.square(self.samples.beta.T), self.weights)
        beta_sq = np.true_divide(beta_sq, divisor, where=divisor != 0, out=np.zeros(self.beta.shape))
        return np.sqrt(np.clip(beta_sq - np.square(self.conditional_beta), a_min=0.0, a_max=None))


class StreamingSampleContainer(object):
    """
    Class used to process MCMC samples and compute summary statistics.
    Instead of storing all MCMC samples in memory, summary statistics are computed online.
    """
    def __init__(self):
        self._num_samples = 0.0
        self._weight_sum = 0.0
        self._weights = []

    def __call__(self, sample):
        self._weight_sum += sample.weight
        self._num_samples += 1.0
        self._weights.append(sample.weight)

        if self._num_samples == 1.0:
            self._pip = sample.pip * sample.weight
            self._beta = sample.beta * sample.weight
            self._beta_sq = np.square(sample.beta) * sample.weight
            self._gamma = sample.gamma * sample.weight
            if hasattr(sample, 'sigma'):
                self._sigma = sample.sigma * sample.weight
                self._sigma_sq = np.square(sample.sigma) * sample.weight
            if hasattr(sample, 'log_nu'):
                self._log_nu = sample.log_nu * sample.weight
                self._log_nu_sq = np.square(sample.log_nu) * sample.weight
                self._nu = np.exp(sample.log_nu) * sample.weight
                self._nu_sq = np.exp(2.0 * sample.log_nu) * sample.weight
            if hasattr(sample, 'h_alpha'):
                self._h_alpha = sample.h_alpha * sample.weight
                self._h_beta = sample.h_beta * sample.weight
                self._h = (sample.h_alpha / (sample.h_alpha + sample.h_beta)) * sample.weight
        else:
            factor = 1.0 - 1.0 / self._num_samples
            self._pip = factor * self._pip + (sample.pip * sample.weight) / self._num_samples
            self._beta = factor * self._beta + (sample.beta * sample.weight) / self._num_samples
            self._beta_sq = factor * self._beta_sq + (np.square(sample.beta) * sample.weight) / self._num_samples
            self._gamma = factor * self._gamma + (sample.gamma * sample.weight) / self._num_samples
            if hasattr(sample, 'sigma'):
                self._sigma = factor * self._sigma + (sample.sigma * sample.weight) / self._num_samples
                self._sigma_sq = factor * self._sigma_sq + (np.square(sample.sigma) * sample.weight) / self._num_samples
            if hasattr(sample, 'log_nu'):
                self._log_nu = factor * self._log_nu + (sample.log_nu * sample.weight) / self._num_samples
                self._log_nu_sq = factor * self._log_nu_sq +\
                    (np.square(sample.log_nu) * sample.weight) / self._num_samples
                self._nu = factor * self._nu + (np.exp(sample.log_nu) * sample.weight) / self._num_samples
                self._nu_sq = factor * self._nu_sq + (np.exp(2.0 * sample.log_nu) * sample.weight) / self._num_samples
            if hasattr(sample, 'h_alpha'):
                self._h_alpha = factor * self._h_alpha + (sample.h_alpha * sample.weight) / self._num_samples
                self._h_beta = factor * self._h_beta + (sample.h_beta * sample.weight) / self._num_samples
                h_alpha_beta = sample.h_alpha / (sample.h_alpha + sample.h_beta)
                self._h = factor * self._h + h_alpha_beta * sample.weight / self._num_samples

    @cached_property
    def _normalizer(self):
        return self._num_samples / self._weight_sum

    @cached_property
    def pip(self):
        return self._normalizer * self._pip

    @cached_property
    def beta(self):
        return self._normalizer * self._beta

    @cached_property
    def beta_std(self):
        return np.sqrt(self._normalizer * self._beta_sq - np.square(self.beta))

    @cached_property
    def log_nu(self):
        return self._normalizer * self._log_nu

    @cached_property
    def log_nu_std(self):
        return np.sqrt(self._normalizer * self._log_nu_sq - self.log_nu ** 2).item()

    @cached_property
    def sigma(self):
        return self._normalizer * self._sigma

    @cached_property
    def sigma_std(self):
        return np.sqrt(self._normalizer * self._sigma_sq - self.sigma ** 2).item()

    @cached_property
    def nu(self):
        return (self._normalizer * self._nu).item()

    @cached_property
    def nu_std(self):
        return np.sqrt(self._normalizer * self._nu_sq - self.nu ** 2).item()

    @cached_property
    def h_alpha(self):
        return self._normalizer * self._h_alpha

    @cached_property
    def h_beta(self):
        return self._normalizer * self._h_beta

    @cached_property
    def h(self):
        return self._normalizer * self._h

    @cached_property
    def conditional_beta(self):
        delta = self._beta.shape[0] - self._gamma.shape[0]
        gamma = np.concatenate([self._gamma, delta * [1.0 / self._normalizer]]) if delta > 0 else self._gamma
        return np.true_divide(self._beta, gamma, where=gamma != 0, out=np.zeros(self._beta.shape))

    @cached_property
    def conditional_beta_std(self):
        delta = self._beta.shape[0] - self._gamma.shape[0]
        gamma = np.concatenate([self._gamma, delta * [1.0 / self._normalizer]]) if delta > 0 else self._gamma
        beta_sq = np.true_divide(self._beta_sq, gamma, where=gamma != 0, out=np.zeros(self._beta.shape))
        return np.sqrt(np.clip(beta_sq - np.square(self.conditional_beta), a_min=0.0, a_max=None))
