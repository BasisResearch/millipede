from functools import cached_property

import numpy as np

from .util import stack_namespaces


class SimpleSampleContainer(object):
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
        return np.dot(self.samples.add_prob.T, self.weights)

    @cached_property
    def beta(self):
        return np.dot(self.samples.beta.T, self.weights)

    @cached_property
    def conditional_beta(self):
        divisor = np.dot(self.samples.gamma.T, self.weights)
        return np.true_divide(self.beta, divisor, where=divisor != 0, out=np.zeros(self.beta.shape))


class StreamingSampleContainer(object):
    def __init__(self):
        self._num_samples = 0.0
        self._weight_sum = 0.0

    def __call__(self, sample):
        self._weight_sum += sample.weight
        self._num_samples += 1.0
        if self._num_samples == 1.0:
            self._pip = sample.add_prob * sample.weight
            self._beta = sample.beta * sample.weight
            self._gamma = sample.gamma * sample.weight
        else:
            factor = 1.0 - 1.0 / self._num_samples
            self._pip = factor * self._pip + (sample.add_prob * sample.weight) / self._num_samples
            self._beta = factor * self._beta + (sample.beta * sample.weight) / self._num_samples
            self._gamma = factor * self._gamma + (sample.gamma * sample.weight) / self._num_samples

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
    def conditional_beta(self):
        return np.true_divide(self._beta, self._gamma, where=self._gamma != 0, out=np.zeros(self.beta.shape))
