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
        return self.beta / np.dot(self.samples.gamma.T, self.weights)


class StreamingSampleContainer(object):
    def __init__(self):
        self._num_samples = 0.0
        self._weights = []

    def __call__(self, sample):
        self._weights.append(sample.weight)
        self._num_samples += 1.0
        if self._num_samples == 1.0:
            self._raw_pip = sample.add_prob * sample.weight
            self._raw_beta = sample.beta * sample.weight
            self._raw_gamma = sample.gamma * sample.weight
        else:
            factor = 1.0 - 1.0 / self._num_samples
            self._raw_pip = factor * self._raw_pip + (sample.beta * sample.weight) / self._num_samples
            self._raw_beta = factor * self._raw_beta + (sample.beta * sample.weight) / self._num_samples
            self._raw_gamma = factor * self._raw_gamma + (sample.gamma * sample.weight) / self._num_samples

    @cached_property
    def _normalizer(self):
        return len(self._weights) / np.array(self._weights).sum()

    @cached_property
    def pip(self):
        return self._normalizer * self._raw_pip

    @cached_property
    def beta(self):
        return self._normalizer * self._raw_beta

    @cached_property
    def conditional_beta(self):
        return self._raw_beta / self._raw_gamma
