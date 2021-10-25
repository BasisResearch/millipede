import math
import time
from functools import cached_property

import numpy as np
import torch

from millipede import NormalLikelihoodSampler

from .util import namespace_to_numpy, stack_namespaces


class SimpleSampleContainer(object):
    def __init__(self):
        self._raw_samples = []

    def __call__(self, sample):
        self._raw_samples.append(sample)

    @cached_property
    def samples(self):
        samples = stack_namespaces(self._raw_samples)
        del self._raw_samples
        return samples

    @cached_property
    def weights(self):
        weights = self.samples.weight
        return weights / weights.sum()

    @cached_property
    def pip(self):
        return np.dot(self.samples.add_prob.T, self.weights)


class StreamingSampleContainer(object):
    def __init__(self):
        self._raw_pip = None
        self._num_samples = 0.0
        self._weights = []

    def __call__(self, sample):
        self._weights.append(sample.weight)
        self._num_samples += 1.0
        if self._raw_pip is None:
            self._raw_pip = sample.add_prob * sample.weight
        else:
            self._raw_pip = (1.0 - 1.0 / self._num_samples) * self._raw_pip +\
                (sample.add_prob * sample.weight) / self._num_samples

    @cached_property
    def pip(self):
        return len(self._weights) * self._raw_pip / np.array(self._weights).sum()


class NormalLikelihoodVariableSelector(object):
    def __init__(self, dataframe, response_column, S=5, c=100.0, explore=5, precompute_XX=False,
                 prior="isotropic", tau=0.01, compute_betas=True,
                 nu0=0.0, lambda0=0.0, precision="double"):

        if precision not in ['single', 'double']:
            raise ValueError("precision must be one of `single` or `double`")
        if response_column not in dataframe.columns:
            raise ValueError("response_column must be a valid column in the dataframe.")

        self.index = dataframe.index.values
        X = dataframe.drop(response_column, axis=1).values
        Y = dataframe[response_column].values

        if precision == 'single':
            X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
        elif precision == 'double':
            X, Y = torch.from_numpy(X).double(), torch.from_numpy(Y).double()

        self.sampler = NormalLikelihoodSampler(X, Y, S=S, c=c, explore=explore,
                                               precompute_XX=precompute_XX, prior=prior, tau=tau,
                                               compute_betas=compute_betas, nu0=nu0, lambda0=lambda0)

    def run(self, T=1000, T_burnin=500, verbose=True, report_frequency=100, streaming=True):
        if not isinstance(T, int) and T > 0:
            raise ValueError("T must be a positive integer.")
        if not isinstance(T_burnin, int) and T_burnin > 0:
            raise ValueError("T_burnin must be a positive integer.")

        if streaming:
            container = StreamingSampleContainer()
        else:
            container = SimpleSampleContainer()

        ts = [time.time()]
        digits_to_print = str(1 + int(math.log(T + T_burnin + 1, 10)))

        for t, (burned, sample) in enumerate(self.sampler.gibbs_chain(T=T, T_burnin=T_burnin)):
            ts.append(time.time())
            if burned:
                container(namespace_to_numpy(sample))
            if verbose and t % report_frequency == 0 or t == T + T_burnin - 1:
                s = ("[Iteration {:0" + digits_to_print + "d}]").format(t)
                s += "\tnumber of active features: {}".format(sample.gamma.sum().item())
                if t >= report_frequency:
                    dt = 1000.0 * (ts[-1] - ts[-1 - report_frequency]) / report_frequency
                    s += "\taverage iteration time: {:.4f} ms".format(dt)
                print(s)

        if not streaming:
            self.samples = container.samples

        self.pip = container.pip
