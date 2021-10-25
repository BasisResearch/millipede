import torch
import numpy as np
from millipede import NormalLikelihoodSampler
from .util import namespace_to_numpy, stack_namespaces


class VariableSelector(object):
    def __init__(self):
        pass


class NormalLikelihoodVariableSelector(VariableSelector):
    def __init__(self, dataframe, response_column, S=5, c=100.0, explore=5, precompute_XX=False,
                 prior="isotropic", tau=0.01, compute_betas=False,
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

    def run(self, T=1000, T_burnin=500, verbose=True, report_frequency=100):
        if not isinstance(T, int) and T > 0:
            raise ValueError("T must be a positive integer.")
        if not isinstance(T_burnin, int) and T_burnin > 0:
            raise ValueError("T_burnin must be a positive integer.")

        samples = []

        for t, (burned, s) in enumerate(self.sampler.gibbs_chain(T=T, T_burnin=T_burnin)):
            if burned:
                samples.append(namespace_to_numpy(s))

        samples = stack_namespaces(samples)
        weights = samples.weight / samples.weight.sum()

        self.pip = np.dot(samples.add_prob.T, weights)
