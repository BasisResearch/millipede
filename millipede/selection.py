import math
import time

import pandas as pd
import torch

from millipede import NormalLikelihoodSampler

from .containers import SimpleSampleContainer, StreamingSampleContainer
from .util import namespace_to_numpy


class NormalLikelihoodVariableSelector(object):
    def __init__(self, dataframe, response_column, S=5, c=100.0, explore=5, precompute_XX=False,
                 prior="isotropic", tau=0.01,
                 nu0=0.0, lambda0=0.0, precision="double", device="cpu"):

        if precision not in ['single', 'double']:
            raise ValueError("precision must be one of `single` or `double`")
        if device not in ['cpu', 'gpu']:
            raise ValueError("device must be one of `cpu` or `gpu`")
        if response_column not in dataframe.columns:
            raise ValueError("response_column must be a valid column in the dataframe.")

        X, Y = dataframe.drop(response_column, axis=1), dataframe[response_column]
        self.X_columns = X.columns

        if precision == 'single':
            X, Y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()
        elif precision == 'double':
            X, Y = torch.from_numpy(X.values).double(), torch.from_numpy(Y.values).double()

        if device == 'cpu':
            X, Y = X.cpu(), Y.cpu()
        elif device == 'gpu':
            X, Y = X.cuda(), Y.cuda()

        self.sampler = NormalLikelihoodSampler(X, Y, S=S, c=c, explore=explore,
                                               precompute_XX=precompute_XX, prior=prior, tau=tau,
                                               compute_betas=True, nu0=nu0, lambda0=lambda0)

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
                s += "\t# of active features: {}".format(sample.gamma.sum().item())
                if t >= report_frequency:
                    dt = 1000.0 * (ts[-1] - ts[-1 - report_frequency]) / report_frequency
                    s += "   mean iteration time: {:.2f} ms".format(dt)
                print(s)

        if not streaming:
            self.samples = container.samples

        self.pip = pd.Series(container.pip, index=self.X_columns, name="PIP")
        self.beta = pd.Series(container.beta, index=self.X_columns, name="Coefficient")
        self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns,
                                          name="Conditional Coefficient")
