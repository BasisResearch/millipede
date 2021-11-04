import math
import time

import numpy as np
import pandas as pd
import torch

from millipede import CountLikelihoodSampler, NormalLikelihoodSampler

from .containers import SimpleSampleContainer, StreamingSampleContainer
from .util import namespace_to_numpy


class NormalLikelihoodVariableSelector(object):
    """
    Bayesian variable selection for a linear model with a Normal likelihood.
    The likelihood variance is controlled by a Inverse Gamma prior.
    """
    def __init__(self, dataframe, response_column, S=5, c=100.0, explore=5, precompute_XX=False,
                 prior="isotropic", tau=0.01, tau_intercept=1.0e-4,
                 nu0=0.0, lambda0=0.0, precision="double", device="cpu",
                 include_intercept=True):

        if precision not in ['single', 'double']:
            raise ValueError("precision must be one of `single` or `double`")
        if device not in ['cpu', 'gpu']:
            raise ValueError("device must be one of `cpu` or `gpu`")
        if response_column not in dataframe.columns:
            raise ValueError("response_column must be a valid column in the dataframe.")

        X, Y = dataframe.drop(response_column, axis=1), dataframe[response_column]
        self.X_columns = X.columns.tolist()

        if precision == 'single':
            X, Y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()
        elif precision == 'double':
            X, Y = torch.from_numpy(X.values).double(), torch.from_numpy(Y.values).double()

        if device == 'cpu':
            X, Y = X.cpu(), Y.cpu()
        elif device == 'gpu':
            X, Y = X.cuda(), Y.cuda()

        self.include_intercept = include_intercept
        self.sampler = NormalLikelihoodSampler(X, Y, S=S, c=c, explore=explore,
                                               precompute_XX=precompute_XX, prior=prior,
                                               tau=tau, tau_intercept=tau_intercept,
                                               compute_betas=True, nu0=nu0, lambda0=lambda0,
                                               include_intercept=include_intercept)

    def run(self, T=1000, T_burnin=500, verbose=True, report_frequency=100, streaming=True, seed=None):
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

        for t, (burned, sample) in enumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
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
            self.weights = self.samples.weight
        else:
            self.weights = np.array(container._weights)

        self.pip = pd.Series(container.pip, index=self.X_columns, name="PIP")
        if self.include_intercept:
            self.beta = pd.Series(container.beta, index=self.X_columns + ["intercept"], name="Coefficient")
            self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns + ["intercept"],
                                              name="Conditional Coefficient")
        else:
            self.beta = pd.Series(container.beta, index=self.X_columns, name="Coefficient")
            self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns,
                                              name="Conditional Coefficient")

        self.summary = pd.concat([self.pip, self.beta, self.conditional_beta], axis=1)

        self.stats = {}
        quantiles = [5.0, 10.0, 20.0, 50.0, 90.0, 95.0]
        q5, q10, q20, q50, q90, q95 = np.percentile(self.weights, quantiles).tolist()
        s = "5/10/20/50/90/95:  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
        self.stats['Weight quantiles'] = s.format(q5, q10, q20, q50, q90, q95)
        s = "mean/std/min/max:  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
        self.stats['Weight moments'] = s.format(self.weights.mean().item(), self.weights.std().item(),
                                                self.weights.min().item(), self.weights.max().item())
        elapsed_time = time.time() - ts[0]
        self.stats['Elapsed MCMC time'] = "{:.1f} seconds".format(elapsed_time)
        self.stats['Mean iteration time'] = "{:.3f} ms".format(1000.0 * elapsed_time / (T + T_burnin))
        self.stats['Number of retained samples'] = T
        self.stats['Number of burn-in samples'] = T_burnin

        if verbose:
            for k, v in self.stats.items():
                print('{}: '.format(k), v)


class BinomialLikelihoodVariableSelector(object):
    """
    Bayesian variable selection for a generalized linear model with a Binomial likelihood.
    """
    def __init__(self, dataframe, response_column, total_count_column,
                 S=5, explore=5, tau=0.01, tau_intercept=1.0e-4,
                 precision="double", device="cpu", xi_target=0.25):

        if precision not in ['single', 'double']:
            raise ValueError("precision must be one of `single` or `double`")
        if device not in ['cpu', 'gpu']:
            raise ValueError("device must be one of `cpu` or `gpu`")
        if response_column not in dataframe.columns:
            raise ValueError("response_column must be a valid column in the dataframe.")
        if total_count_column not in dataframe.columns:
            raise ValueError("total_count_column must be a valid column in the dataframe.")

        X = dataframe.drop([response_column, total_count_column], axis=1)
        Y = dataframe[response_column]
        TC = dataframe[total_count_column]
        self.X_columns = X.columns.tolist()

        if precision == 'single':
            X, Y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()
            TC = torch.from_numpy(TC.values).float()
        elif precision == 'double':
            X, Y = torch.from_numpy(X.values).double(), torch.from_numpy(Y.values).double()
            TC = torch.from_numpy(TC.values).double()

        if device == 'cpu':
            X, Y, TC = X.cpu(), Y.cpu(), TC.cpu()
        elif device == 'gpu':
            X, Y, TC = X.cuda(), Y.cuda(), TC.cuda()

        self.sampler = CountLikelihoodSampler(X, Y, TC=TC, S=S, explore=explore,
                                              tau=tau, tau_intercept=tau_intercept,
                                              xi_target=xi_target)

    def run(self, T=1000, T_burnin=500, verbose=True, report_frequency=100, streaming=True, seed=None):
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

        for t, (burned, sample) in enumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
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
            self.weights = self.samples.weight
        else:
            self.weights = np.array(container._weights)

        self.pip = pd.Series(container.pip, index=self.X_columns, name="PIP")
        self.beta = pd.Series(container.beta, index=self.X_columns + ['Intercept'], name="Coefficient")
        self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns + ['Intercept'],
                                          name="Conditional Coefficient")

        self.summary = pd.concat([self.pip, self.beta, self.conditional_beta], axis=1)

        self.stats = {}
        quantiles = [5.0, 10.0, 20.0, 50.0, 90.0, 95.0]
        q5, q10, q20, q50, q90, q95 = np.percentile(self.weights, quantiles).tolist()
        s = "5/10/20/50/90/95:  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
        self.stats['Weight quantiles'] = s.format(q5, q10, q20, q50, q90, q95)
        s = "mean/std/min/max:  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
        self.stats['Weight moments'] = s.format(self.weights.mean().item(), self.weights.std().item(),
                                                self.weights.min().item(), self.weights.max().item())
        elapsed_time = time.time() - ts[0]
        self.stats['Elapsed MCMC time'] = "{:.1f} seconds".format(elapsed_time)
        self.stats['Mean iteration time'] = "{:.3f} ms".format(1000.0 * elapsed_time / (T + T_burnin))
        self.stats['Number of retained samples'] = T
        self.stats['Number of burn-in samples'] = T_burnin

        if verbose:
            for k, v in self.stats.items():
                print('{}: '.format(k), v)


class NegativeBinomialLikelihoodVariableSelector(object):
    """
    Bayesian variable selection for a generalized linear model with a Negative Binomial likelihood.
    """
    def __init__(self, dataframe, response_column, psi0_column,
                 S=5, explore=5, tau=0.01, tau_intercept=1.0e-4,
                 precision="double", device="cpu", log_nu_rw_scale=0.05,
                 xi_target=0.25, init_nu=5.0):

        if precision not in ['single', 'double']:
            raise ValueError("precision must be one of `single` or `double`")
        if device not in ['cpu', 'gpu']:
            raise ValueError("device must be one of `cpu` or `gpu`")
        if response_column not in dataframe.columns:
            raise ValueError("response_column must be a valid column in the dataframe.")
        if psi0_column not in dataframe.columns:
            raise ValueError("psi0 must be a valid column in the dataframe.")

        X = dataframe.drop([response_column, psi0_column], axis=1)
        Y = dataframe[response_column]
        psi0 = dataframe[psi0_column]
        self.X_columns = X.columns.tolist()

        if precision == 'single':
            X, Y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()
            psi0 = torch.from_numpy(psi0.values).float()
        elif precision == 'double':
            X, Y = torch.from_numpy(X.values).double(), torch.from_numpy(Y.values).double()
            psi0 = torch.from_numpy(psi0.values).double()

        if device == 'cpu':
            X, Y, psi0 = X.cpu(), Y.cpu(), psi0.cpu()
        elif device == 'gpu':
            X, Y, psi0 = X.cuda(), Y.cuda(), psi0.cuda()

        self.sampler = CountLikelihoodSampler(X, Y, psi0=psi0, S=S, explore=explore,
                                              tau=tau, tau_intercept=tau_intercept,
                                              log_nu_rw_scale=log_nu_rw_scale,
                                              xi_target=xi_target, init_nu=init_nu)

    def run(self, T=1000, T_burnin=500, verbose=True, report_frequency=100, streaming=True, seed=None):
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

        for t, (burned, sample) in enumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
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
            self.weights = self.samples.weight
        else:
            self.weights = np.array(container._weights)

        self.pip = pd.Series(container.pip, index=self.X_columns, name="PIP")
        self.beta = pd.Series(container.beta, index=self.X_columns + ['Intercept'], name="Coefficient")
        self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns + ['Intercept'],
                                          name="Conditional Coefficient")

        self.summary = pd.concat([self.pip, self.beta, self.conditional_beta], axis=1)

        self.stats = {}
        quantiles = [5.0, 10.0, 20.0, 50.0, 90.0, 95.0]
        q5, q10, q20, q50, q90, q95 = np.percentile(self.weights, quantiles).tolist()
        s = "5/10/20/50/90/95:  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
        self.stats['Weight quantiles'] = s.format(q5, q10, q20, q50, q90, q95)
        s = "mean/std/min/max:  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
        self.stats['Weight moments'] = s.format(self.weights.mean().item(), self.weights.std().item(),
                                                self.weights.min().item(), self.weights.max().item())
        self.stats['Nu posterior mean'] = '{:.3f}'.format(np.exp(container.log_nu))

        elapsed_time = time.time() - ts[0]
        self.stats['Elapsed MCMC time'] = "{:.1f} seconds".format(elapsed_time)
        self.stats['Mean iteration time'] = "{:.3f} ms".format(1000.0 * elapsed_time / (T + T_burnin))
        self.stats['Number of retained samples'] = T
        self.stats['Number of burn-in samples'] = T_burnin

        if verbose:
            for k, v in self.stats.items():
                print('{}: '.format(k), v)
