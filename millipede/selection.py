import math
import time

import numpy as np
import pandas as pd
import torch
from tqdm.contrib import tenumerate

from millipede import CountLikelihoodSampler, NormalLikelihoodSampler

from .containers import SimpleSampleContainer, StreamingSampleContainer
from .util import namespace_to_numpy


def populate_weight_stats(stats, weights, quantiles=[5.0, 10.0, 20.0, 50.0, 90.0, 95.0]):
    q5, q10, q20, q50, q90, q95 = np.percentile(weights, quantiles).tolist()
    s = "5/10/20/50/90/95:  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
    stats['Weight quantiles'] = s.format(q5, q10, q20, q50, q90, q95)
    s = "mean/std/min/max:  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
    stats['Weight moments'] = s.format(weights.mean().item(), weights.std().item(),
                                       weights.min().item(), weights.max().item())


class NormalLikelihoodVariableSelector(object):
    r"""
    Bayesian variable selection for a linear model with a Normal likelihood.
    The likelihood variance is controlled by an Inverse Gamma prior.
    This class is appropriate for continuous-valued responses.

    Usage::

        selector = NormalLikelihoodVariableSelector(dataframe, 'response', ...)
        selector.run(T=2000, T_burnin=1000)
        print(selector.summary)

    The details of the model used in :class:`NormalLikelihoodVariableSelector` are as follows.
    The covariates :math:`X` and responses :math:`Y` are defined as follows:

    .. math::

        X \in \mathbb{R}^{N \times P} \qquad \qquad Y \in \mathbb{R}^{N}

    and are provided by the user. The user should put some thought into whether the covariates :math:`X`
    and responses :math:`Y` should be centered and/or normalized. This is generally a good idea for the responses
    :math:`Y`, but whether pre-processing for :math:`X` is advisable depends on the nature of the dataset.

    The inclusion of each covariate is governed by a Bernoulli random variable :math:`\gamma_p`.
    In particular :math:`\gamma_p = 0` corresponds to exclusion and :math:`\gamma_p = 1` corresponds to inclusion.
    The prior probability of inclusion is governed by :math:`h` or alternatively :math:`S`:

    .. math::

        h \in [0, 1] \qquad \rm{with} \qquad S \equiv hP

    Putting this together, the model specification for an isotopric prior (with an intercept
    :math:`\beta_0` included) is as follows:

    .. math::

        &\gamma_p \sim \rm{Bernoulli}(h) \qquad \rm{for} \qquad p=1,2,...,P

        &\sigma^2 \sim \rm{InverseGamma}(\nu_0 / 2, \nu_0 \lambda_0 / 2)

        &\beta_0 \sim \rm{Normal}(0, \sigma^2\tau_\rm{intercept}^{-1})

        &\beta_\gamma \sim \rm{Normal}(0, \sigma^2 \tau^{-1} \mathbb{1}_\gamma)

        &Y_n \sim \rm{Normal}(\beta_0 + X_{n, \gamma} \cdot \beta_\gamma, \sigma^2)
        \qquad \rm{for} \qquad n=1,2,...,N

    Note that the dimension of :math:`\beta_\gamma` depends on the number of covariates
    included in a particular model (i.e. on the number of non-zero entries in :math:`\gamma`).
    The hyperparameters :math:`\nu_0` and :math:`\lambda_0` govern the prior over
    :math:`\sigma^2`. The default choice :math:`\nu_0=\lambda_0=0` corresponds to an
    improper prior :math:`p(\sigma^2) \propto 1/\sigma^2`.

    For a gprior the prior over the coefficients (including the intercept :math:`\beta_0` if it is included)
    is instead specified as follows:

    .. math::

        \beta_{\gamma} \sim \rm{Normal}(0, c \sigma^2 (X_\gamma^{\rm{T}} X_\gamma)^{-1})

    where :math:`c > 0` is a user-specified hyperparameter.

    :param DataFrame dataframe: A `pandas.DataFrame` that contains covariates and responses. Each row
        encodes a single data point. All columns apart from the response column are assumed to be covariates.
    :param str response_column: The name of the column in `dataframe` that contains the continuous-valued responses.
    :param float S: The expected number of covariates to include in the model a priori. Defaults to 5.
    :param str prior: One of the two supported priors for the coefficients: 'isotropic' or 'gprior'.
        Defaults to 'isotropic'.
    :param bool include_intercept: Whether to include an intercept term. If included the intercept term is
       is included in all models so that the corresponding coefficient does not have a PIP. Defaults to True.
    :param float tau: Controls the precision of the coefficients in the isotropic prior. Defaults to 0.01.
    :param float tau_intercept: Controls the precision of the intercept in the isotropic prior. Defaults to 1.0e-4.
    :param float c: Controls the precision of the coefficients in the gprior. Defaults to 100.0.
    :param float nu0: Controls the prior over the variance in the Normal likelihood. Defaults to 0.0.
    :param float lambda0: Controls the prior over the variance in the Normal likelihood. Defaults to 0.0.
    :param str precision: Whether computations should be done with 'single' (i.e. 32-bit) or 'double' (i.e. 64-bit)
        floating point precision. Defaults to 'double'.
    :param str device: Whether computations should be done on CPU ('cpu') or GPU ('gpu'). Defaults to 'cpu'.
    :param float explore: This hyperparameter controls how greedy the MCMC algorithm is. Defaults to 5.0.
        For expert users only.
    :param bool precompute_XX: Whether the covariance matrix :math:`X^{\rm T} X \in \mathbb{R}^{P \times P}`
        should be pre-computed. Defaults to False. Note that setting this to True may result in out-of-memory errors
        for sufficiently large covariate matrices :math:`X`.
        However, if sufficient memory is available, setting precompute_XX to True should be faster.
    """
    def __init__(self, dataframe, response_column,
                 S=5, prior="isotropic",
                 include_intercept=True,
                 tau=0.01, tau_intercept=1.0e-4,
                 c=100.0,
                 nu0=0.0, lambda0=0.0,
                 precision="double", device="cpu",
                 explore=5, precompute_XX=False):

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
                                               include_intercept=include_intercept,
                                               verbose_constructor=False)

    def run(self, T=2000, T_burnin=1000, verbosity='bar', report_frequency=200, streaming=True, seed=None):
        """
        Run MCMC inference for :math:`T + T_{\rm burn-in}` iterations. After completion the results
        of the MCMC run can be accessed in the `summary` and `stats` attributes. Additionally,
        if `streaming == False` the `samples` attribute will contain raw samples from the MCMC algorithm.

        :param int T: Positive integer that controls the number of MCMC samples that are
            generated (i.e. after burn-in/adapation). Defaults to 2000.
        :param int T_burnin: Positive integer that controls the number of MCMC samples that are
            generated during burn-in/adapation. Defaults to 1000.
        :param str verbosity: Controls the verbosity of the `run` method. If 'stdout', progress is reported via stdout.
            If `bar`, then progress is reported via a progress bar. If `None`, then nothing is reported.
            Defaults to 'bar'.
        :param int report_frequency: Controls the frequency with which progress is reported if the `verbosity`
            argument is `stdout`. Defaults to 200.
        :param bool streaming: If True, MCMC samples are not stored in memory and summary statistics are computed
            online. Otherwise all `T` MCMC samples are stored in memory. Defaults to True. Only disable streaming if
            you wish to do something with the samples in the `samples` attribute (and have sufficient memory available).
        """
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

        if verbosity == 'bar':
            enumerate_samples = tenumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed),
                                           total=T + T_burnin)
        else:
            enumerate_samples = enumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed))

        for t, (burned, sample) in enumerate_samples:
            ts.append(time.time())
            if burned:
                container(namespace_to_numpy(sample))
            if verbosity == 'stdout' and (t % report_frequency == 0 or t == T + T_burnin - 1):
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
            self.beta_std = pd.Series(container.beta_std, index=self.X_columns + ["intercept"],
                                      name="Coefficient StdDev")
            self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns + ["intercept"],
                                              name="Conditional Coefficient")
            self.conditional_beta_std = pd.Series(container.conditional_beta_std, index=self.X_columns + ["intercept"],
                                                  name="Conditional Coefficient StdDev")
        else:
            self.beta = pd.Series(container.beta, index=self.X_columns, name="Coefficient")
            self.beta_std = pd.Series(container.beta_std, index=self.X_columns, name="Coefficient StdDev")
            self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns,
                                              name="Conditional Coefficient")
            self.conditional_beta_std = pd.Series(container.conditional_beta_std, index=self.X_columns,
                                                  name="Conditional Coefficient StdDev")

        self.summary = pd.concat([self.pip, self.beta, self.beta_std,
                                  self.conditional_beta, self.conditional_beta_std], axis=1)

        self.stats = {}
        populate_weight_stats(self.stats, self.weights)

        elapsed_time = time.time() - ts[0]
        self.stats['Elapsed MCMC time'] = "{:.1f} seconds".format(elapsed_time)
        self.stats['Mean iteration time'] = "{:.3f} ms".format(1000.0 * elapsed_time / (T + T_burnin))
        self.stats['Number of retained samples'] = T
        self.stats['Number of burn-in samples'] = T_burnin

        if verbosity == 'stdout':
            for k, v in self.stats.items():
                print('{}: '.format(k), v)


class BinomialLikelihoodVariableSelector(object):
    r"""
    Bayesian variable selection for a generalized linear model with a Binomial likelihood
    and a logistic link function. This class is appropriate for count-valued responses that are bounded.

    Usage::

        selector = BinomialLikelihoodVariableSelector(dataframe, 'response', 'total_count', ...)
        selector.run(T=2000, T_burnin=1000)
        print(selector.summary)

    The details of the model used in :class:`BinomialLikelihoodVariableSelector` are as follows.
    The covariates :math:`X`, responses :math:`Y`, and total counts :math:`T` are defined as:

    .. math::

        X \in \mathbb{R}^{N \times P} \qquad \qquad Y \in \mathbb{Z}_{\ge 0}^{N}
        \qquad \qquad T \in \mathbb{Z}_{\ge 1}^{N}

    and are provided by the user. The user should put some thought into whether :math:`X`
    should be centered and/or normalized.

    The inclusion of each covariate is governed by a Bernoulli random variable :math:`\gamma_p`.
    In particular :math:`\gamma_p = 0` corresponds to exclusion and :math:`\gamma_p = 1` corresponds to inclusion.
    The prior probability of inclusion is governed by :math:`h` or alternatively :math:`S`:

    .. math::

        h \in [0, 1] \qquad \rm{with} \qquad S \equiv hP

    The rest of the model is specified as:

    .. math::

        &\gamma_p \sim \rm{Bernoulli}(h) \qquad \rm{for} \qquad p=1,2,...,P

        &\beta_0 \sim \rm{Normal}(0, \tau_\rm{intercept}^{-1})

        &\beta_\gamma \sim \rm{Normal}(0, \tau^{-1} \mathbb{1}_\gamma)

        &Y_n \sim \rm{Binomial}(T_n, \sigma(\beta_0 + X_{n, \gamma} \cdot \beta_\gamma))
        \qquad \rm{for} \qquad n=1,2,...,N

    where :math:`\sigma(\cdot)` is the logistic or sigmoid function and :math:`T_n` denotes the
    :math:`N`-dimensional vector of total counts. That is each Binomial likelihood is equivalent
    to :math:`T_n` corresponding Bernoulli likelihoods.
    Note that the dimension of :math:`\beta_\gamma` depends on the number of covariates
    included in a particular model (i.e. on the number of non-zero entries in :math:`\gamma`).
    The intercept :math:`\beta_0` is always included in the model.

    :param DataFrame dataframe: A `pandas.DataFrame` that contains covariates and responses. Each row
        encodes a single data point. All columns apart from the response and total count column
        are assumed to be covariates.
    :param str response_column: The name of the column in `dataframe` that contains the count-valued responses.
    :param str total_count_column: The name of the column in `dataframe` that contains the total count
        for each data point.
    :param float S: The expected number of covariates to include in the model a priori. Defaults to 5.
    :param float tau: Controls the precision of the coefficients in the isotropic prior. Defaults to 0.01.
    :param float tau_intercept: Controls the precision of the intercept in the isotropic prior. Defaults to 1.0e-4.
    :param str precision: Whether computations should be done with 'single' (i.e. 32-bit) or 'double' (i.e. 64-bit)
        floating point precision. Defaults to 'double'.
    :param str device: Whether computations should be done on CPU ('cpu') or GPU ('gpu'). Defaults to 'cpu'.
    :param float explore: This hyperparameter controls how greedy the MCMC algorithm is. Defaults to 5.0.
        For expert users only.
    :param float xi_target: This hyperparameter controls how frequently the MCMC algorithm makes Polya-Gamma updates.
        Defaults to 0.25. For expert users only.
    """
    def __init__(self, dataframe, response_column, total_count_column,
                 S=5, tau=0.01, tau_intercept=1.0e-4,
                 precision="double", device="cpu",
                 explore=5, xi_target=0.25):

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
                                              xi_target=xi_target,
                                              verbose_constructor=False)

    def run(self, T=2000, T_burnin=1000, verbosity='bar', report_frequency=100, streaming=True, seed=None):
        """
        Run MCMC inference for :math:`T + T_{\rm burn-in}` iterations. After completion the results
        of the MCMC run can be accessed in the `summary` and `stats` attributes. Additionally,
        if `streaming == False` the `samples` attribute will contain raw samples from the MCMC algorithm.

        :param int T: Positive integer that controls the number of MCMC samples that are
            generated (i.e. after burn-in/adapation). Defaults to 2000.
        :param int T_burnin: Positive integer that controls the number of MCMC samples that are
            generated during burn-in/adapation. Defaults to 1000.
        :param str verbosity: Controls the verbosity of the `run` method. If 'stdout', progress is reported via stdout.
            If `bar`, then progress is reported via a progress bar. If `None`, then nothing is reported.
            Defaults to 'bar'.
        :param int report_frequency: Controls the frequency with which progress is reported if the `verbosity`
            argument is `stdout`. Defaults to 200.
        :param bool streaming: If True, MCMC samples are not stored in memory and summary statistics are computed
            online. Otherwise all `T` MCMC samples are stored in memory. Defaults to True. Only disable streaming if
            you wish to do something with the samples in the `samples` attribute (and have sufficient memory available).
        """
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

        if verbosity == 'bar':
            enumerate_samples = tenumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed),
                                           total=T + T_burnin)
        else:
            enumerate_samples = enumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed))

        for t, (burned, sample) in enumerate_samples:
            ts.append(time.time())
            if burned:
                container(namespace_to_numpy(sample))
            if verbosity == 'stdout' and (t % report_frequency == 0 or t == T + T_burnin - 1):
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
        self.beta_std = pd.Series(container.beta_std, index=self.X_columns + ['Intercept'], name="Coefficient StdDev")
        self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns + ['Intercept'],
                                          name="Conditional Coefficient")
        self.conditional_beta_std = pd.Series(container.conditional_beta_std, index=self.X_columns + ['Intercept'],
                                              name="Conditional Coefficient StdDev")

        self.summary = pd.concat([self.pip, self.beta, self.beta_std,
                                  self.conditional_beta, self.conditional_beta_std], axis=1)

        self.stats = {}
        populate_weight_stats(self.stats, self.weights)

        elapsed_time = time.time() - ts[0]
        self.stats['Elapsed MCMC time'] = "{:.1f} seconds".format(elapsed_time)
        self.stats['Mean iteration time'] = "{:.3f} ms".format(1000.0 * elapsed_time / (T + T_burnin))
        self.stats['Number of retained samples'] = T
        self.stats['Number of burn-in samples'] = T_burnin
        self.stats['Adapted xi value'] = "{:.3f}".format(self.sampler.xi.item())
        s = "Mean acc. prob.: {:.3f}  Accepted/Attempted: {}/{}"
        s = s.format(np.mean(self.sampler.acceptance_probs), self.sampler.accepted_omega_updates,
                     self.sampler.attempted_omega_updates)
        self.stats['Polya-Gamma MH stats'] = s

        if verbosity == 'stdout':
            for k, v in self.stats.items():
                print('{}: '.format(k), v)


class BernoulliLikelihoodVariableSelector(BinomialLikelihoodVariableSelector):
    r"""
    Bayesian variable selection for a generalized linear model with a Bernoulli likelihood
    and a logistic link function. This class is appropriate for binary-valued responses.

    Usage::

        selector = BernoulliLikelihoodVariableSelector(dataframe, 'response', ...)
        selector.run(T=2000, T_burnin=1000)
        print(selector.summary)

    The details of the model used in :class:`BernoulliLikelihoodVariableSelector` are as follows.
    The covariates :math:`X` and responses :math:`Y` are defined as follows:

    .. math::

        X \in \mathbb{R}^{N \times P} \qquad \qquad Y \in \{0, 1\}^{N}

    and are provided by the user. The user should put some thought into whether :math:`X`
    should be centered and/or normalized.

    The inclusion of each covariate is governed by a Bernoulli random variable :math:`\gamma_p`.
    In particular :math:`\gamma_p = 0` corresponds to exclusion and :math:`\gamma_p = 1` corresponds to inclusion.
    The prior probability of inclusion is governed by :math:`h` or alternatively :math:`S`:

    .. math::

        h \in [0, 1] \qquad \rm{with} \qquad S \equiv hP

    The rest of the model is specified as:

    .. math::

        &\gamma_p \sim \rm{Bernoulli}(h) \qquad \rm{for} \qquad p=1,2,...,P

        &\beta_0 \sim \rm{Normal}(0, \tau_\rm{intercept}^{-1})

        &\beta_\gamma \sim \rm{Normal}(0, \tau^{-1} \mathbb{1}_\gamma)

        &Y_n \sim \rm{Bernoulli}(\sigma(\beta_0 + X_{n, \gamma} \cdot \beta_\gamma))
        \qquad \rm{for} \qquad n=1,2,...,N

    where :math:`\sigma(\cdot)` is the logistic or sigmoid function.
    Note that the dimension of :math:`\beta_\gamma` depends on the number of covariates
    included in a particular model (i.e. on the number of non-zero entries in :math:`\gamma`).
    The intercept :math:`\beta_0` is always included in the model.

    :param DataFrame dataframe: A `pandas.DataFrame` that contains covariates and responses. Each row
        encodes a single data point. All columns apart from the response column are assumed to be covariates.
    :param str response_column: The name of the column in `dataframe` that contains the binary-valued responses.
    :param float S: The expected number of covariates to include in the model a priori. Defaults to 5.
    :param float tau: Controls the precision of the coefficients in the isotropic prior. Defaults to 0.01.
    :param float tau_intercept: Controls the precision of the intercept in the isotropic prior. Defaults to 1.0e-4.
    :param str precision: Whether computations should be done with 'single' (i.e. 32-bit) or 'double' (i.e. 64-bit)
        floating point precision. Defaults to 'double'.
    :param str device: Whether computations should be done on CPU ('cpu') or GPU ('gpu'). Defaults to 'cpu'.
    :param float explore: This hyperparameter controls how greedy the MCMC algorithm is. Defaults to 5.0.
        For expert users only.
    :param float xi_target: This hyperparameter controls how frequently the MCMC algorithm makes Polya-Gamma updates.
        Defaults to 0.25. For expert users only.
    """
    def __init__(self, dataframe, response_column,
                 S=5, tau=0.01, tau_intercept=1.0e-4,
                 precision="double", device="cpu",
                 explore=5, xi_target=0.25):

        dataframe['DummyTotalCount'] = 1.0
        super().__init__(dataframe, response_column, 'DummyTotalCount', S=S, explore=explore,
                         tau=tau, tau_intercept=tau_intercept, precision=precision,
                         device=device, xi_target=xi_target)


class NegativeBinomialLikelihoodVariableSelector(object):
    r"""
    Bayesian variable selection for a generalized linear model with a Negative Binomial likelihood and
    an exponential link function. This class is appropriate for count-valued responses that are unbounded.

    Usage::

        selector = NegativeBinomialLikelihoodVariableSelector(dataframe, 'response', 'psi0', ...)
        selector.run(T=2000, T_burnin=1000)
        print(selector.summary)

    The details of the model used in :class:`NegativeBinomialLikelihoodVariableSelector` are as follows.
    The covariates :math:`X`, responses :math:`Y`, and offsets :math:`\psi_0` are defined as follows

    .. math::

        X \in \mathbb{R}^{N \times P} \qquad \qquad Y \in \mathbb{Z}_{\ge 0}^{N}
        \qquad \qquad \psi_0 \in \mathbb{R}^{N}

    and are provided by the user. The user should put some thought into whether :math:`X`
    should be centered and/or normalized.

    The inclusion of each covariate is governed by a Bernoulli random variable :math:`\gamma_p`.
    In particular :math:`\gamma_p = 0` corresponds to exclusion and :math:`\gamma_p = 1` corresponds to inclusion.
    The prior probability of inclusion is governed by :math:`h` or alternatively :math:`S`:

    .. math::

        h \in [0, 1] \qquad \rm{with} \qquad S \equiv hP

    The full model specification for the Negative Binomial case is as follows:

    .. math::

        &\gamma_p \sim \rm{Bernoulli}(h) \qquad \rm{for} \qquad p=1,2,...,P

        &\beta_0 \sim \rm{Normal}(0, \tau_\rm{intercept}^{-1})

        &\beta_\gamma \sim \rm{Normal}(0, \tau^{-1} \mathbb{1}_\gamma)

        &\log \nu \sim \rm{ImproperPrior}(-\infty, \infty)

        &Y_n \sim \rm{NegBinomial}(\rm{mean}=\rm{exp}(\beta_0 + X_{n, \gamma} \cdot \beta_\gamma + \psi_{0, n}), \nu)
        \qquad \rm{for} \qquad n=1,2,...,N

    Here :math:`\nu` governs the dispersion or variance of the Negative Binomial likelihood.
    The vector :math:`\psi_0 \in \mathbb{R}^N` allows the user to supply a datapoint-specific offset.
    In many cases setting :math:`\psi_{0, n} = 0` is a reasonable choice.
    We note that we use a parameterization of the Negative Binomial distribution where the variance is given by

    .. math::

        \rm{variance} = \rm{mean} + \rm{mean}^2 / \nu

    so that small values of :math:`\nu` correspond to large dispersion/variance and :math:`\nu \to \infty` recovers
    the Poisson distribution.

    Note that above the dimension of :math:`\beta_\gamma` depends on the number of covariates
    included in a particular model (i.e. on the number of non-zero entries in :math:`\gamma`).
    The intercept :math:`\beta_0` is always included in the model.

    :param DataFrame dataframe: A `pandas.DataFrame` that contains covariates and responses. Each row
        encodes a single data point. All columns apart from the response and psi0 column are assumed to be covariates.
    :param str response_column: The name of the column in `dataframe` that contains the count-valued responses.
    :param str psi0_column: The name of the column in `dataframe` that contains the offset
        :math:`\psi_{0, n}` for each data point.
    :param float S: The expected number of covariates to include in the model a priori. Defaults to 5.
    :param float tau: Controls the precision of the coefficients in the isotropic prior. Defaults to 0.01.
    :param float tau_intercept: Controls the precision of the intercept in the isotropic prior. Defaults to 1.0e-4.
    :param str precision: Whether computations should be done with 'single' (i.e. 32-bit) or 'double' (i.e. 64-bit)
        floating point precision. Defaults to 'double'.
    :param str device: Whether computations should be done on CPU ('cpu') or GPU ('gpu'). Defaults to 'cpu'.
    :param float log_nu_rw_scale: This hyperparameter controls the proposal distribution for :math:`\log \nu` updates.
        Defaults to 0.05. For expert users only.
    :param float explore: This hyperparameter controls how greedy the MCMC algorithm is. Defaults to 5.0.
        For expert users only.
    :param float xi_target: This hyperparameter controls how frequently the MCMC algorithm makes Polya-Gamma updates.
        Defaults to 0.25. For expert users only.
    :param float init_nu: This hyperparameter controls the initial value of the dispersion parameter `nu`.
        Defaults to 5.0. For expert users only.
    """
    def __init__(self, dataframe, response_column, psi0_column,
                 S=5, tau=0.01, tau_intercept=1.0e-4,
                 precision="double", device="cpu",
                 log_nu_rw_scale=0.05, explore=5.0,
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
                                              xi_target=xi_target, init_nu=init_nu,
                                              verbose_constructor=False)

    def run(self, T=2000, T_burnin=1000, verbosity='bar', report_frequency=100, streaming=True, seed=None):
        """
        Run MCMC inference for :math:`T + T_{\rm burn-in}` iterations. After completion the results
        of the MCMC run can be accessed in the `summary` and `stats` attributes. Additionally,
        if `streaming == False` the `samples` attribute will contain raw samples from the MCMC algorithm.

        :param int T: Positive integer that controls the number of MCMC samples that are
            generated (i.e. after burn-in/adapation). Defaults to 2000.
        :param int T_burnin: Positive integer that controls the number of MCMC samples that are
            generated during burn-in/adapation. Defaults to 1000.
        :param str verbosity: Controls the verbosity of the `run` method. If 'stdout', progress is reported via stdout.
            If `bar`, then progress is reported via a progress bar. If `None`, then nothing is reported.
            Defaults to 'bar'.
        :param int report_frequency: Controls the frequency with which progress is reported if the `verbosity`
            argument is `stdout`. Defaults to 200.
        :param bool streaming: If True, MCMC samples are not stored in memory and summary statistics are computed
            online. Otherwise all `T` MCMC samples are stored in memory. Defaults to True. Only disable streaming if
            you wish to do something with the samples in the `samples` attribute (and have sufficient memory available).
        """
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

        if verbosity == 'bar':
            enumerate_samples = tenumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed),
                                           total=T + T_burnin)
        else:
            enumerate_samples = enumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed))

        for t, (burned, sample) in enumerate_samples:
            ts.append(time.time())
            if burned:
                container(namespace_to_numpy(sample))
            if verbosity == 'stdout' and (t % report_frequency == 0 or t == T + T_burnin - 1):
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
        self.beta_std = pd.Series(container.beta_std, index=self.X_columns + ['Intercept'], name="Coefficient StdDev")
        self.conditional_beta = pd.Series(container.conditional_beta, index=self.X_columns + ['Intercept'],
                                          name="Conditional Coefficient")
        self.conditional_beta_std = pd.Series(container.conditional_beta_std, index=self.X_columns + ['Intercept'],
                                              name="Conditional Coefficienti StdDev")

        self.summary = pd.concat([self.pip, self.beta, self.beta_std,
                                  self.conditional_beta, self.conditional_beta_std], axis=1)

        self.stats = {}
        populate_weight_stats(self.stats, self.weights)

        self.stats['nu posterior'] = '{:.3f} +- {:.3f}'.format(container.nu, container.nu_std)
        self.stats['log(nu) posterior'] = '{:.3f} +- {:.3f}'.format(container.log_nu, container.log_nu_std)

        elapsed_time = time.time() - ts[0]
        self.stats['Elapsed MCMC time'] = "{:.1f} seconds".format(elapsed_time)
        self.stats['Mean iteration time'] = "{:.3f} ms".format(1000.0 * elapsed_time / (T + T_burnin))
        self.stats['Number of retained samples'] = T
        self.stats['Number of burn-in samples'] = T_burnin
        self.stats['Adapted xi value'] = "{:.3f}".format(self.sampler.xi.item())
        s = "Mean acc. prob.: {:.3f}  Accepted/Attempted: {}/{}"
        s = s.format(np.mean(self.sampler.acceptance_probs), self.sampler.accepted_omega_updates,
                     self.sampler.attempted_omega_updates)
        self.stats['Polya-Gamma MH stats'] = s

        if verbosity == 'stdout':
            for k, v in self.stats.items():
                print('{}: '.format(k), v)
