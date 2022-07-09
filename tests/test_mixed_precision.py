import numpy as np
import pandas
import torch
from common import assert_close

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


def test_mixed_precision(include_intercept=False, N=128, P=16, T=3 * 1000, T_burnin=500, S=1.0, seed=1):
    torch.manual_seed(seed)
    X = torch.bernoulli(0.1 * torch.ones(N, P)).half()
    Y = X[:, 0].double() - 0.5 * X[:, 1].double() + 0.05 * torch.randn(N).double()

    samples = []
    sampler = NormalLikelihoodSampler(X.double(), Y,
                                      precompute_XX=False, prior='isotropic',
                                      compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                                      tau=0.01, c=100.0, include_intercept=include_intercept,
                                      tau_intercept=1.0e-4, mixed_precision=False)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.pip.T, weights)
    print("pip[:4]", pip[:4])
    assert_close(pip[:2], np.array([1.0, 1.0]), atol=0.05)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.03)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([1.0, -0.5]), atol=0.05)
    assert_close(beta[2:P], np.zeros(P - 2), atol=0.03)

    ###################
    # mixed precision #
    ###################

    samples = []
    sampler = NormalLikelihoodSampler(X, Y,
                                      precompute_XX=False, prior='isotropic',
                                      compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                                      tau=0.01, c=100.0, include_intercept=include_intercept,
                                      tau_intercept=1.0e-4, mixed_precision=True)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip_mixed = np.dot(samples.pip.T, weights)
    print("pip_mixed[:4]", pip_mixed[:4])
    assert_close(pip, pip_mixed, atol=1.0e-10)

    beta_mixed = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta, beta_mixed, atol=1.0e-10)

    ###################
    # mixed precision #
    ###################

    dataframe = pandas.DataFrame(X.data.numpy(), columns=['feat{}'.format(c) for c in range(P)])
    dataframe['response'] = Y.data.numpy()

    selector = NormalLikelihoodVariableSelector(dataframe, 'response',
                                                tau=0.01, c=100.0,
                                                precompute_XX=False,
                                                include_intercept=include_intercept, prior='isotropic',
                                                S=S, nu0=0.0, lambda0=0.0, precision='mixeddouble')

    selector.run(T=T, T_burnin=T_burnin, streaming=True, seed=seed)

    assert_close(selector.pip.values, pip_mixed, atol=1.0e-10)
    assert_close(selector.beta.values, beta_mixed, atol=1.0e-10)

    dataframe = pandas.DataFrame(X.byte().data.numpy(), columns=['feat{}'.format(c) for c in range(P)])
    dataframe['response'] = Y.data.numpy()

    selector = NormalLikelihoodVariableSelector(dataframe, 'response',
                                                tau=0.01, c=100.0,
                                                precompute_XX=False,
                                                include_intercept=include_intercept, prior='isotropic',
                                                S=S, nu0=0.0, lambda0=0.0, precision='mixeddouble')

    selector.run(T=T, T_burnin=T_burnin, streaming=True, seed=seed)

    assert_close(selector.pip.values, pip_mixed, atol=1.0e-10)
    assert_close(selector.beta.values, beta_mixed, atol=1.0e-10)
