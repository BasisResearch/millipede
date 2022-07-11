import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import ASISampler, ASIVariableSelector, NormalLikelihoodSampler
from millipede.util import namespace_to_numpy, stack_namespaces


@pytest.mark.parametrize("include_intercept", [True, False])
def test_linear_correlated(include_intercept, N=128, P=16, intercept=2.34, T=70 * 1000, T_burnin=6000, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = Z + 0.05 * torch.randn(N).double()

    if include_intercept:
        Y += intercept

    S = 1.0

    samples = []
    sampler = NormalLikelihoodSampler(X, Y,
                                      precompute_XX=False, prior='isotropic',
                                      compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                                      tau=0.01, c=0.0, include_intercept=include_intercept,
                                      tau_intercept=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    wtgs_pip = np.dot(samples.pip.T, weights)
    assert_close(wtgs_pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(wtgs_pip[2:], np.zeros(P - 2), atol=0.15)

    wtgs_beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(wtgs_beta[:2], np.array([0.5, 0.5]), atol=0.15)

    if include_intercept:
        assert_close(wtgs_beta[-1].item(), intercept, atol=0.05)

    assert_close(wtgs_beta[2:P], np.zeros(P - 2), atol=0.02)

    ##########
    #  ASI  #
    ##########

    samples = []
    sampler = ASISampler(X, Y,
                         precompute_XX=False, prior='isotropic',
                         compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                         tau=0.01, include_intercept=include_intercept,
                         tau_intercept=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    pip = samples.pip.mean(0)
    beta = samples.beta.mean(0)

    assert_close(wtgs_pip[:2], pip[:2], rtol=0.07)
    assert_close(wtgs_pip[2:], pip[2:], rtol=0.1)

    assert_close(wtgs_beta[:2], beta[:2], rtol=0.07)
    assert_close(np.zeros(P - 2), beta[2:P], atol=1.0e-4)

    if include_intercept:
        assert_close(beta[-1].item(), intercept, atol=0.05)

    columns = ['feat{}'.format(c) for c in range(P)] + ['response']
    XY = torch.cat([X, Y.unsqueeze(-1)], axis=-1)
    dataframe = pandas.DataFrame(XY.data.numpy(), columns=columns)

    selector = ASIVariableSelector(dataframe, 'response',
                                   tau=0.01,
                                   precompute_XX=False,
                                   include_intercept=include_intercept, prior='isotropic',
                                   S=S, nu0=0.0, lambda0=0.0, precision='double',
                                   device='cpu')

    selector.run(T=T, T_burnin=T_burnin, report_frequency=0, streaming=True, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-10)
    assert_close(selector.beta.values, beta, atol=1.0e-10)
