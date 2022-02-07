import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import (
    BernoulliLikelihoodVariableSelector,
    BinomialLikelihoodVariableSelector,
    CountLikelihoodSampler,
    NegativeBinomialLikelihoodVariableSelector,
)
from millipede.util import namespace_to_numpy, stack_namespaces


@pytest.mark.parametrize("variable_S", [False, True])
def test_binomial(variable_S, streaming=False, N=256, P=16, T=2000, T_burnin=200, intercept=0.17, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.005 * torch.randn(N, 2).double()
    TC = 7.0 * torch.ones(N).double()
    Y = torch.distributions.Binomial(total_count=TC,
                                     logits=Z + intercept + 0.01 * torch.randn(N)).sample().double()

    S = (1.0, P - 1.0) if variable_S else 1.0

    samples = []
    sampler = CountLikelihoodSampler(X, Y, TC=TC, S=S, tau=0.01, tau_intercept=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.20)
    assert_close(beta[2:P], np.zeros(P - 2), atol=0.05)
    assert_close(beta[-1].item(), intercept, atol=0.05)

    # test selector
    XYTC = torch.cat([X, Y.unsqueeze(-1), TC.unsqueeze(-1)], axis=-1)
    columns = ['feat{}'.format(c) for c in range(P)] + ['response', 'total_count']
    dataframe = pandas.DataFrame(XYTC.data.numpy(), columns=columns)

    selector = BinomialLikelihoodVariableSelector(dataframe, 'response', 'total_count',
                                                  S=S, tau=0.01, tau_intercept=1.0e-4,
                                                  precision='double', device='cpu')
    selector.run(T=T, T_burnin=T_burnin, report_frequency=1100, streaming=streaming, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-10)
    assert_close(selector.beta.values, beta, atol=1.0e-10)

    print("[selector.stats]\n", selector.stats)


def test_bernoulli(streaming=True, N=256, P=16, T=2000, T_burnin=200, intercept=0.17, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.005 * torch.randn(N, 2).double()
    Y = torch.distributions.Bernoulli(logits=Z + intercept + 0.01 * torch.randn(N)).sample().double()
    TC = torch.ones(N).double()

    samples = []
    sampler = CountLikelihoodSampler(X, Y, TC=TC, S=1.0, tau=0.01, tau_intercept=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.20)
    assert_close(beta[2:P], np.zeros(P - 2), atol=0.05)
    assert_close(beta[-1].item(), intercept, atol=0.05)

    # test selector
    XY = torch.cat([X, Y.unsqueeze(-1)], axis=-1)
    columns = ['feat{}'.format(c) for c in range(P)] + ['response']
    dataframe = pandas.DataFrame(XY.data.numpy(), columns=columns)

    selector = BernoulliLikelihoodVariableSelector(dataframe, 'response',
                                                   S=1.0, tau=0.01, tau_intercept=1.0e-4,
                                                   precision='double', device='cpu')
    selector.run(T=T, T_burnin=T_burnin, report_frequency=1100, streaming=streaming, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-10)
    assert_close(selector.beta.values, beta, atol=1.0e-10)


@pytest.mark.parametrize("streaming", [False, True])
def test_negative_binomial(streaming, N=150, P=16, T=2000, T_burnin=500, psi0=0.37, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    noise = torch.exp(0.1 * torch.randn(N))
    Y = torch.distributions.Poisson(Z.exp() * noise).sample()

    samples = []
    sampler = CountLikelihoodSampler(X, Y, psi0=psi0, TC=None, S=1.0, tau=0.01, tau_intercept=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    nu = np.exp(np.dot(samples.log_nu, weights))
    assert nu > 2.0 and nu < 20.0

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.1)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(beta[2:P], np.zeros(P - 2), atol=0.1)
    assert_close(beta[-1].item(), -psi0, atol=0.1)

    # test selector
    XYpsi0 = torch.cat([X, Y.unsqueeze(-1), X.new_ones(N, 1) * psi0], axis=-1)
    columns = ['feat{}'.format(c) for c in range(P)] + ['response', 'psi0']
    dataframe = pandas.DataFrame(XYpsi0.data.numpy(), columns=columns)

    selector = NegativeBinomialLikelihoodVariableSelector(dataframe, 'response', 'psi0',
                                                          S=1.0, tau=0.01, tau_intercept=1.0e-4,
                                                          precision='double', device='cpu')
    selector.run(T=T, T_burnin=T_burnin, report_frequency=1250, streaming=streaming, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-10)
    assert_close(selector.beta.values, beta, atol=1.0e-10)

    if streaming:
        print(selector.summary)
