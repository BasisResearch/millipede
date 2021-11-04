import numpy as np
import pandas
import torch
from common import assert_close

from millipede import BinomialLikelihoodVariableSelector, CountLikelihoodSampler
from millipede.util import namespace_to_numpy, stack_namespaces


def test_binomial(N=256, P=16, T=2200, T_burnin=300, bias=0.17):
    torch.manual_seed(1)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.005 * torch.randn(N, 2).double()
    Y = torch.distributions.Bernoulli(logits=Z + bias + 0.01 * torch.randn(N)).sample().double()
    TC = torch.ones(N).double()

    samples = []
    sampler = CountLikelihoodSampler(X, Y, TC=TC, S=1.0, tau=0.01, tau_bias=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(beta[2:P], np.zeros(P - 2), atol=0.05)
    assert_close(beta[-1].item(), bias, atol=0.1)

    # test selector
    XYTC = torch.cat([X, Y.unsqueeze(-1), TC.unsqueeze(-1)], axis=-1)
    columns = ['feat{}'.format(c) for c in range(P)] + ['response', 'total_count']
    dataframe = pandas.DataFrame(XYTC.data.numpy(), columns=columns)

    selector = BinomialLikelihoodVariableSelector(dataframe, 'response', 'total_count',
                                                  S=1.0, tau=0.01, precision='double', device='cpu')
    selector.run(T=T, T_burnin=T_burnin, report_frequency=500)

    assert_close(selector.pip.values, pip, atol=0.2)
    assert_close(selector.beta.values, beta, atol=0.2)


def test_negative_binomial(N=128, P=16, T=2000, T_burnin=500):
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = torch.distributions.Poisson(Z.exp() + 0.1 * torch.rand(N)).sample()

    samples = []
    sampler = CountLikelihoodSampler(X, Y, psi0=0.0, TC=None, S=1.0, tau=0.01, tau_bias=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    nu = np.exp(np.dot(samples.log_nu, weights))
    assert nu > 2.0 and nu < 10.0

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.1)
