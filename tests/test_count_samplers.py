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


@pytest.mark.parametrize("subset_size", [12, None])
@pytest.mark.parametrize("variable_S", [False, True])
def test_binomial(subset_size, variable_S, streaming=False, N=512, P=16, T=2000, T_burnin=200, intercept=0.17, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    X_assumed = torch.randn(N, 2).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.005 * torch.randn(N, 2).double()
    logits = Z + 0.33 * X_assumed[:, -1] + intercept + 0.01 * torch.randn(N)
    TC = 7.0 * torch.ones(N).double()
    Y = torch.distributions.Binomial(total_count=TC, logits=logits).sample().double()

    S = (1.0, P - 1.0) if variable_S else 1.0

    samples = []
    sampler = CountLikelihoodSampler(X, Y, X_assumed=X_assumed, TC=TC, S=S, tau=0.01,
                                     tau_intercept=1.0e-4, subset_size=subset_size)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.pip.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.20)
    assert_close(beta[2:P + 1], np.zeros(P - 1), atol=0.05)
    assert_close(beta[P + 1].item(), 0.33, atol=0.05)
    assert_close(beta[-1].item(), intercept, atol=0.05)

    # test selector
    XYTC = torch.cat([X, Y.unsqueeze(-1), TC.unsqueeze(-1), X_assumed], axis=-1)
    assumed_columns = ['afeat0', 'afeat1']
    columns = ['feat{}'.format(c) for c in range(P)] + ['response', 'total_count'] + assumed_columns
    dataframe = pandas.DataFrame(XYTC.data.numpy(), columns=columns)

    selector = BinomialLikelihoodVariableSelector(dataframe, 'response', 'total_count',
                                                  assumed_columns=assumed_columns,
                                                  S=S, tau=0.01, tau_intercept=1.0e-4,
                                                  precision='double', device='cpu',
                                                  subset_size=subset_size)
    selector.run(T=T, T_burnin=T_burnin, report_frequency=1100, streaming=streaming, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-10)
    assert_close(selector.beta.values, beta, atol=1.0e-10)

    print("[selector.stats]\n", selector.stats)


@pytest.mark.parametrize("subset_size", [12, None])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_bernoulli(subset_size, device, streaming=True, N=256, P=16, T=3000, T_burnin=200, intercept=0.17, seed=1):
    if device == "gpu" and not torch.cuda.is_available():
        return

    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.005 * torch.randn(N, 2).double()
    Y = torch.distributions.Bernoulli(logits=Z + intercept + 0.01 * torch.randn(N)).sample().double()
    TC = torch.ones(N).double()

    samples = []
    if device == 'cpu':
        sampler = CountLikelihoodSampler(X, Y, TC=TC, S=1.0, tau=0.01, tau_intercept=1.0e-4, subset_size=subset_size)
    elif device == 'gpu':
        sampler = CountLikelihoodSampler(X.cuda(), Y.cuda(), TC=TC.cuda(), S=1.0,
                                         tau=0.01, tau_intercept=1.0e-4, subset_size=subset_size)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.pip.T, weights)
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
                                                   precision='double', device=device, subset_size=subset_size)
    selector.run(T=T, T_burnin=T_burnin, report_frequency=1100, streaming=streaming, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-10)
    assert_close(selector.beta.values, beta, atol=1.0e-10)


@pytest.mark.parametrize("subset_size", [12, None])
@pytest.mark.parametrize("noisy", [False, True])
def test_negative_binomial(subset_size, noisy, N=256, P=16, T=4000, T_burnin=500, psi0=0.37, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    X_assumed = torch.randn(N, 2).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()

    if noisy:
        log_rate = Z + 0.33 * X_assumed[:, -1] + 0.50 * torch.randn(N)
    else:
        log_rate = Z + 0.33 * X_assumed[:, -1] + 0.05 * torch.randn(N)

    Y = torch.distributions.Poisson(log_rate.exp()).sample()

    streaming = noisy

    samples = []
    sampler = CountLikelihoodSampler(X, Y, X_assumed=X_assumed, psi0=psi0, subset_size=subset_size,
                                     TC=None, S=1.0, tau=0.01, tau_intercept=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    if noisy:
        nu = np.exp(np.dot(samples.log_nu, weights))
        assert nu > 1.0 and nu < 6.0

    pip = np.dot(samples.pip.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.1)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(beta[2:P + 1], np.zeros(P - 1), atol=0.1)
    assert_close(beta[P + 1].item(), 0.33, atol=0.1 if not noisy else 0.2)
    assert_close(beta[-1].item(), -psi0, atol=0.1 if not noisy else 0.2)

    # test selector
    XYpsi0 = torch.cat([X, Y.unsqueeze(-1), X.new_ones(N, 1) * psi0, X_assumed], axis=-1)
    assumed_columns = ['afeat0', 'afeat1']
    columns = ['feat{}'.format(c) for c in range(P)] + ['response', 'psi0'] + assumed_columns
    dataframe = pandas.DataFrame(XYpsi0.data.numpy(), columns=columns)

    selector = NegativeBinomialLikelihoodVariableSelector(dataframe, 'response', 'psi0',
                                                          assumed_columns=assumed_columns,
                                                          S=1.0, tau=0.01, tau_intercept=1.0e-4,
                                                          precision='double', device='cpu',
                                                          subset_size=subset_size)
    selector.run(T=T, T_burnin=T_burnin, report_frequency=1250, streaming=streaming, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-10)
    assert_close(selector.beta.values, beta, atol=1.0e-10)

    if streaming:
        print(selector.summary)
