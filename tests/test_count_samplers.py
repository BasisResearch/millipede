import numpy as np
import torch
from common import assert_close

from millipede import CountLikelihoodSampler
from millipede.util import namespace_to_numpy, stack_namespaces


def test_binomial(N=128, P=16, T=1000, T_burnin=500):
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = (Z + 0.05 * torch.randn(N)) > 0.0
    Y = Y.double()
    TC = torch.ones(N).double()

    samples = []
    sampler = CountLikelihoodSampler(X, Y, TC=TC, S=1.0, tau=0.01)

    for t, (burned, s) in enumerate(sampler.gibbs_chain(T=T, T_burnin=T_burnin)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.1)


def test_negative_binomial(N=128, P=16, T=2000, T_burnin=500):
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = torch.distributions.Poisson(Z.exp() + 0.1 * torch.rand(N)).sample()

    samples = []
    sampler = CountLikelihoodSampler(X, Y, psi0=0.0, TC=None, S=1.0, tau=0.01)

    for t, (burned, s) in enumerate(sampler.gibbs_chain(T=T, T_burnin=T_burnin)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    nu = np.exp(np.dot(samples.log_nu, weights))
    assert nu > 2.0 and nu < 10.0

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.1)
