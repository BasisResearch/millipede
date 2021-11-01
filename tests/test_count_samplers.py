import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import CountLikelihoodSampler
from millipede.util import namespace_to_numpy, stack_namespaces


def test_linear_correlated(N=128, P=16, bias=2.34,
                           T=3, T_burnin=2, report_frequency=2):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(1)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = (Z + 0.05 * torch.randn(N)) > 0.0
    Y = Y.double()
    Tc = torch.ones(N).double()

    samples = []
    sampler = CountLikelihoodSampler(X, Y, T=Tc, S=1.0, tau=0.01)

    for t, (burned, s) in enumerate(sampler.gibbs_chain(T=T, T_burnin=T_burnin)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.add_prob.T, weights)
    print("PIP", pip)
