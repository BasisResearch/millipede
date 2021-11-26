import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import MegaSampler
from millipede.util import namespace_to_numpy, stack_namespaces


def test_linear_correlated(precompute_XX=False, include_intercept=False,
                           N=128, P=16, intercept=2.34, report_frequency=200,
                           T=1000, T_burnin=200, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = Z + 0.05 * torch.randn(N).double()

    if include_intercept:
        Y += intercept

    samples = []
    sampler = MegaSampler(X, Y, precompute_XX=precompute_XX,
                          compute_betas=True, S=1.0, nu0=0.0, lambda0=0.0,
                          tau=0.01, c=100.0, include_intercept=include_intercept,
                          tau_intercept=1.0e-4)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)

    pip = np.mean(samples.add_prob, axis=0)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.mean(samples.beta, axis=0)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)

    if include_intercept:
        assert_close(beta[-1].item(), intercept, atol=0.1)
    assert_close(beta[2:P], np.zeros(P - 2), atol=0.02)
