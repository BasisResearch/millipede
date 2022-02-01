import time
import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import MegaNormalLikelihoodSampler
from millipede.util import namespace_to_numpy, stack_namespaces


def test_linear_correlated(prior='isotropic', precompute_XX=False,
                           include_intercept=True, N=128, P=256, subset_size=8, intercept=2.34,
                           T=250 * 1000, T_burnin=1000, report_frequency=300, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = Z + 0.05 * torch.randn(N).double()

    if include_intercept:
        Y += intercept

    samples = []
    sampler = MegaNormalLikelihoodSampler(X, Y, precompute_XX=precompute_XX, prior=prior,
                                          subset_size=subset_size,
                                          compute_betas=True, S=1.0, nu0=0.0, lambda0=0.0,
                                          tau=0.01, c=100.0, include_intercept=include_intercept,
                                          tau_intercept=1.0e-4)

    t0 = time.time()
    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))
    print("elapsed: ", time.time() - t0)

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.add_prob.T, weights)
    print("pip [subset size: {}]\n".format(subset_size), pip[:5], pip[2:].sum().item())

    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)


test_linear_correlated(subset_size=128)
test_linear_correlated(subset_size=64)
test_linear_correlated(subset_size=255)
