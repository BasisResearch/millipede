import numpy as np
import pytest
import torch
from common import assert_close

from millipede.normal import NormalLikelihoodSampler
from millipede.util import namespace_to_numpy, stack_namespaces


@pytest.mark.parametrize("precompute_XX", [False, True])
@pytest.mark.parametrize("prior", ["isotropic", "gprior"])
def test_linear_correlated(prior, precompute_XX, N=256, P=16):
    X = torch.randn(N, P).double()
    X[:, 1] = X[:, 0] + 0.001 * torch.randn(N).double()
    Y = X[:, 0] + 0.05 * torch.randn(N).double()

    samples = []
    sampler = NormalLikelihoodSampler(X, Y, precompute_XX=precompute_XX, prior=prior,
                                      compute_betas=True, S=1.0, nu0=0.0, lambda0=0.0,
                                      tau=0.01, c=50.0)

    for t, (burned, s) in enumerate(sampler.gibbs_chain(T=2000, T_burnin=200)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = (samples.add_prob * weights[:, None]).sum(0)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.05)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(beta[2:], np.zeros(P - 2), atol=0.02)
