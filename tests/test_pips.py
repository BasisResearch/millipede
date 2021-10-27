import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


@pytest.mark.parametrize("precompute_XX", [False, True])
@pytest.mark.parametrize("prior", ["isotropic", "gprior"])
@pytest.mark.parametrize("include_bias", [True, False])
def test_linear_correlated(prior, precompute_XX, include_bias, N=128, P=16, bias=2.34,
                           T=3000, T_burnin=200, report_frequency=1600):
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = Z + 0.05 * torch.randn(N).double()

    if include_bias:
        Y += bias

    samples = []
    sampler = NormalLikelihoodSampler(X, Y, precompute_XX=precompute_XX, prior=prior,
                                      compute_betas=True, S=1.0, nu0=0.0, lambda0=0.0,
                                      tau=0.01, c=50.0, include_bias=include_bias)

    for t, (burned, s) in enumerate(sampler.gibbs_chain(T=T, T_burnin=T_burnin)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.add_prob.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)

    if include_bias:
        assert_close(beta[-1].item(), bias, atol=0.1)
    assert_close(beta[2:P], np.zeros(P - 2), atol=0.02)

    XY = torch.cat([X, Y.unsqueeze(-1)], axis=-1)
    columns = ['feat{}'.format(c) for c in range(P)] + ['response']
    dataframe = pandas.DataFrame(XY.data.numpy(), columns=columns)
    selector = NormalLikelihoodVariableSelector(dataframe, 'response', tau=0.01, c=50.0, include_bias=include_bias,
                                                prior=prior, S=1.0, nu0=0.0, lambda0=0.0, precision='single')

    selector.run(T=T, T_burnin=T_burnin, report_frequency=report_frequency)

    assert_close(selector.pip.values, pip, atol=0.15)
    assert_close(selector.beta.values, beta, atol=0.15)

    assert_close(selector.pip.values[:2], np.array([0.5, 0.5]), atol=0.15)
    assert_close(selector.pip.values[2:], np.zeros(P - 2), atol=0.1)

    assert_close(selector.beta.values[:2], np.array([0.5, 0.5]), atol=0.2)
    if include_bias:
        assert_close(selector.beta.values[-1].item(), bias, atol=0.1)
    assert_close(selector.beta.values[2:P], np.zeros(P - 2), atol=0.05)

    assert_close(selector.conditional_beta.values[:2], np.array([1.0, 1.0]), atol=0.25)
    if include_bias:
        assert_close(selector.conditional_beta.values[-1].item(), bias, atol=0.1)
        assert_close(selector.conditional_beta.values[-1].item(), selector.beta.values[-1].item(), atol=1.0e-6)
    assert_close(selector.conditional_beta.values[2:P], np.zeros(P - 2), atol=0.15)
