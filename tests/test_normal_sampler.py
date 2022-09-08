import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


@pytest.mark.parametrize("precompute_XX", [False, True])
@pytest.mark.parametrize("prior", ["gprior", "isotropic"])
@pytest.mark.parametrize("include_intercept", [True, False])
@pytest.mark.parametrize("variable_S_X_assumed", [(False, False), (True, True)])
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_linear_correlated(device, prior, precompute_XX, include_intercept, variable_S_X_assumed,
                           N=128, P=16, intercept=2.34, T=4500, T_burnin=200, report_frequency=1100, seed=1):
    if device == "gpu" and not torch.cuda.is_available():
        return

    variable_S, X_assumed = variable_S_X_assumed

    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Z = torch.randn(N).double()
    X[:, 0:2] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
    Y = Z + 0.05 * torch.randn(N).double()

    if include_intercept:
        Y += intercept

    X_assumed = torch.randn(N, 2).double() if X_assumed else None
    if X_assumed is not None:
        Y += 0.5 * X_assumed[:, -1]

    S = 1.0 if not variable_S else (0.25, 0.25 * P - 0.25)
    subset_size = 12 if precompute_XX else None

    if prior == 'isotropic':
        sigma_scale_factor = (torch.ones(N) + 0.1 * torch.rand(N)).double()
    else:
        sigma_scale_factor = None

    samples = []
    if device == "cpu":
        sampler = NormalLikelihoodSampler(X, Y, X_assumed=X_assumed, sigma_scale_factor=sigma_scale_factor,
                                          precompute_XX=precompute_XX, prior=prior,
                                          compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                                          tau=0.01, c=100.0, include_intercept=include_intercept,
                                          tau_intercept=1.0e-4, subset_size=subset_size)
    elif device == "gpu":
        sampler = NormalLikelihoodSampler(X.cuda(), Y.cuda(),
                                          X_assumed=X_assumed.cuda() if X_assumed is not None else None,
                                          sigma_scale_factor=sigma_scale_factor.cuda() if
                                              sigma_scale_factor is not None else None,
                                          precompute_XX=precompute_XX, prior=prior,
                                          compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                                          tau=0.01, c=100.0, include_intercept=include_intercept,
                                          tau_intercept=1.0e-4, subset_size=subset_size)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    pip = np.dot(samples.pip.T, weights)
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)

    if include_intercept:
        assert_close(beta[-1].item(), intercept, atol=0.1)

    if X_assumed is None:
        assert_close(beta[2:P], np.zeros(P - 2), atol=0.02)
    else:
        assert_close(beta[2:P + 2], np.concatenate([np.zeros(P - 1), np.array([0.5])]), atol=0.02)

    columns = ['feat{}'.format(c) for c in range(P)] + ['response']
    XY = torch.cat([X, Y.unsqueeze(-1)], axis=-1)
    if X_assumed is None:
        assumed_columns = []
    else:
        XY = torch.cat([XY, X_assumed], axis=-1)
        assumed_columns = ['afeat0', 'afeat1']
        columns += assumed_columns

    dataframe = pandas.DataFrame(XY.data.numpy(), columns=columns)
    if sigma_scale_factor is not None:
        sigma_scale_factor_column = 'sigma_scale_factor'
        dataframe[sigma_scale_factor_column] = sigma_scale_factor
    else:
        sigma_scale_factor_column = None

    selector = NormalLikelihoodVariableSelector(dataframe, 'response', assumed_columns=assumed_columns,
                                                sigma_scale_factor_column=sigma_scale_factor_column,
                                                tau=0.01, c=100.0,
                                                precompute_XX=precompute_XX,
                                                include_intercept=include_intercept, prior=prior,
                                                S=S, nu0=0.0, lambda0=0.0, precision='double',
                                                device=device, subset_size=subset_size)

    selector.run(T=T, T_burnin=T_burnin, report_frequency=report_frequency, streaming=precompute_XX, seed=seed)

    assert_close(selector.pip.values, pip, atol=1.0e-9)
    assert_close(selector.beta.values, beta, atol=1.0e-9)

    assert_close(selector.pip.values[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(selector.pip.values[2:], np.zeros(P - 2), atol=0.1)

    assert_close(selector.beta.values[:2], np.array([0.5, 0.5]), atol=0.2)
    if include_intercept:
        assert_close(selector.beta.values[-1].item(), intercept, atol=0.1)
    if X_assumed is None:
        assert_close(selector.beta.values[2:P], np.zeros(P - 2), atol=0.02)
    else:
        assert_close(selector.beta.values[2:P + 2], np.concatenate([np.zeros(P - 1), np.array([0.5])]), atol=0.02)

    assert_close(selector.conditional_beta.values[:2], np.array([1.0, 1.0]), atol=0.25)
    if include_intercept:
        assert_close(selector.conditional_beta.values[-1].item(), intercept, atol=0.1)
        assert_close(selector.conditional_beta.values[-1].item(),
                     selector.beta.values[-1].item(), atol=1.0e-6)
    assert_close(selector.conditional_beta.values[2:P], np.zeros(P - 2), atol=0.30)

    print("[selector.stats]\n", selector.stats)
