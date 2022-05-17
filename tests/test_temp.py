import numpy as np
import pandas
import pytest
import torch
from common import assert_close

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


@pytest.mark.parametrize("precompute_XX", [False])
@pytest.mark.parametrize("prior", ["isotropic"])
@pytest.mark.parametrize("include_intercept", [False])
@pytest.mark.parametrize("variable_S_X_assumed", [(False, False)])
@pytest.mark.parametrize("device", ["cpu"])
def test_linear_correlated(device, prior, precompute_XX, include_intercept, variable_S_X_assumed,
                           N=128, P=32, intercept=2.34, T=20 * 1000, T_burnin=3000, report_frequency=1100, seed=1):
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

    samples = []
    if device == "cpu":
        sampler = NormalLikelihoodSampler(X, Y, X_assumed=X_assumed,
                                          precompute_XX=precompute_XX, prior=prior,
                                          compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                                          tau=0.01, c=100.0, include_intercept=include_intercept,
                                          tau_intercept=1.0e-4,
                                          subset_size=12)
    elif device == "gpu":
        sampler = NormalLikelihoodSampler(X.cuda(), Y.cuda(),
                                          X_assumed=X_assumed.cuda() if X_assumed is not None else None,
                                          precompute_XX=precompute_XX, prior=prior,
                                          compute_betas=True, S=S, nu0=0.0, lambda0=0.0,
                                          tau=0.01, c=100.0, include_intercept=include_intercept,
                                          tau_intercept=1.0e-4)

    import time
    t0 = time.time()

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    t1 = time.time()
    print("ELAPSED MCMC", t1-t0)

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    stats = {}
    quantiles = [5.0, 10.0, 20.0, 50.0, 90.0, 95.0]
    q5, q10, q20, q50, q90, q95 = np.percentile(weights, quantiles).tolist()
    s = "5/10/20/50/90/95:  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
    stats['Weight quantiles'] = s.format(q5, q10, q20, q50, q90, q95)
    s = "mean/std/min/max:  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
    stats['Weight moments'] = s.format(weights.mean().item(), weights.std().item(),
                                       weights.min().item(), weights.max().item())
    for k, v in stats.items():
        print(k, v)

    pip = np.dot(samples.pip.T, weights)
    print("pip[0:8]", pip[0:8])
    print("pip[8:16]", pip[8:16])
    assert_close(pip[:2], np.array([0.5, 0.5]), atol=0.2)
    assert_close(pip[2:], np.zeros(P - 2), atol=0.15)

    beta = np.dot(np.transpose(samples.beta), weights)
    assert_close(beta[:2], np.array([0.5, 0.5]), atol=0.15)

    return

    if include_intercept:
        assert_close(beta[-1].item(), intercept, atol=0.1)

    if X_assumed is None:
        assert_close(beta[2:P], np.zeros(P - 2), atol=0.02)
    else:
        assert_close(beta[2:P + 2], np.concatenate([np.zeros(P - 1), np.array([0.5])]), atol=0.02)
