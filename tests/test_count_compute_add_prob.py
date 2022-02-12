from types import SimpleNamespace

import pytest
import torch
from common import assert_close
from torch import zeros

from millipede import CountLikelihoodSampler


@pytest.mark.parametrize("P", [4, 7])
@pytest.mark.parametrize("P_assumed", [0, 1, 2])
@pytest.mark.parametrize("N", [1, 19])
def test_compute_add_prob(P, P_assumed, N, tau=0.47, tau_intercept=0.13, atol=1.0e-6):
    X = torch.randn(N, P).double()
    X_assumed = torch.randn(N, P_assumed).double() if P_assumed > 0 else None

    if X_assumed is not None:
        Xb = torch.cat([X, X_assumed, torch.ones(X.size(0), 1).double()], dim=-1)
        included_covariates = list(range(P, P + P_assumed + 1))
    else:
        Xb = torch.cat([X, torch.ones(X.size(0), 1).double()], dim=-1)
        included_covariates = [P]

    TC = 10 * torch.ones(N).long()
    TC[N // 2:] = 5
    Y = torch.distributions.Binomial(total_count=TC, logits=X[:, 0]).sample()
    kappa = Y - 0.5 * TC
    omega = torch.randn(N).exp().double()
    Xbom = Xb * omega.sqrt().unsqueeze(-1)
    beta = torch.randn(P + P_assumed + 1).double()
    Z = torch.einsum("np,n->p", Xb, kappa)

    def compute_log_factor(ind):
        precision = tau * torch.eye(len(ind))
        precision[-1, -1] = tau_intercept
        F_inv = Xbom[:, ind].t() @ Xbom[:, ind] + precision
        F = torch.inverse(F_inv)
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        logdet = -torch.logdet(precision) + torch.logdet(F_inv)
        return 0.5 * ZFZ - 0.5 * logdet

    sampler = CountLikelihoodSampler(X, Y, X_assumed=X_assumed,
                                     TC=TC, S=1, tau=tau, tau_intercept=tau_intercept)

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1 + included_covariates) - compute_log_factor(ind0 + included_covariates) +\
            sampler.log_h_ratio

    def get_sample(gamma):
        sample = SimpleNamespace(gamma=gamma.bool(), beta=beta, beta_mean=beta,
                                 _omega=omega, idx=0, weight=0, _psi0=0.0, _kappa=kappa,
                                 _kappa_omega=kappa, _Z=Z, _log_h_ratio=sampler.log_h_ratio)
        sample._active = torch.nonzero(sample.gamma).squeeze(-1)
        sample._activeb = torch.cat([sample._active, torch.tensor(included_covariates)])
        return sample

    # 0 0 0 0
    sample = sampler.sample_beta(get_sample(zeros(P)))
    log_odds = sampler._compute_add_prob(sample)

    for p in range(P):
        assert_close(compute_log_factor_ratio([p], []), log_odds[p], atol=atol)

    # 1 0 0 0
    sample = sampler.sample_beta(get_sample(torch.tensor([1] + [0] * (P - 1))))
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0], []), log_odds[0], atol=atol)
    for p in range(1, P):
        assert_close(compute_log_factor_ratio([0, p], [0]), log_odds[p], atol=atol)

    # 1 1 0 0
    sample = sampler.sample_beta(get_sample(torch.tensor([1, 1] + [0] * (P - 2))))
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0, 1], [1]), log_odds[0], atol=atol)
    assert_close(compute_log_factor_ratio([0, 1], [0]), log_odds[1], atol=atol)
    for p in range(2, P):
        assert_close(compute_log_factor_ratio([0, 1, p], [0, 1]), log_odds[p], atol=atol)

    # 1 1 1 0
    sample = sampler.sample_beta(get_sample(torch.tensor([1, 1, 1] + [0] * (P - 3))))
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0, 1, 2], [1, 2]), log_odds[0], atol=atol)
    assert_close(compute_log_factor_ratio([0, 1, 2], [0, 2]), log_odds[1], atol=atol)
    assert_close(compute_log_factor_ratio([0, 1, 2], [0, 1]), log_odds[2], atol=atol)

    for p in range(3, P):
        assert_close(compute_log_factor_ratio([0, 1, 2, p], [0, 1, 2]), log_odds[p], atol=atol)
