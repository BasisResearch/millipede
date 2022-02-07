from types import SimpleNamespace

import pytest
import torch
from torch import zeros

from common import assert_close
from millipede import CountLikelihoodSampler


@pytest.mark.parametrize("P", [4, 7])
@pytest.mark.parametrize("N", [1, 19])
def test_compute_add_prob(P, N, tau=0.47, tau_intercept=0.13, atol=1.0e-6):
    X = torch.randn(N, P).double()
    Xb = torch.cat([X, torch.ones(X.size(0), 1)], dim=-1).double()
    TC = 10 * torch.ones(N).long()
    TC[N // 2:] = 5
    Y = torch.distributions.Binomial(total_count=TC, logits=X[:, 0]).sample()
    kappa = Y - 0.5 * TC
    omega = torch.randn(N).exp().double()
    beta = torch.randn(P + 1).double()
    Xbom = Xb * omega.sqrt().unsqueeze(-1)
    Z = torch.einsum("np,n->p", Xb, kappa)

    def compute_log_factor(ind):
        precision = tau * torch.eye(len(ind))
        precision[-1, -1] = tau_intercept
        F_inv = Xbom[:, ind].t() @ Xbom[:, ind] + precision
        F = torch.inverse(F_inv)
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        logdet = -torch.logdet(precision) + torch.logdet(F_inv)
        return 0.5 * ZFZ - 0.5 * logdet

    sampler = CountLikelihoodSampler(X, Y, TC=TC, S=1, tau=tau, tau_intercept=tau_intercept)

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1) - compute_log_factor(ind0) + sampler.log_h_ratio

    def get_sample(gamma):
        sample = SimpleNamespace(gamma=gamma.bool(), add_prob=zeros(P).double(), _i_prob=zeros(P).double(),
                                 beta_mean=beta, _omega=omega, idx=0, weight=0, _psi0=0.0, _kappa=kappa,
                                 _kappa_omega=kappa, _Z=Z, beta=beta, _log_h_ratio=sampler.log_h_ratio)
        sample._active = torch.nonzero(sample.gamma).squeeze(-1)
        sample._activeb = torch.cat([sample._active, torch.tensor([P])])
        return sample

    # 0 0 0 0
    sample = sampler.sample_beta(get_sample(zeros(P)))
    log_odds = sampler._compute_add_prob(sample)

    for p in range(P):
        assert_close(compute_log_factor_ratio([p, P], [P]), log_odds[p], atol=atol)

    # 1 0 0 0
    sample = sampler.sample_beta(get_sample(torch.tensor([1] + [0] * (P - 1))))
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0, P], [P]), log_odds[0], atol=atol)
    for p in range(1, P):
        assert_close(compute_log_factor_ratio([0, p, P], [0, P]), log_odds[p], atol=atol)

    # 1 1 0 0
    sample = sampler.sample_beta(get_sample(torch.tensor([1, 1] + [0] * (P - 2))))
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0, 1, P], [1, P]), log_odds[0], atol=atol)
    assert_close(compute_log_factor_ratio([0, 1, P], [0, P]), log_odds[1], atol=atol)
    for p in range(2, P):
        assert_close(compute_log_factor_ratio([0, 1, p, P], [0, 1, P]), log_odds[p], atol=atol)

    # 1 1 1 0
    sample = sampler.sample_beta(get_sample(torch.tensor([1, 1, 1] + [0] * (P - 3))))
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0, 1, 2, P], [1, 2, P]), log_odds[0], atol=atol)
    assert_close(compute_log_factor_ratio([0, 1, 2, P], [0, 2, P]), log_odds[1], atol=atol)
    assert_close(compute_log_factor_ratio([0, 1, 2, P], [0, 1, P]), log_odds[2], atol=atol)

    for p in range(3, P):
        assert_close(compute_log_factor_ratio([0, 1, 2, p, P], [0, 1, 2, P]), log_odds[p], atol=atol)
