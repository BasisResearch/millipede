import math
from types import SimpleNamespace

import pytest
import torch
from common import assert_close
from torch import zeros

from millipede import NormalLikelihoodSampler


@pytest.mark.parametrize("precompute_XX", [False, True])
def test_isotropic_compute_add_log_prob(precompute_XX, N=35, P=9, tau=0.47):
    X = torch.randn(N, P).double()
    Y = X[:, 0] + 0.2 * torch.randn(N).double()
    sampler = NormalLikelihoodSampler(X, Y, S=1, c=0.0, tau=tau,
                                      precompute_XX=precompute_XX, prior="isotropic")
    YY = sampler.YY
    Z = sampler.Z

    def compute_log_factor(ind):
        F = torch.inverse(X[:, ind].t() @ X[:, ind] + tau * torch.eye(len(ind)))
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        return -0.5 * N * (YY - ZFZ).log() + 0.5 * F.logdet()

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1) - compute_log_factor(ind0) +\
            sampler.log_h_ratio + 0.5 * math.log(tau)

    # TEST GAMMA = 0 0 0 0
    sample = SimpleNamespace(gamma=zeros(P).bool(), add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
    log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
    for p in range(P):
        assert_close(compute_log_factor_ratio([p], []), log_odds[p], atol=1.0e-7)

    # TEST GAMMA = 1 0 0 0
    gamma = [1] + [0] * (P - 1)
    sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                             add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
    log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
    for p in range(1, P):
        assert_close(compute_log_factor_ratio([0, p], [0]), log_odds[p], atol=1.0e-7)

    # TEST GAMMA = 1 1 0 0
    gamma = [1, 1] + [0] * (P - 2)
    sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                             add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
    log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
    for p in range(2, 5):
        assert_close(compute_log_factor_ratio([0, 1, p], [0, 1]), log_odds[p], atol=1.0e-7)


@pytest.mark.parametrize("precompute_XX", [False, True])
def test_gprior_compute_add_log_prob(precompute_XX, N=35, P=9):
    X = torch.randn(N, P).double()
    Y = X[:, 0] + 0.2 * torch.randn(N).double()
    sampler = NormalLikelihoodSampler(X, Y, S=1, tau=0.0, c=0.73,
                                      precompute_XX=precompute_XX, prior="gprior")
    YY = sampler.YY
    Z = sampler.Z

    def compute_log_factor(ind):
        F = torch.inverse(X[:, ind].t() @ X[:, ind])
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        return -0.5 * N * (YY - sampler.c_one_c * ZFZ).log()

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1) - compute_log_factor(ind0) + sampler.hc_prefactor

    # TEST GAMMA = 0 0 0 0
    sample = SimpleNamespace(gamma=zeros(P).bool(), add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
    log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
    for p in range(P):
        assert_close(compute_log_factor_ratio([p], []), log_odds[p], atol=1.0e-7)

    # TEST GAMMA = 1 0 0 0
    gamma = [1] + [0] * (P - 1)
    sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                             add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
    log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
    for p in range(1, P):
        assert_close(compute_log_factor_ratio([0, p], [0]), log_odds[p], atol=1.0e-7)

    # TEST GAMMA = 1 1 0 0
    gamma = [1, 1] + [0] * (P - 2)
    sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                             add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
    log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
    for p in range(2, P):
        assert_close(compute_log_factor_ratio([0, 1, p], [0, 1]), log_odds[p], atol=1.0e-7)
