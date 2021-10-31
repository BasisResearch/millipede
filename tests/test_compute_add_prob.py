import math
from types import SimpleNamespace

import pytest
import torch
from common import assert_close
from torch import zeros

from millipede import NormalLikelihoodSampler


def get_sample(gamma, include_bias):
    P = len(gamma)
    sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                             add_prob=zeros(P), _i_prob=zeros(P),
                             _idx=0, weight=0.0)
    sample._active = torch.nonzero(sample.gamma).squeeze(-1)
    if include_bias:
        sample._activeb = torch.cat([sample._active, torch.tensor([P])])
    return sample


def check_gammas(sampler, include_bias, P, compute_log_factor_ratio):
    # TEST GAMMA = 0 0 0
    sample = get_sample([0] * P, include_bias)
    log_odds = sampler._compute_add_prob(sample)
    for p in range(P):
        assert_close(compute_log_factor_ratio([p], []), log_odds[p], atol=1.0e-7)

    # TEST GAMMA = 1 0 0
    sample = get_sample([1] + [0] * (P - 1), include_bias)
    log_odds = sampler._compute_add_prob(sample)
    for p in range(1, P):
        assert_close(compute_log_factor_ratio([0, p], [0]), log_odds[p], atol=1.0e-7)

    # TEST GAMMA = 1 1 0
    sample = get_sample([1, 1] + [0] * (P - 2), include_bias)
    log_odds = sampler._compute_add_prob(sample)
    for p in range(2, P):
        assert_close(compute_log_factor_ratio([0, 1, p], [0, 1]), log_odds[p], atol=1.0e-7)

    # TEST GAMMA = 1 1 1
    sample = get_sample([1, 1, 1] + [0] * (P - 3), include_bias)
    log_odds = sampler._compute_add_prob(sample)
    for p in range(3, P):
        assert_close(compute_log_factor_ratio([0, 1, 2, p], [0, 1, 2]), log_odds[p], atol=1.0e-7)


@pytest.mark.parametrize("precompute_XX", [False, True])
@pytest.mark.parametrize("include_bias", [False, True])
def test_isotropic_compute_add_log_prob(precompute_XX, include_bias, N=5, P=4, tau=0.47, tau_bias=0.11):
    X = torch.randn(N, P).double()
    if include_bias:
        X = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)

    Y = X[:, 0] + 0.2 * torch.randn(N).double()
    sampler = NormalLikelihoodSampler(X, Y, S=1, c=0.0, tau=tau, tau_bias=tau_bias, include_bias=include_bias,
                                      precompute_XX=precompute_XX, prior="isotropic")
    YY = sampler.YY
    Z = sampler.Z

    def compute_log_factor(ind):
        if include_bias:
            ind = ind + [P]
            precision = tau * torch.eye(len(ind))
            precision[-1, -1] = tau_bias
        else:
            precision = tau * torch.eye(len(ind))
        F = torch.inverse(X[:, ind].t() @ X[:, ind] + precision)
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        return -0.5 * N * (YY - ZFZ).log() + 0.5 * F.logdet()

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1) - compute_log_factor(ind0) +\
            sampler.log_h_ratio + 0.5 * math.log(tau)

    check_gammas(sampler, include_bias, P, compute_log_factor_ratio)


@pytest.mark.parametrize("precompute_XX", [False, True])
@pytest.mark.parametrize("include_bias", [False, True])
def test_gprior_compute_add_log_prob(precompute_XX, include_bias, N=5, P=4):
    X = torch.randn(N, P).double()
    if include_bias:
        X = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)

    Y = X[:, 0] + 0.2 * torch.randn(N).double()
    sampler = NormalLikelihoodSampler(X, Y, S=1, tau=0.0, c=0.73, include_bias=include_bias,
                                      precompute_XX=precompute_XX, prior="gprior")
    YY = sampler.YY
    Z = sampler.Z

    def compute_log_factor(ind):
        if include_bias:
            ind = ind + [P]
        F = torch.inverse(X[:, ind].t() @ X[:, ind])
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        return -0.5 * N * (YY - sampler.c_one_c * ZFZ).log()

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1) - compute_log_factor(ind0) + sampler.hc_prefactor

    check_gammas(sampler, include_bias, P, compute_log_factor_ratio)
