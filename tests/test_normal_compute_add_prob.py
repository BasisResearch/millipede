import math
from types import SimpleNamespace

import pytest
import torch
from common import assert_close
from torch import zeros

from millipede import NormalLikelihoodSampler


def get_sample(gamma, included_covariates, log_h_ratio, active_subset):
    P = len(gamma)
    sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                             pip=zeros(P), _i_prob=zeros(P),
                             _idx=0, weight=0.0)
    sample._active = torch.nonzero(sample.gamma).squeeze(-1)
    sample._active_subset = active_subset
    if len(included_covariates) > 0:
        sample._activeb = torch.cat([sample._active, torch.tensor(included_covariates)])
    sample._log_h_ratio = log_h_ratio
    return sample


def assert_close_active(a, b, active, atol=1.0e-7):
    if active:
        assert_close(a, b, atol=atol)


def check_gammas(sampler, included_covariates, P, compute_log_factor_ratio, active_subset=None):
    active_subset = torch.arange(P) if active_subset is None else active_subset
    # TEST GAMMA = 0 0 0
    sample = get_sample([0] * P, included_covariates, sampler.log_h_ratio, active_subset)
    log_odds = sampler._compute_add_prob(sample)

    for p in range(P):
        assert_close_active(compute_log_factor_ratio([p], []), log_odds[p], p in active_subset, atol=1.0e-7)

    # TEST GAMMA = 1 0 0
    sample = get_sample([1] + [0] * (P - 1), included_covariates, sampler.log_h_ratio, active_subset)
    log_odds = sampler._compute_add_prob(sample)

    assert_close_active(compute_log_factor_ratio([0], []), log_odds[0], 0 in active_subset, atol=1.0e-7)
    for p in range(1, P):
        assert_close_active(compute_log_factor_ratio([0, p], [0]), log_odds[p], p in active_subset, atol=1.0e-7)

    # TEST GAMMA = 1 1 0
    sample = get_sample([1, 1] + [0] * (P - 2), included_covariates, sampler.log_h_ratio, active_subset)
    log_odds = sampler._compute_add_prob(sample)

    assert_close_active(compute_log_factor_ratio([0, 1], [1]), log_odds[0], 0 in active_subset, atol=1.0e-7)
    assert_close_active(compute_log_factor_ratio([0, 1], [0]), log_odds[1], 1 in active_subset, atol=1.0e-7)
    for p in range(2, P):
        assert_close_active(compute_log_factor_ratio([0, 1, p], [0, 1]), log_odds[p], p in active_subset, atol=1.0e-7)

    # TEST GAMMA = 1 1 1
    sample = get_sample([1, 1, 1] + [0] * (P - 3), included_covariates, sampler.log_h_ratio, active_subset)
    log_odds = sampler._compute_add_prob(sample)

    assert_close_active(compute_log_factor_ratio([0, 1, 2], [1, 2]), log_odds[0], 0 in active_subset, atol=1.0e-7)
    assert_close_active(compute_log_factor_ratio([0, 1, 2], [0, 2]), log_odds[1], 1 in active_subset, atol=1.0e-7)
    assert_close_active(compute_log_factor_ratio([0, 1, 2], [0, 1]), log_odds[2], 2 in active_subset, atol=1.0e-7)
    for p in range(3, P):
        assert_close_active(compute_log_factor_ratio([0, 1, 2, p], [0, 1, 2]),
                            log_odds[p], p in active_subset, atol=1.0e-7)


@pytest.mark.parametrize("P", [4, 7])
@pytest.mark.parametrize("P_assumed", [0, 1, 2])
@pytest.mark.parametrize("precompute_XX", [False, True])
@pytest.mark.parametrize("include_intercept", [False, True])
@pytest.mark.parametrize("subset_size", [None, -1])
def test_isotropic_compute_add_log_prob(P, P_assumed, precompute_XX, include_intercept, subset_size,
                                        N=11, tau=0.47, tau_intercept=0.11):
    subset_size = None if subset_size is None else P // 2
    X = torch.randn(N, P).double()
    X_assumed = torch.randn(N, P_assumed).double() if P_assumed > 0 else None

    Y = X[:, 0] + 0.2 * torch.randn(N).double()

    S = 1.0 if include_intercept else (torch.randn(P) / 100).exp().double() / P
    sigma_scale_factor = None if subset_size is None else (torch.ones(N) + 0.1 * torch.rand(N)).double()
    sampler = NormalLikelihoodSampler(X, Y, X_assumed=X_assumed,
                                      sigma_scale_factor=sigma_scale_factor, S=S, c=0.0,
                                      tau=tau, tau_intercept=tau_intercept, include_intercept=include_intercept,
                                      precompute_XX=precompute_XX, prior="isotropic", subset_size=subset_size)
    sigma_scale_factor = torch.ones(N).double() if sigma_scale_factor is None else sigma_scale_factor

    included_covariates = []
    if P_assumed > 0:
        X = torch.cat([X, X_assumed], dim=-1)
        included_covariates = list(range(P, P + P_assumed))
        if include_intercept:
            included_covariates.append(P + P_assumed)
    else:
        if include_intercept:
            included_covariates.append(P)
    if include_intercept:
        X = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)

    YY = sampler.YY
    Z = sampler.Z

    def compute_log_factor(ind):
        ind.extend(included_covariates)
        precision = tau * torch.eye(len(ind))
        if include_intercept:
            precision[-1, -1] = tau_intercept
        X_ind = X[:, ind] / sigma_scale_factor.unsqueeze(-1)
        F = torch.inverse(X_ind.t() @ X_ind + precision)
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        return -0.5 * N * (YY - ZFZ).log() + 0.5 * F.logdet()

    def compute_log_factor_ratio(ind1, ind0):
        added_idx = list(set(ind1) - set(ind0))[0]
        log_h_ratio = sampler.log_h_ratio[added_idx] if isinstance(sampler.log_h_ratio, torch.Tensor) \
            else sampler.log_h_ratio
        return compute_log_factor(ind1) - compute_log_factor(ind0) + log_h_ratio + 0.5 * math.log(tau)

    active_subset = None if subset_size is None else torch.arange(P)[torch.randperm(P)[:subset_size]]
    check_gammas(sampler, included_covariates, P, compute_log_factor_ratio, active_subset)


@pytest.mark.parametrize("P", [4, 7])
@pytest.mark.parametrize("P_assumed", [0, 1, 2])
@pytest.mark.parametrize("precompute_XX", [False, True])
@pytest.mark.parametrize("include_intercept", [True, False])
@pytest.mark.parametrize("subset_size", [None, -1])
def test_gprior_compute_add_log_prob(P, P_assumed, precompute_XX, include_intercept, subset_size, N=11):
    X = torch.randn(N, P).double()
    X_assumed = torch.randn(N, P_assumed).double() if P_assumed > 0 else None
    Y = X[:, 0] + 0.2 * torch.randn(N).double()
    subset_size = None if subset_size is None else P // 2

    sampler = NormalLikelihoodSampler(X, Y, X_assumed=X_assumed, S=1.0,
                                      tau=0.0, c=0.73, include_intercept=include_intercept,
                                      precompute_XX=precompute_XX, prior="gprior", subset_size=subset_size)

    included_covariates = []
    if P_assumed > 0:
        X = torch.cat([X, X_assumed], dim=-1)
        included_covariates = list(range(P, P + P_assumed))
        if include_intercept:
            included_covariates.append(P + P_assumed)
    else:
        if include_intercept:
            included_covariates.append(P)
    if include_intercept:
        X = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)

    YY = sampler.YY
    Z = sampler.Z

    def compute_log_factor(ind):
        ind.extend(included_covariates)
        F = torch.inverse(X[:, ind].t() @ X[:, ind])
        ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
        return -0.5 * N * (YY - sampler.c_one_c * ZFZ).log()

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1) - compute_log_factor(ind0) + sampler.log_h_ratio - sampler.log_one_c_sqrt

    active_subset = None if subset_size is None else torch.arange(P)[torch.randperm(P)[:subset_size]]
    check_gammas(sampler, included_covariates, P, compute_log_factor_ratio, active_subset)
