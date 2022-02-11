import math

import pytest
import torch
from common import assert_close
from polyagamma import polyagamma_pdf

from millipede import CountLikelihoodSampler


@pytest.mark.parametrize("P", [5, 9])
@pytest.mark.parametrize("N", [3, 19])
def test_sample_omega_binomial(P, N):
    X = torch.randn(N, P).double()
    TC = 3 * torch.ones(N).long()
    TC[N // 2:] = 4
    Y = torch.distributions.Binomial(total_count=TC, logits=X[:, 0]).sample()

    sampler = CountLikelihoodSampler(X, Y, TC=TC, S=1.0, tau=0.01, tau_intercept=1.0e-4)
    sampler.t = 0
    sampler.T_burnin = 0

    sample = sampler.initialize_sample()
    sample.gamma = torch.tensor([1] * 3 + [0] * (P - 3))
    sample._active = torch.nonzero(sample.gamma).squeeze(-1)
    sample._activeb = torch.cat([sample._active, torch.tensor([P])])
    sample = sampler.sample_beta(sample)
    sample = sampler._compute_probs(sample)

    _save_intermediates = {}
    sample = sampler.sample_omega_binomial(sample, _save_intermediates=_save_intermediates)
    TC = _save_intermediates['TC_np']

    lp1 = polyagamma_pdf(_save_intermediates['omega'], h=TC, z=_save_intermediates['psi_prop'],
                         return_log=1).sum().item()
    lp2 = polyagamma_pdf(_save_intermediates['omega_prop'], h=TC, z=_save_intermediates['psi'],
                         return_log=1).sum().item()
    lp3 = polyagamma_pdf(_save_intermediates['omega'], h=TC, return_log=1).sum().item()
    lp4 = polyagamma_pdf(_save_intermediates['omega_prop'], h=TC, return_log=1).sum().item()

    actual = _save_intermediates['accept234'].item()
    expected = lp1 - lp2 - lp3 + lp4
    assert_close(actual, expected, atol=1.0e-13)


@pytest.mark.parametrize("P", [5, 9])
@pytest.mark.parametrize("N", [3, 6])
def test_sample_omega_negative_binomial(P, N, intercept=-0.77):
    X = torch.randn(N, P).double()
    logits = intercept + X[:, 0] + 0.2 * torch.randn(N)
    Y = torch.distributions.Poisson(logits.exp()).sample()

    sampler = CountLikelihoodSampler(X, Y, psi0=intercept, TC=None, S=1.0, tau=0.01, tau_intercept=1.0e-4)
    sampler.t = 0
    sampler.T_burnin = 0

    sample = sampler.initialize_sample()
    sample.gamma = torch.tensor([1] * 3 + [0] * (P - 3))
    sample._active = torch.nonzero(sample.gamma).squeeze(-1)
    sample._activeb = torch.cat([sample._active, torch.tensor([P])])
    sample = sampler.sample_beta(sample)

    _save_intermediates = {}
    sample = sampler.sample_omega_nb(sample, _save_intermediates=_save_intermediates)

    T_prop = _save_intermediates['T_prop']
    T_curr = _save_intermediates['T_curr']
    omega = _save_intermediates['omega']
    omega_prop = _save_intermediates['omega_prop']
    N_delta_nu = N * _save_intermediates['delta_nu']

    lp1 = polyagamma_pdf(omega, h=T_curr, z=_save_intermediates['psi_mixed_prop'], return_log=1).sum().item()
    lp2 = polyagamma_pdf(omega_prop, h=T_prop, z=_save_intermediates['psi_mixed'], return_log=1).sum().item()
    lp3 = polyagamma_pdf(omega, h=T_curr, return_log=1).sum().item()
    lp4 = polyagamma_pdf(omega_prop, h=T_prop, return_log=1).sum().item()

    actual = _save_intermediates['accept23'].item() - math.log(2.0) * N_delta_nu
    expected = lp1 - lp2 - lp3 + lp4
    assert_close(actual, expected, atol=1.0e-13)
