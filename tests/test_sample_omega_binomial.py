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

    sampler = CountLikelihoodSampler(X, Y, TC=TC, S=1.0, tau=0.01, tau_bias=1.0e-4)
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
