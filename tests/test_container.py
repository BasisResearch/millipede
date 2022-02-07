import math
from types import SimpleNamespace

import numpy as np
import pytest
from common import assert_close

from millipede.containers import SimpleSampleContainer, StreamingSampleContainer


@pytest.mark.parametrize("include_intercept", [True, False])
def test_containers(include_intercept, P=101, atol=1.0e-7):
    c1 = SimpleSampleContainer()
    c2 = StreamingSampleContainer()

    for _ in range(4):
        gamma = np.random.binomial(1, 0.5 * np.ones(P))
        beta = np.random.randn(P)
        beta = beta * np.array(gamma, dtype=beta.dtype)
        if include_intercept:
            beta = np.concatenate([beta, np.random.rand(1)])

        sample = SimpleNamespace(gamma=gamma,
                                 beta=beta,
                                 add_prob=np.random.rand(P),
                                 log_nu=np.random.randn(),
                                 S_alpha=np.random.rand(),
                                 S_beta=np.random.rand(),
                                 weight=np.random.rand())
        c1(sample)
        c2(sample)

    assert_close(c1.pip, c2.pip, atol=atol)
    assert_close(c1.beta, c2.beta, atol=atol)
    assert_close(c1.conditional_beta, c2.conditional_beta, atol=atol)
    assert_close(c1.conditional_beta_std, c2.conditional_beta_std, atol=atol)
    assert_close(c1.S_alpha, c2.S_alpha, atol=atol)

    for p in range(P + int(include_intercept)):
        beta = c1.samples.beta[:, p]
        nz = np.nonzero(beta)[0]
        beta = beta[nz]
        weight = c1.samples.weight[nz]

        beta_mean = (np.sum(weight * beta) / np.sum(weight)).item() if len(beta) > 0 else 0.0
        expected = beta_mean if len(beta) > 0 else 0.0
        assert_close(c1.conditional_beta[p].item(), expected)

        beta_var = (np.sum(np.square(beta) * weight) / np.sum(weight)).item() - beta_mean ** 2 if len(beta) > 0 else 0.0
        expected = math.sqrt(np.clip(beta_var, a_min=0.0, a_max=None)) if len(beta) > 0 else 0.0
        assert_close(c1.conditional_beta_std[p].item(), expected)

    assert_close(c1.log_nu, c2.log_nu, atol=atol)
    assert_close(c1.log_nu_std, c2.log_nu_std, atol=atol)
    assert_close(c1.nu, c2.nu, atol=atol)
    assert_close(c1.nu_std, c2.nu_std, atol=atol)
    assert_close(c1.beta_std, c2.beta_std, atol=atol)
