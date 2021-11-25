from types import SimpleNamespace
import math
import time
import argparse

import numpy as np

import torch
from torch.distributions import Binomial, Bernoulli
from torch.linalg import norm
from torch import randn, sigmoid
from torch import triangular_solve as trisolve
from torch import cholesky_solve as chosolve

from torch import einsum, matmul
from torch.distributions import Categorical

from .normal import NormalLikelihoodSampler
from .util import get_loo_inverses, leave_one_out, safe_cholesky


class ASISampler(NormalLikelihoodSampler):
    def __init__(self, X, Y, S=5,
                 prior='isotropic',
                 include_intercept=True,
                 tau=0.01, tau_intercept=1.0e-4, c=100.0,
                 nu0=0.0, lambda0=0.0,
                 precompute_XX=False,
                 compute_betas=False, verbose_constructor=True):
        super().__init__(X=X, Y=Y, S=S, prior=prior, include_intercept=include_intercept,
                         tau=tau, tau_intercept=tau_intercept, c=c,
                         nu0=nu0, lambda0=lambda0, explore=5, precompute_XX=precompute_XX,
                         compute_betas=compute_betas, verbose_constructor=False)

        self.epsilon_asi = 0.1 / self.P
        self.zeta = torch.tensor(0.90)
        self.pi = self.h * torch.ones(self.P)
        self.acc_target = 0.25
        self.lambda_exponent = 0.75
        self.log_h = math.log(self.h)
        self.log_1mh = math.log1p(-self.h)

        self.update_zeta_AD(0.0)

        print("Initialized ASISampler with (N, P, S) = ({}, {}, {:.1f})".format(self.N, self.P, S))

    def logit_eps(self, x):
        x = (x - self.epsilon_asi) / (1.0 - x - self.epsilon_asi)
        return torch.log(x)

    def sigmoid_eps(self, y):
        return (1.0 - self.epsilon_asi) * sigmoid(y) + self.epsilon_asi * sigmoid(-y)

    def update_zeta_AD(self, delta):
        self.zeta = self.sigmoid_eps(self.logit_eps(self.zeta) + delta)
        pi = self.epsilon_asi + (1.0 - 2.0 * self.epsilon_asi) * self.pi
        self.A = self.zeta * torch.min(torch.ones(1, device=pi.device, dtype=pi.dtype), pi / (1.0 - pi))
        self.D = self.zeta * torch.min(torch.ones(1, device=pi.device, dtype=pi.dtype), (1.0 - pi) / pi)

    def initialize_sample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        sample = SimpleNamespace(gamma=torch.zeros(self.P, device=self.device).bool(),
                                 add_prob=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64))
        if self.compute_betas:
            sample.beta = torch.zeros(self.P, device=self.device, dtype=self.dtype)

        if self.include_intercept:
            sample._activeb = torch.tensor([self.P], device=self.device, dtype=torch.int64)

        sample = self._compute_probs(sample)

        return sample

    def mcmc_move(self, sample):
        self.t += 1

        gamma_curr = sample.gamma.double()
        q_curr = Bernoulli(gamma_curr * self.D + (1.0 - gamma_curr) * self.A)
        flips = q_curr.sample()
        flips_bool = flips.bool()

        gamma_prop = sample.gamma.clone()
        gamma_prop[flips_bool] = ~gamma_prop[flips_bool]

        q_prop = Bernoulli(gamma_prop.double() * self.D + (1.0 - gamma_prop.double()) * self.A)

        logq_prop = q_curr.log_prob(flips).sum().item()
        logq_curr = q_prop.log_prob(flips).sum().item()

        _, activeb = self.get_active_indices(sample)
        activeb_prop = torch.cat([torch.nonzero(gamma_prop).squeeze(-1), torch.tensor([self.P])])

        log_target_curr, L_curr = self.compute_log_target(sample._omega, activeb=activeb)
        log_target_prop, L_prop, beta_prop, beta_prop_active, beta_mean_prop = \
            self.compute_log_target(sample._omega, activeb=activeb_prop, sample_beta=True)

        accept = log_target_prop - log_target_curr + logq_curr - logq_prop
        accept = min(1.0, accept.exp().item())

        if self.t > self.Tb:
            self.gamma_acc_probs.append(accept)
        accept_bool = self.uniform_dist.sample().item() < accept

        if accept_bool:
            sample.gamma = gamma_prop
            sample._beta = beta_prop
            sample.beta_mean = beta_mean_prop
            sample._psi = torch.mv(self.Xb[:, activeb_prop], beta_prop_active)
            self.L_active = L_prop
        else:
            self.L_active = L_curr

        sample.add_prob = self._compute_add_prob(sample)

        if self.t <= self.Tb:
            t = self.t
            self.pi = (t / (t + 1)) * self.pi + sample.add_prob / (t + 1)
            phi_t = 1.0 / math.pow(t, self.lambda_exponent)
            self.update_zeta_AD(phi_t * (accept - self.acc_target))

        sample = self.sample_omega(sample)
        return sample

    def compute_log_target(self, omega, sample=None, activeb=None, sample_beta=False):
        if activeb is None:
            _, activeb = self.get_active_indices(sample)
        Xb_active = self.Xb[:, activeb]
        num_active = activeb.size(-1) - 1
        num_inactive = self.P - num_active

        precision = Xb_active.t() @ (omega.unsqueeze(-1) * Xb_active)
        precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)

        L = torch.cholesky(precision, upper=False)
        LZ = trisolve(self.Z[activeb].unsqueeze(-1), L, upper=False)[0].squeeze(-1)
        logdet = L.diag().log().sum() - L.size(-1) * self.half_log_tau
        h_factor = num_active * self.log_h + num_inactive * self.log_1mh

        if sample_beta:
            mean = chosolve(self.X_kappa[activeb].unsqueeze(-1), L).squeeze(-1)
            beta_active = mean + trisolve(torch.randn(activeb.size(-1), 1), L, upper=False)[0].squeeze(-1)
            beta, beta_mean = torch.zeros(self.P + 1), torch.zeros(self.P + 1)
            beta[activeb], beta_mean[activeb] = beta_active, mean
            return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet + h_factor, L, beta, beta_active, beta_mean

        return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet + h_factor, L
