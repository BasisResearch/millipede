from types import SimpleNamespace
import math
import time

import numpy as np

import torch
from torch import dot
from torch.distributions import Categorical, Uniform
from torch.linalg import norm
from torch.nn.functional import softplus
from torch import matmul, einsum, sigmoid
from torch import triangular_solve as trisolve
from torch import cholesky_solve as chosolve

from polyagamma import random_polyagamma

from .util import namespace_to_numpy, stack_namespaces, leave_one_out, safe_cholesky
from .sampler import MCMCSampler


class CountLikelihoodSampler(MCMCSampler):
    def __init__(self, X, Y, T=None, S=5, explore=5.0, tau=0.01,
                 log_nu_rw_scale=0.03, omega_mh=True, psi0=None):
        super().__init__()
        assert (T is None and psi0 is not None) or (T is not None and psi0 is None)

        self.Xb = torch.cat([X, torch.ones(X.size(0), 1)], dim=-1)
        self.Y = Y
        self.Y64 = self.Y.type_as(X)
        self.tau = tau

        self.negbin = psi0 is not None
        if self.negbin:
            assert psi0.shape == Y.shape or psi0.shape == ()
            self.psi0 = psi0
            self.log_nu_rw_scale = log_nu_rw_scale
        else:
            assert Y.shape == T.shape
            self.T = T
            self.T_np = T.data.cpu().numpy()
            self.T64 = self.T.type_as(X)

        self.N, self.P = X.shape
        assert self.N == Y.size(-1)

        if S >= self.P or S <= 0:
            raise ValueError("S must satisfy 0 < S < P")

        self.h = S / self.P
        self.explore = explore / self.P
        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        self.half_log_tau = 0.5 * math.log(tau)

        self.epsilon = 1.0e-18
        self.xi = torch.tensor([5.0])
        self.xi_target = 0.25

        self.rng = np.random.default_rng(0)
        self.L_active = None
        self.omega_mh = omega_mh
        self.one = torch.ones_like(self.Xb[0, 0])

        self.uniform_dist = Uniform(0.0, torch.ones_like(self.one))

        s = "Initialized {}-PGSampler with (N, P, S, epsilon) = ({}, {}, {:.1f}, {:.1f})"
        print(s.format("NegBin" if self.negbin else "Binomial", self.N, self.P, S, explore))

    def initialize_sample(self):
        self.accepted_omega_updates = 0
        self.attempted_omega_updates = 0
        self.acc_probs = []

        if not self.negbin:
            log_nu = None
            _omega = torch.from_numpy(random_polyagamma(self.T_np, random_state=self.rng)).type_as(self.Xb)
        else:
            log_nu = torch.tensor(math.log(3.0))
            _omega = torch.from_numpy(random_polyagamma(self.Y.data.cpu().numpy() + log_nu.exp().item(),
                                                        random_state=self.rng)).type_as(self.Xb)

        _psi0 = self.psi0 - log_nu if self.negbin else 0.0
        _kappa = 0.5 * (self.Y - log_nu.exp()) if self.negbin else self.Y - 0.5 * self.T
        _kappa_omega = _kappa - _omega * _psi0
        _Z = einsum("np,n->p", self.Xb, _kappa_omega)

        sample = SimpleNamespace(gamma=torch.zeros(self.P).bool(),
                                 add_prob=torch.zeros(self.P),
                                 _i_prob=torch.zeros(self.P),
                                 _psi=torch.zeros(self.N),
                                 beta_mean=torch.zeros(self.P + 1),
                                 beta=torch.zeros(self.P + 1),
                                 _omega=_omega,
                                 _idx=0,
                                 weight=0,
                                 log_nu=log_nu,
                                 _kappa=_kappa,
                                 _kappa_omega=_kappa_omega,
                                 _Z=_Z,
                                 _psi0=_psi0,
                                 _active=torch.tensor([], dtype=torch.int64),
                                 _activeb=torch.tensor([self.P], dtype=torch.int64))

        sample = self.sample_beta(sample)  # populate self.L_active
        sample = self._compute_probs(sample)
        return sample

    def _compute_add_prob(self, sample, return_log_odds=False):
        active, activeb = sample._active, sample._activeb
        inactive = torch.nonzero(~sample.gamma).squeeze(-1)
        num_active = active.size(-1)
        assert num_active < self.P and num_active <= self.N

        X_omega = self.Xb * sample._omega.sqrt().unsqueeze(-1)
        X_omega_k = X_omega[:, inactive]

        Z_k = sample._Z[inactive]
        X_omega_active = X_omega[:, activeb]
        Z_active = sample._Z[activeb]

        Zt_active = trisolve(Z_active.unsqueeze(-1), self.L_active, upper=False)[0].squeeze(-1)
        Xt_active = trisolve(X_omega_active.t(), self.L_active, upper=False)[0].t()
        XtZt_active = einsum("np,p->n", Xt_active, Zt_active)

        XX_k = norm(X_omega_k, dim=0).pow(2.0)
        G_k_inv = XX_k + self.tau - norm(einsum("ni,nk->ik", Xt_active, X_omega_k), dim=0).pow(2.0)
        W_k_sq = (einsum("np,n->p", X_omega_k, XtZt_active) - Z_k).pow(2.0) / (G_k_inv + self.epsilon)
        log_det_ratio_inactive = -0.5 * G_k_inv.log() + self.half_log_tau

        if num_active > 1:
            active_loo = leave_one_out(active)  # I  I-1
            active_loob = torch.cat([active_loo,
                                     (self.P * torch.ones(active_loo.size(0))).long().unsqueeze(-1)], dim=-1)
            X_active_loo = X_omega[:, active_loob].permute(1, 2, 0)  # I I N
            XX_active_loo = matmul(X_active_loo, X_active_loo.transpose(-1, -2))  # I I I
            XX_active_loo.diagonal(dim1=-2, dim2=-1).add_(self.tau)

            Z_active_loo = sample._Z[active_loob]
            L_XX_active_loo = safe_cholesky(XX_active_loo)
            Zt_active_loo_sq = trisolve(Z_active_loo.unsqueeze(-1),
                                        L_XX_active_loo, upper=False)[0].squeeze(-1).pow(2.0).sum(-1)
            log_det_ratio_active = L_XX_active_loo.diagonal(dim1=-1, dim2=-2).log().sum(-1) -\
                self.L_active.diagonal(dim1=-1, dim2=-2).log().sum(-1) + self.half_log_tau
        elif num_active == 1:
            tau_plus_omega = self.tau + sample._omega.sum()
            Zt_active_loo_sq = sample._kappa_omega.sum().pow(2.0) / tau_plus_omega
            Xom_active = X_omega[:, active].squeeze(-1)
            G_k_inv = Xom_active.pow(2.0).sum() + self.tau -\
                (Xom_active * sample._omega.sqrt()).sum().pow(2.0) / tau_plus_omega
            log_det_ratio_active = -0.5 * G_k_inv.log() + self.half_log_tau
        elif num_active == 0:
            Zt_active_loo_sq = 0.0  # dummy values since no active covariates
            log_det_ratio_active = torch.tensor(0.0)

        log_odds_inactive = 0.5 * W_k_sq + log_det_ratio_inactive + self.log_h_ratio
        log_odds_active = 0.5 * (Zt_active.pow(2.0).sum() - Zt_active_loo_sq) + log_det_ratio_active + self.log_h_ratio

        log_odds = self.Xb.new_zeros(self.P)
        log_odds[inactive] = log_odds_inactive
        log_odds[active] = log_odds_active

        if return_log_odds:
            return log_odds

        add_prob = sigmoid(log_odds)
        return add_prob

    def _compute_probs(self, sample):
        sample.add_prob = self._compute_add_prob(sample)

        gamma = sample.gamma.double()
        prob_gamma_i = gamma * sample.add_prob + (1.0 - gamma) * (1.0 - sample.add_prob)
        i_prob = 0.5 * (sample.add_prob + self.explore) / (prob_gamma_i + self.epsilon)

        if self.t <= self.T_burnin:
            self.xi += (self.xi_target - self.xi / (self.xi + i_prob.sum())) / math.sqrt(self.t + 1)
            if self.t == self.T_burnin:
                print("[step {}]    Final adapted xi: {:.3f}".format(self.t, self.xi.item()))

        sample._i_prob = torch.cat([self.xi, i_prob])

        return sample

    def gibbs_move(self, sample):
        self.t += 1

        sample._idx = Categorical(probs=sample._i_prob).sample() - 1

        if sample._idx.item() >= 0:
            sample.gamma[sample._idx] = ~sample.gamma[sample._idx]
            sample._active = torch.nonzero(sample.gamma).squeeze(-1)
            sample._activeb = torch.cat([sample._active, torch.tensor([self.P])])
            sample = self.sample_beta(sample)
        else:
            sample = self.sample_omega_nb(sample) if self.negbin else self.sample_omega_binomial(sample)

        sample = self._compute_probs(sample)
        sample.weight = sample._i_prob.mean().reciprocal()

        return sample

    def sample_beta(self, sample):
        activeb = sample._activeb
        Xb_active = self.Xb[:, activeb]
        precision = Xb_active.t() @ (sample._omega.unsqueeze(-1) * Xb_active)
        precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)
        self.L_active = safe_cholesky(precision)

        sample.beta.zero_()
        sample.beta_mean.zero_()

        beta_active = chosolve(sample._Z[activeb].unsqueeze(-1), self.L_active).squeeze(-1)
        sample.beta_mean[activeb] = beta_active
        sample.beta[activeb] = beta_active + \
            trisolve(torch.randn(activeb.size(-1), 1), self.L_active, upper=False)[0].squeeze(-1)

        sample._psi = torch.mv(Xb_active, beta_active)
        return sample

    def sample_omega_binomial(self, sample):
        t0 = time.time()

        omega_prop = random_polyagamma(self.T_np, sample._psi.data.cpu().numpy(), random_state=self.rng)
        omega_prop = torch.from_numpy(omega_prop).type_as(self.Xb)

        activeb = sample._activeb
        Xb_active = self.Xb[:, activeb]

        # some of these computations could be reused/saved but they are cheap
        # so we do them from scratch to avoid unnecessary complexity
        def compute_log_target(omega):
            precision = Xb_active.t() @ (omega.unsqueeze(-1) * Xb_active)
            precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)

            L = safe_cholesky(precision)
            LZ = trisolve(sample._Z[activeb].unsqueeze(-1), L, upper=False)[0].squeeze(-1)
            logdet = L.diag().log().sum() - L.size(-1) * self.half_log_tau

            return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet, L

        log_target_prop, L_prop = compute_log_target(omega_prop)
        beta_mean_prop = chosolve(sample._Z[activeb].unsqueeze(-1), L_prop).squeeze(-1)
        beta_prop = beta_mean_prop + \
            trisolve(torch.randn(activeb.size(-1), 1), L_prop, upper=False)[0].squeeze(-1)
        psi_prop = torch.mv(Xb_active, beta_mean_prop)

        if self.omega_mh:
            log_target_curr, _ = compute_log_target(sample._omega)
            delta_psi = psi_prop - sample._psi

            accept1 = log_target_prop - log_target_curr
            accept2 = dot(sample._kappa - self.Y64, delta_psi)
            accept3 = 0.5 * (dot(omega_prop, sample._psi.pow(2.0)) - dot(sample._omega, psi_prop.pow(2.0)))
            accept4 = dot(self.T64, softplus(psi_prop) - softplus(sample._psi))
            accept = min(1.0, (accept1 + accept2 + accept3 + accept4).exp().item())

            if self.t >= self.T_burnin:
                self.acc_probs.append(accept)
            accept = self.uniform_dist.sample().item() < accept

            if self.t >= self.T_burnin:
                self.attempted_omega_updates += 1
                self.accepted_omega_updates += int(accept)

        elif self.t >= self.T_burnin:  # always accept mh move
            self.accepted_omega_updates += 1
            self.attempted_omega_updates += 1

        if not self.omega_mh or accept or (self.t < self.T_burnin // 2):
            sample._omega = omega_prop
            sample._psi = psi_prop
            sample.beta_mean.zero_()
            sample.beta_mean[activeb] = beta_mean_prop
            sample.beta.zero_()
            sample.beta[activeb] = beta_prop
            self.L_active = L_prop

        return sample

    def sample_omega_nb(self, sample):
        t0 = time.time()

        activeb = sample._activeb
        Xb_active = self.Xb[:, activeb]

        log_nu_prop = sample.log_nu + self.log_nu_rw_scale * torch.randn(1).item()
        nu_curr, nu_prop = sample.log_nu.exp(), log_nu_prop.exp()
        T_curr, T_prop = self.Y + nu_curr, self.Y + nu_prop

        psi0_prop = self.psi0 - log_nu_prop
        psi_mixed = sample._psi + psi0_prop
        omega_prop = random_polyagamma(T_prop.data.cpu().numpy(), psi_mixed.data.cpu().numpy(),
                                       random_state=self.rng)
        omega_prop = torch.from_numpy(omega_prop).type_as(self.Xb)

        kappa_prop = 0.5 * (self.Y - nu_prop)
        kappa_omega_prop = kappa_prop - omega_prop * psi0_prop
        Z_prop = einsum("np,n->p", self.Xb, kappa_omega_prop)

        def compute_log_target(omega, Z):
            precision = Xb_active.t() @ (omega.unsqueeze(-1) * Xb_active)
            precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)

            L = safe_cholesky(precision)
            LZ = trisolve(Z[activeb].unsqueeze(-1), L, upper=False)[0].squeeze(-1)
            logdet = L.diag().log().sum() - L.size(-1) * self.half_log_tau

            return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet, L

        log_target_prop, L_prop = compute_log_target(omega_prop, Z_prop)
        beta_mean_prop = chosolve(Z_prop[activeb].unsqueeze(-1), L_prop).squeeze(-1)
        beta_prop = beta_mean_prop + \
            trisolve(torch.randn(activeb.size(-1), 1), L_prop, upper=False)[0].squeeze(-1)

        psi_prop = torch.mv(Xb_active, beta_mean_prop)
        log_target_curr, _ = compute_log_target(sample._omega, sample._Z)

        accept1 = log_target_prop - log_target_curr \
                  + (torch.lgamma(T_prop) - torch.lgamma(nu_prop)).sum() \
                  - (torch.lgamma(T_curr) - torch.lgamma(nu_curr)).sum() \
                  + (kappa_prop * psi0_prop).sum() - (sample._kappa * sample._psi0).sum() \
                  + 0.5 * ((sample._omega * sample._psi0.pow(2.0)).sum() -
                           (omega_prop * psi0_prop.pow(2.0)).sum())

        psi_mixed_prop = psi_prop + sample._psi0
        accept2 = dot(sample._kappa, psi_mixed_prop) - dot(kappa_prop, psi_mixed) \
                  + 0.5 * (dot(omega_prop, psi_mixed.pow(2.0)) -
                           dot(sample._omega, psi_mixed_prop.pow(2.0)))

        accept3 = dot(self.Y64, psi_mixed - psi_mixed_prop) \
                  - dot(T_prop, softplus(psi_mixed)) + dot(T_curr, softplus(psi_mixed_prop))
        accept = min(1.0, (accept1 + accept2 + accept3).exp().item())

        if self.t >= self.T_burnin:
            self.acc_probs.append(accept)

        accept = self.uniform_dist.sample().item() < accept
        self.attempted_omega_updates += 1
        self.accepted_omega_updates += int(accept)

        if accept or self.t < self.T_burnin // 5:
            sample.log_nu = log_nu_prop
            sample._omega = omega_prop
            sample._psi = psi_prop
            self.L_active = L_prop
            sample._kappa = kappa_prop
            sample._psi0 = psi0_prop
            sample._kappa_omega = kappa_omega_prop
            sample._Z = Z_prop
            sample.beta_mean.zero_()
            sample.beta_mean[activeb] = beta_mean_prop
            sample.beta.zero_()
            sample.beta[activeb] = beta_prop

        return sample
