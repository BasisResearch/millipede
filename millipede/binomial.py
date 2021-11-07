import math
from types import SimpleNamespace

import numpy as np
import torch
from polyagamma import random_polyagamma
from torch import cholesky_solve as chosolve
from torch import dot, einsum, matmul, sigmoid
from torch import triangular_solve as trisolve
from torch.distributions import Categorical, Uniform
from torch.linalg import norm
from torch.nn.functional import softplus

from .sampler import MCMCSampler
from .util import leave_one_out, safe_cholesky


class CountLikelihoodSampler(MCMCSampler):
    def __init__(self, X, Y, TC=None, S=5, explore=5.0, tau=0.01, tau_intercept=1.0e-4,
                 log_nu_rw_scale=0.05, omega_mh=True, psi0=None, init_nu=5.0, xi_target=0.25):
        super().__init__()
        if not ((TC is None and psi0 is not None) or (TC is not None and psi0 is None)):
            raise ValueError('CountLikelihoodSampler supports two modes of operation. ' +
                             'In order to specify a binomial likelihood the user must provide TC but ~not~ ' +
                             'provide psi0. For a negative binomial likelihood the user must provide psi0 ' +
                             'but ~not~ provide TC.')

        self.dtype = X.dtype
        self.device = X.device

        self.Xb = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)
        self.Y = Y
        self.Y_float = self.Y.type_as(X)
        self.tau = tau
        self.N, self.P = X.shape

        self.negbin = psi0 is not None
        if self.negbin:
            psi0 = X.new_tensor(psi0) if isinstance(psi0, float) else psi0
            if not (psi0.shape == Y.shape or psi0.shape == ()):
                raise ValueError("psi0 should either be a scalar or a one-dimensional array with " +
                                 "the same number of elements as Y.")
            if init_nu <= 0.0:
                raise ValueError("init_nu must be positive.")
            self.init_nu = init_nu
            self.psi0 = psi0
            self.log_nu_rw_scale = log_nu_rw_scale
        else:
            if not Y.shape == TC.shape or Y.ndim != 1:
                raise ValueError("Y and TC should both be one-dimensional arrays.")
            self.TC = TC
            self.TC_np = TC.data.cpu().numpy().copy()
            self.TC_float = self.TC.type_as(X)

        if self.N != Y.size(-1):
            raise ValueError("X and Y should be of shape (N, P) and (N,), respectively.")
        if S >= self.P or S <= 0:
            raise ValueError("S must satisfy 0 < S < P")
        if tau <= 0.0:
            raise ValueError("tau must be positive.")
        if tau_intercept <= 0.0:
            raise ValueError("tau_intercept must be positive.")
        if explore < 0.0:
            raise ValueError("tau must be non-negative.")
        if log_nu_rw_scale < 0.0 and self.negbin:
            raise ValueError("log_nu_rw_scale must be non-negative.")
        if xi_target <= 0.0 or xi_target >= 1.0:
            raise ValueError("xi_target must be in the interval (0, 1).")

        self.h = S / self.P
        self.explore = explore / self.P
        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        self.half_log_tau = 0.5 * math.log(tau)
        self.tau_intercept = tau_intercept

        self.epsilon = 1.0e-18
        self.xi = torch.tensor([5.0])
        self.xi_target = xi_target

        self.omega_mh = omega_mh
        self.uniform_dist = Uniform(0.0, X.new_ones(1)[0])

        s = "Initialized CountLikelihoodSampler with {} likelihood and (N, P, S, epsilon) = ({}, {}, {:.1f}, {:.1f})"
        print(s.format("Negative Binomial" if self.negbin else "Binomial", self.N, self.P, S, explore))

    def initialize_sample(self, seed=None):
        self.accepted_omega_updates = 0
        self.attempted_omega_updates = 0
        self.acceptance_probs = []

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        if not self.negbin:
            log_nu = None
            _omega = torch.from_numpy(random_polyagamma(self.TC_np, random_state=self.rng)).type_as(self.Xb)
        else:
            log_nu = torch.tensor(math.log(self.init_nu))
            _omega = torch.from_numpy(random_polyagamma(self.Y.data.cpu().numpy() + self.init_nu,
                                                        random_state=self.rng)).type_as(self.Xb)

        _psi0 = self.psi0 - log_nu if self.negbin else 0.0
        _kappa = 0.5 * (self.Y - log_nu.exp()) if self.negbin else self.Y - 0.5 * self.TC
        _kappa_omega = _kappa - _omega * _psi0
        _Z = einsum("np,n->p", self.Xb, _kappa_omega)

        sample = SimpleNamespace(gamma=torch.zeros(self.P).bool(),
                                 add_prob=self.Xb.new_zeros(self.P),
                                 _i_prob=self.Xb.new_zeros(self.P),
                                 _psi=self.Xb.new_zeros(self.N),
                                 beta_mean=self.Xb.new_zeros(self.P + 1),
                                 beta=self.Xb.new_zeros(self.P + 1),
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

        sample = self.sample_beta(sample)  # populate self._L_active
        sample = self._compute_probs(sample)
        return sample

    def _compute_add_prob(self, sample):
        active, activeb = sample._active, sample._activeb
        inactive = torch.nonzero(~sample.gamma).squeeze(-1)
        num_active = active.size(-1)
        assert num_active < self.P, "The MCMC sampler has been driven into a regime where " +\
            "all covariates have been selected. Are you sure you have chosen a reasonable prior? " +\
            "Are you sure there is signal in your data?"

        X_omega = self.Xb * sample._omega.sqrt().unsqueeze(-1)
        X_omega_k = X_omega[:, inactive]

        Z_k = sample._Z[inactive]
        X_omega_active = X_omega[:, activeb]
        Z_active = sample._Z[activeb]

        Zt_active = trisolve(Z_active.unsqueeze(-1), self._L_active, upper=False)[0].squeeze(-1)
        Xt_active = trisolve(X_omega_active.t(), self._L_active, upper=False)[0].t()
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
            XX_active_loo[:, -1, -1].add_(self.tau_intercept - self.tau)

            Z_active_loo = sample._Z[active_loob]
            L_XX_active_loo = safe_cholesky(XX_active_loo)
            Zt_active_loo_sq = trisolve(Z_active_loo.unsqueeze(-1),
                                        L_XX_active_loo, upper=False)[0].squeeze(-1).pow(2.0).sum(-1)
            log_det_ratio_active = L_XX_active_loo.diagonal(dim1=-1, dim2=-2).log().sum(-1) -\
                self._L_active.diagonal(dim1=-1, dim2=-2).log().sum(-1) + self.half_log_tau
        elif num_active == 1:
            tau_plus_omega = self.tau_intercept + sample._omega.sum()
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

        return log_odds

    def _compute_probs(self, sample):
        sample.add_prob = sigmoid(self._compute_add_prob(sample))

        gamma = sample.gamma.double()
        prob_gamma_i = gamma * sample.add_prob + (1.0 - gamma) * (1.0 - sample.add_prob)
        i_prob = 0.5 * (sample.add_prob + self.explore) / (prob_gamma_i + self.epsilon)

        if self.t <= self.T_burnin:  # adapt xi
            self.xi += (self.xi_target - self.xi / (self.xi + i_prob.sum())) / math.sqrt(self.t + 1)

        sample._i_prob = torch.cat([self.xi, i_prob])

        return sample

    def mcmc_move(self, sample):
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
        precision[-1, -1].add_(self.tau_intercept - self.tau)
        self._L_active = safe_cholesky(precision)

        sample.beta.zero_()
        sample.beta_mean.zero_()

        beta_active = chosolve(sample._Z[activeb].unsqueeze(-1), self._L_active).squeeze(-1)
        sample.beta_mean[activeb] = beta_active
        sample.beta[activeb] = beta_active + \
            trisolve(torch.randn(activeb.size(-1), 1, device=self.device, dtype=self.dtype),
                     self._L_active, upper=False)[0].squeeze(-1)

        sample._psi = torch.mv(Xb_active, beta_active)
        return sample

    def sample_omega_binomial(self, sample, _save_intermediates=None):
        omega_prop = random_polyagamma(self.TC_np, sample._psi.data.cpu().numpy(), random_state=self.rng)
        omega_prop = torch.from_numpy(omega_prop).type_as(self.Xb)

        activeb = sample._activeb
        Xb_active = self.Xb[:, activeb]

        # some of these computations could be reused/saved but they are cheap
        # so we do them from scratch to avoid unnecessary complexity
        def compute_log_target(omega):
            precision = Xb_active.t() @ (omega.unsqueeze(-1) * Xb_active)
            precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)
            precision[-1, -1].add_(self.tau_intercept - self.tau)

            L = safe_cholesky(precision)
            LZ = trisolve(sample._Z[activeb].unsqueeze(-1), L, upper=False)[0].squeeze(-1)
            logdet = L.diag().log().sum() - L.size(-1) * self.half_log_tau

            return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet, L

        log_target_prop, L_prop = compute_log_target(omega_prop)
        beta_mean_prop = chosolve(sample._Z[activeb].unsqueeze(-1), L_prop).squeeze(-1)
        beta_prop = beta_mean_prop + \
            trisolve(torch.randn(activeb.size(-1), 1, device=self.device, dtype=self.dtype),
                     L_prop, upper=False)[0].squeeze(-1)
        psi_prop = torch.mv(Xb_active, beta_mean_prop)

        if self.omega_mh:
            log_target_curr, _ = compute_log_target(sample._omega)
            delta_psi = psi_prop - sample._psi

            accept1 = log_target_prop - log_target_curr
            accept2 = dot(sample._kappa - self.Y_float, delta_psi)
            accept3 = 0.5 * (dot(omega_prop, sample._psi.pow(2.0)) - dot(sample._omega, psi_prop.pow(2.0)))
            accept4 = dot(self.TC_float, softplus(psi_prop) - softplus(sample._psi))
            accept = min(1.0, (accept1 + accept2 + accept3 + accept4).exp().item())

            if _save_intermediates is not None:
                _save_intermediates['omega'] = sample._omega.data.cpu().numpy().copy()
                _save_intermediates['omega_prop'] = omega_prop.data.cpu().numpy().copy()
                _save_intermediates['psi'] = sample._psi.data.cpu().numpy().copy()
                _save_intermediates['psi_prop'] = psi_prop.data.cpu().numpy().copy()
                _save_intermediates['TC_np'] = self.TC_np.copy()
                _save_intermediates['accept234'] = accept2 + accept3 + accept4

            if self.t >= self.T_burnin:
                self.acceptance_probs.append(accept)
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
            self._L_active = L_prop

        return sample

    def sample_omega_nb(self, sample, _save_intermediates=None):
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
            precision[-1, -1].add_(self.tau_intercept - self.tau)

            L = safe_cholesky(precision)
            LZ = trisolve(Z[activeb].unsqueeze(-1), L, upper=False)[0].squeeze(-1)
            logdet = L.diag().log().sum() - L.size(-1) * self.half_log_tau

            return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet, L

        log_target_prop, L_prop = compute_log_target(omega_prop, Z_prop)
        beta_mean_prop = chosolve(Z_prop[activeb].unsqueeze(-1), L_prop).squeeze(-1)
        beta_prop = beta_mean_prop + \
            trisolve(torch.randn(activeb.size(-1), 1, device=self.device, dtype=self.dtype),
                     L_prop, upper=False)[0].squeeze(-1)

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

        accept3 = dot(self.Y_float, psi_mixed - psi_mixed_prop) \
                  - dot(T_prop, softplus(psi_mixed)) + dot(T_curr, softplus(psi_mixed_prop))
        accept = min(1.0, (accept1 + accept2 + accept3).exp().item())

        if _save_intermediates is not None:
            _save_intermediates['omega'] = sample._omega.data.cpu().numpy()
            _save_intermediates['omega_prop'] = omega_prop.data.cpu().numpy()
            _save_intermediates['psi_mixed'] = psi_mixed.data.cpu().numpy()
            _save_intermediates['psi_mixed_prop'] = psi_mixed_prop.data.cpu().numpy()
            _save_intermediates['T_curr'] = T_curr.data.cpu().numpy()
            _save_intermediates['T_prop'] = T_prop.data.cpu().numpy()
            _save_intermediates['delta_nu'] = nu_curr.item() - nu_prop.item()
            _save_intermediates['accept23'] = accept2 + accept3

        if self.t >= self.T_burnin:
            self.acceptance_probs.append(accept)
        accept = self.uniform_dist.sample().item() < accept

        if self.t >= self.T_burnin:
            self.attempted_omega_updates += 1
            self.accepted_omega_updates += int(accept)

        if accept or self.t < min(50, self.T_burnin // 4):
            sample.log_nu = log_nu_prop
            sample._omega = omega_prop
            sample._psi = psi_prop
            self._L_active = L_prop
            sample._kappa = kappa_prop
            sample._psi0 = psi0_prop
            sample._kappa_omega = kappa_omega_prop
            sample._Z = Z_prop
            sample.beta_mean.zero_()
            sample.beta_mean[activeb] = beta_mean_prop
            sample.beta.zero_()
            sample.beta[activeb] = beta_prop

        return sample
