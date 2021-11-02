import math
from types import SimpleNamespace

import torch
from torch import einsum, matmul, sigmoid
from torch import triangular_solve as trisolve
from torch.distributions import Categorical
from torch.linalg import norm

from .sampler import MCMCSampler
from .util import leave_one_out, safe_cholesky


class NormalLikelihoodSampler(MCMCSampler):
    """
    MCMC sampler for Bayesian variable selection for a linear model with a Normal likelihood.
    The likelihood variance is controlled by a Inverse Gamma prior.

    Usage of this class is only recommended for advanced users. For most users it should
    suffice to use `NormalLikelihoodVariableSelector`.
    """
    def __init__(self, X, Y, S=5, c=100.0, explore=5, precompute_XX=False,
                 prior="isotropic", tau=0.01, tau_bias=1.0e-4, compute_betas=False,
                 nu0=0.0, lambda0=0.0, include_bias=True):
        assert prior in ['isotropic', 'gprior']

        self.N, self.P = X.shape
        assert (self.N,) == Y.shape

        assert X.dtype == Y.dtype
        assert X.device == Y.device
        self.device = X.device
        self.dtype = X.dtype

        self.prior = prior
        self.X = X
        self.Y = Y
        self.c = c if prior == 'gprior' else 0.0
        self.tau = tau if prior == 'isotropic' else 0.0

        if prior == 'isotropic':
            self.tau_bias = tau_bias if include_bias else tau
        else:
            self.tau_bias = 0.0

        if include_bias:
            self.X = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)
            self.Pb = self.P + 1
        else:
            self.Pb = self.P

        if S >= self.P or S <= 0:
            raise ValueError("S must satisfy 0 < S < P")
        if prior == 'gprior' and self.c <= 0.0:
            raise ValueError("c must satisfy c > 0.0")
        if prior == 'isotropic' and self.tau <= 0.0:
            raise ValueError("tau must satisfy tau > 0.0")
        if explore <= 0.0:
            raise ValueError("explore must satisfy explore > 0.0")
        if nu0 < 0.0:
            raise ValueError("nu0 must satisfy nu0 >= 0.0")
        if lambda0 < 0.0:
            raise ValueError("lambda0 must satisfy lambda0 >= 0.0")

        self.YY = Y.pow(2.0).sum() + nu0 * lambda0
        self.Z = einsum("np,n->p", self.X, Y)

        if precompute_XX:
            self.XX = self.X.t() @ self.X
            self.XX_diag = self.XX.diagonal()
        else:
            self.XX = None

        self.h = S / self.P
        self.explore = explore / self.P
        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        self.N_nu0 = self.N + nu0

        self.compute_betas = compute_betas
        self.include_bias = include_bias

        self.c_one_c = self.c / (1.0 + self.c)
        self.log_one_c_sqrt = 0.5 * math.log(1.0 + self.c)
        self.hc_prefactor = self.log_h_ratio - self.log_one_c_sqrt
        self.epsilon = 1.0e-18

        if self.prior == 'isotropic':
            s = "Initialized NormalLikelihoodSampler with isotropic prior and (N, P, S, tau)" +\
                " = ({}, {}, {:.1f}, {:.3f})"
            print(s.format(self.N, self.P, S, self.tau))
        else:
            s = "Initialized NormalLikelihoodSampler with gprior and (N, P, S, c)" +\
                " = ({}, {}, {:.1f}, {:.1f})"
            print(s.format(self.N, self.P, S, self.c))

    def initialize_sample(self):
        sample = SimpleNamespace(gamma=torch.zeros(self.P, device=self.device).bool(),
                                 add_prob=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 _i_prob=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64),
                                 _idx=0, weight=0)
        if self.compute_betas:
            sample.beta = torch.zeros(self.P, device=self.device, dtype=self.dtype)

        if self.include_bias:
            sample._activeb = torch.tensor([self.P], device=self.device, dtype=torch.int64)

        sample = self._compute_probs(sample)
        return sample

    def _compute_add_prob(self, sample, return_log_odds=False):
        active = sample._active
        activeb = sample._activeb if self.include_bias else sample._active
        inactive = torch.nonzero(~sample.gamma).squeeze(-1)
        num_active = active.size(-1)

        assert num_active < self.P, "The MCMC sampler has been driven into a regime where " +\
            "all covariates have been selected. Are you sure you have chosen a reasonable prior? " +\
            "Are you sure there is signal in your data?"

        X_k = self.X[:, inactive]
        Z_k = self.Z[inactive]
        if self.XX is None:
            XX_k = norm(X_k, dim=0).pow(2.0)
        else:
            XX_k = self.XX_diag[inactive]

        if self.include_bias or num_active > 0:
            X_activeb = self.X[:, activeb]
            Z_active = self.Z[activeb]
            if self.XX is not None:
                XX_active = self.XX[activeb][:, activeb]
            else:
                XX_active = X_activeb.t() @ X_activeb
            if self.prior == 'isotropic':
                XX_active.diagonal(dim1=-2, dim2=-1).add_(self.tau)
                if self.include_bias:
                    XX_active[-1, -1].add_(self.tau_bias - self.tau)

            L_active = safe_cholesky(XX_active)

            Zt_active = trisolve(Z_active.unsqueeze(-1), L_active, upper=False)[0].squeeze(-1)
            Xt_active = trisolve(X_activeb.t(), L_active, upper=False)[0].t()
            XtZt_active = einsum("np,p->n", Xt_active, Zt_active)

            if self.XX is None:
                G_k_inv = XX_k + self.tau - norm(einsum("ni,nk->ik", Xt_active, X_k), dim=0).pow(2.0)
            else:
                normsq = trisolve(self.XX[activeb][:, inactive], L_active, upper=False)[0]
                G_k_inv = XX_k + self.tau - norm(normsq, dim=0).pow(2.0)

            W_k_sq = (einsum("np,n->p", X_k, XtZt_active) - Z_k).pow(2.0) / (G_k_inv + self.epsilon)
            Zt_active_sq = Zt_active.pow(2.0).sum()

            if self.prior == 'isotropic':
                log_det_inactive = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)
        else:
            W_k_sq = Z_k.pow(2.0) / (XX_k + self.tau + self.epsilon)
            Zt_active_sq = 0.0
            if self.prior == 'isotropic':
                log_det_inactive = -0.5 * torch.log1p(XX_k / self.tau)

        if self.compute_betas and (num_active > 0 or self.include_bias):
            beta_active = trisolve(Zt_active.unsqueeze(-1), L_active.t(), upper=True)[0].squeeze(-1)
            sample.beta = self.X.new_zeros(self.Pb)
            if self.prior == 'gprior':
                sample.beta[activeb] = self.c_one_c * beta_active
            else:
                sample.beta[activeb] = beta_active
        elif self.compute_betas and num_active == 0:
            sample.beta = self.X.new_zeros(self.Pb)

        if num_active > 1:
            active_loo = leave_one_out(active)  # I  I-1

            if self.include_bias:
                active_loob = torch.cat([active_loo,
                                         (self.P * active_loo.new_ones(active_loo.size(0))).long().unsqueeze(-1)],
                                        dim=-1)
            else:
                active_loob = active_loo

            X_active_loo = self.X[:, active_loob].permute(1, 2, 0)  # I I-1 N
            XX_active_loo = matmul(X_active_loo, X_active_loo.transpose(-1, -2))  # I I-1 I-1
            if self.prior == 'isotropic':
                XX_active_loo.diagonal(dim1=-2, dim2=-1).add_(self.tau)
                if self.include_bias:
                    XX_active_loo[:, -1, -1].add_(self.tau_bias - self.tau)

            Z_active_loo = self.Z[active_loob]
            L_XX_active_loo = safe_cholesky(XX_active_loo)
            Zt_active_loo = trisolve(Z_active_loo.unsqueeze(-1), L_XX_active_loo, upper=False)[0].squeeze(-1)
            Zt_active_loo_sq = Zt_active_loo.pow(2.0).sum(-1)
            if self.prior == 'isotropic':
                log_det_active = L_XX_active_loo.diagonal(dim1=-1, dim2=-2).log().sum(-1) -\
                    L_active.diagonal(dim1=-1, dim2=-2).log().sum(-1) + 0.5 * math.log(self.tau)

        elif num_active == 1:
            if not self.include_bias:
                Zt_active_loo_sq = 0.0
                if self.prior == 'isotropic':
                    log_det_active = -0.5 * torch.log1p(norm(self.X[:, active], dim=0).pow(2.0) / self.tau)
            else:
                Zt_active_loo_sq = norm(self.Z[self.P]).pow(2.0) / (self.tau_bias + float(self.N))
                if self.prior == 'isotropic':
                    G_inv = norm(self.X[:, active], dim=0).pow(2.0) + self.tau \
                        - self.X[:, active].sum().pow(2.0) / (self.tau_bias + float(self.N))
                    log_det_active = -0.5 * G_inv.log() + 0.5 * math.log(self.tau)
        elif num_active == 0:
            Zt_active_loo_sq = 0.0
            log_det_active = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.prior == 'gprior':
            log_S_ratio = -torch.log1p(-self.c_one_c * W_k_sq / (self.YY - self.c_one_c * Zt_active_sq))
            log_odds_inactive = self.hc_prefactor + 0.5 * self.N_nu0 * log_S_ratio

            log_S_ratio = torch.log(self.YY - self.c_one_c * Zt_active_loo_sq) -\
                torch.log(self.YY - self.c_one_c * Zt_active_sq)
            log_odds_active = self.hc_prefactor + 0.5 * self.N_nu0 * log_S_ratio
        elif self.prior == 'isotropic':
            log_S_ratio = -torch.log1p(- W_k_sq / (self.YY - Zt_active_sq))
            log_odds_inactive = self.log_h_ratio + log_det_inactive + 0.5 * self.N_nu0 * log_S_ratio

            log_S_ratio = (self.YY - Zt_active_loo_sq).log() - (self.YY - Zt_active_sq).log()
            log_odds_active = self.log_h_ratio + log_det_active + 0.5 * self.N_nu0 * log_S_ratio

        log_odds = self.X.new_zeros(self.P)
        log_odds[inactive] = log_odds_inactive
        log_odds[active] = log_odds_active

        return log_odds

    def _compute_probs(self, sample):
        sample.add_prob = sigmoid(self._compute_add_prob(sample))
        gamma = sample.gamma.type_as(sample.add_prob)
        prob_gamma_i = gamma * sample.add_prob + (1.0 - gamma) * (1.0 - sample.add_prob)
        sample._i_prob = (sample.add_prob + self.explore) / (prob_gamma_i + self.epsilon)
        return sample

    def gibbs_move(self, sample):
        self.t += 1
        sample._idx = Categorical(probs=sample._i_prob).sample()
        sample.gamma[sample._idx] = ~sample.gamma[sample._idx]

        sample._active = torch.nonzero(sample.gamma).squeeze(-1)
        sample._activeb = torch.cat([sample._active, torch.tensor([self.P], device=self.device)]) \
            if self.include_bias else sample._active

        sample = self._compute_probs(sample)
        sample.weight = sample._i_prob.mean().reciprocal()
        return sample
