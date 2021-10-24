import math
from types import SimpleNamespace

import torch
from torch import einsum, matmul, sigmoid
from torch import triangular_solve as trisolve
from torch.distributions import Bernoulli, Categorical
from torch.linalg import norm

from .sampler import MCMCSampler
from .util import leave_one_out, safe_cholesky


class NormalLikelihoodSampler(MCMCSampler):
    def __init__(self, X, Y, S=5, c=100.0, explore=5, precompute_XX=False,
                 algo="wtgs", prior="isotropic", tau=0.5, compute_betas=False,
                 nu0=0.0, lambda0=0.0):
        super().__init__()
        assert algo in ['wtgs', 'wgs', 'block']
        assert prior in ['isotropic', 'gprior']

        self.prior = prior

        self.X = X
        self.Y = Y
        self.tau = tau if prior == 'isotropic' else 0.0
        self.c = c if prior == 'gprior' else 0.0

        self.nu0 = nu0
        self.lambda0 = lambda0
        self.nulam0 = nu0 * lambda0

        if precompute_XX:
            self.XX = X.t() @ X
            self.XX_diag = self.XX.diagonal()
        else:
            self.XX = None

        self.YY = Y.pow(2.0).sum() + self.nulam0
        self.Z = einsum("np,n->p", X, Y)

        self.N, self.P = X.shape
        assert self.N == Y.size(-1)

        if S >= self.P or S <= 0:
            raise ValueError("S must satisfy 0 < S < P")

        self.algo = algo
        self.h = S / self.P
        self.explore = explore / self.P
        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        self.N_nu0 = self.N + self.nu0

        self.compute_betas = compute_betas
        self.c_one_c = self.c / (1.0 + self.c)
        self.log_one_c_sqrt = 0.5 * math.log(1.0 + self.c)
        self.hc_prefactor = self.log_h_ratio - self.log_one_c_sqrt
        self.epsilon = 1.0e-18

        s = "Initialized {}-BasicSampler with {} prior and (N, P, S, c, tau) = ({}, {}, {:.1f}, {:.1f}, {:.3f})"
        print(s.format(self.algo, self.prior, self.N, self.P, S, self.c, self.tau))

    def initialize_sample(self):
        if self.compute_betas:
            sample = SimpleNamespace(gamma=torch.zeros(self.P).bool(),
                                     add_prob=torch.zeros(self.P),
                                     i_prob=torch.zeros(self.P),
                                     beta=torch.zeros(self.P),
                                     idx=0, weight=0)
        else:
            sample = SimpleNamespace(gamma=torch.zeros(self.P).bool(),
                                     add_prob=torch.zeros(self.P),
                                     i_prob=torch.zeros(self.P),
                                     idx=0, weight=0)

        sample = self._compute_probs(sample)
        return sample

    def _compute_add_prob(self, sample, return_log_odds=False):
        active = torch.nonzero(sample.gamma).squeeze(-1)
        inactive = torch.nonzero(~sample.gamma).squeeze(-1)
        num_active = active.size(-1)
        assert num_active < self.P and num_active <= self.N

        X_k = self.X[:, inactive]
        Z_k = self.Z[inactive]
        if self.XX is None:
            XX_k = norm(X_k, dim=0).pow(2.0)
        else:
            XX_k = self.XX_diag[inactive]

        if num_active > 0:
            X_active = self.X[:, active]
            Z_active = self.Z[active]
            if self.XX is not None:
                XX_active = self.XX[active][:, active]
            else:
                XX_active = X_active.t() @ X_active
            if self.prior == 'isotropic':
                XX_active.diagonal(dim1=-2, dim2=-1).add_(self.tau)

            L_active = safe_cholesky(XX_active)

            Zt_active = trisolve(Z_active.unsqueeze(-1), L_active, upper=False)[0].squeeze(-1)
            Xt_active = trisolve(X_active.t(), L_active, upper=False)[0].t()
            XtZt_active = einsum("np,p->n", Xt_active, Zt_active)

            if self.XX is None:
                normsq = norm(einsum("ni,nk->ik", Xt_active, X_k), dim=0).pow(2.0)
            else:
                normsq = trisolve(self.XX[active][:, inactive], L_active, upper=False)[0]
                normsq = norm(normsq, dim=0).pow(2.0)

            G_k_inv = XX_k + self.tau - normsq
            # G_k_inv = XX_k + self.tau - norm(einsum("ni,nk->ik", Xt_active, X_k), dim=0).pow(2.0)
            W_k_sq = (einsum("np,n->p", X_k, XtZt_active) - Z_k).pow(2.0) / (G_k_inv + self.epsilon)

            Zt_active_sq = Zt_active.pow(2.0).sum()

            if self.prior == 'isotropic':
                log_det_inactive = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)
        else:
            W_k_sq = Z_k.pow(2.0) / (XX_k + self.tau + self.epsilon)
            Zt_active_sq = 0.0
            if self.prior == 'isotropic':
                log_det_inactive = -0.5 * torch.log1p(XX_k / self.tau)

        if self.compute_betas and num_active > 0:
            beta_active = trisolve(Zt_active.unsqueeze(-1), L_active.t(), upper=True)[0].squeeze(-1)
            sample.beta = self.X.new_zeros(self.P)
            if self.prior == 'gprior':
                sample.beta[active] = self.c_one_c * beta_active
            else:
                sample.beta[active] = beta_active
        elif self.compute_betas and num_active == 0:
            sample.beta = self.X.new_zeros(self.P)

        if num_active > 1:
            active_loo = leave_one_out(active)  # I  I-1
            X_active_loo = self.X[:, active_loo].permute(1, 2, 0)  # I I-1 N
            XX_active_loo = matmul(X_active_loo, X_active_loo.transpose(-1, -2))  # I I-1 I-1
            if self.prior == 'isotropic':
                XX_active_loo.diagonal(dim1=-2, dim2=-1).add_(self.tau)

            Z_active_loo = self.Z[active_loo]
            L_XX_active_loo = safe_cholesky(XX_active_loo)
            Zt_active_loo = trisolve(Z_active_loo.unsqueeze(-1), L_XX_active_loo, upper=False)[0].squeeze(-1)
            Zt_active_loo_sq = Zt_active_loo.pow(2.0).sum(-1)
            if self.prior == 'isotropic':
                log_det_active = L_XX_active_loo.diagonal(dim1=-1, dim2=-2).log().sum(-1) -\
                    L_active.diagonal(dim1=-1, dim2=-2).log().sum(-1) + 0.5 * math.log(self.tau)

        elif num_active == 1:
            Zt_active_loo_sq = 0.0
            log_det_active = -0.5 * torch.log1p(norm(X_active, dim=0).pow(2.0) / self.tau)
        elif num_active == 0:
            Zt_active_loo_sq = 0.0
            log_det_active = torch.tensor(0.0)

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

        if return_log_odds:
            return log_odds

        add_prob = sigmoid(log_odds)
        return add_prob

    def _compute_probs(self, sample):
        sample.add_prob = self._compute_add_prob(sample)

        gamma = sample.gamma.double()
        prob_gamma_i = gamma * sample.add_prob + (1.0 - gamma) * (1.0 - sample.add_prob)

        if self.algo == 'wtgs':
            sample.i_prob = (sample.add_prob + self.explore) / (prob_gamma_i + self.epsilon)
        elif self.algo == 'wgs':
            sample.i_prob = sample.add_prob + self.explore
        return sample

    def gibbs_move(self, sample):
        self.t += 1

        sample.idx = Categorical(probs=sample.i_prob).sample()

        if self.algo == 'wtgs':
            sample.gamma[sample.idx] = ~sample.gamma[sample.idx]
        elif self.algo == 'wgs':
            sample.gamma[sample.idx] = Bernoulli(probs=sample.add_prob[sample.idx]).sample()

        sample = self._compute_probs(sample)
        sample.weight = sample.i_prob.mean().reciprocal()

        return sample
