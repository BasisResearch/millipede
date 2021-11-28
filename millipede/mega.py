import math
from types import SimpleNamespace

import torch
from torch import einsum, matmul, sigmoid
from torch import triangular_solve as trisolve
from torch.distributions import Bernoulli, Uniform, Gumbel
from torch.linalg import norm

from .sampler import MCMCSampler
from .util import get_loo_inverses, leave_one_out, safe_cholesky, set_subtract


class MegaSampler(MCMCSampler):
    def __init__(self, X, Y, S=5,
                 spotlight=16,
                 include_intercept=True,
                 tau=0.01, tau_intercept=1.0e-4, c=100.0,
                 nu0=0.0, lambda0=0.0,
                 precompute_XX=False,
                 compute_betas=False, verbose_constructor=True):

        self.spotlight = spotlight
        self.N, self.P = X.shape
        assert (self.N,) == Y.shape

        assert X.dtype == Y.dtype
        assert X.device == Y.device
        self.device = X.device
        self.dtype = X.dtype

        self.X = X
        self.Y = Y
        self.tau = tau
        self.tau_intercept = tau_intercept if include_intercept else tau

        if include_intercept:
            self.X = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)
            self.Pb = self.P + 1
        else:
            self.Pb = self.P

        if S >= self.P or S <= 0:
            raise ValueError("S must satisfy 0 < S < P")
        if self.tau <= 0.0:
            raise ValueError("tau must satisfy tau > 0.0")
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
        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        self.N_nu0 = self.N + nu0

        self.compute_betas = compute_betas
        self.include_intercept = include_intercept

        self.epsilon = 1.0e-18

        self.epsilon_asi = 0.1 / self.P
        self.zeta = torch.tensor(0.95)
        self.acc_target = 0.25
        self.lambda_exponent = 0.75
        self.log_h = math.log(self.h)
        self.log_1mh = math.log1p(-self.h)

        self.update_zeta(0.0)

        self.uniform_dist = Uniform(0.0, torch.ones(1, device=X.device, dtype=X.dtype))
        self.num_accepted = 0

        if verbose_constructor:
            print("Initialized ASISampler with (N, P, S) = ({}, {}, {:.1f})".format(self.N, self.P, S))

    def logit_eps(self, x):
        x = (x - self.epsilon_asi) / (1.0 - x - self.epsilon_asi)
        return torch.log(x)

    def sigmoid_eps(self, y):
        return (1.0 - self.epsilon_asi) * sigmoid(y) + self.epsilon_asi * sigmoid(-y)

    def update_zeta(self, delta):
        self.zeta = self.sigmoid_eps(self.logit_eps(self.zeta) + delta)

    def initialize_sample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        sample = SimpleNamespace(gamma=torch.zeros(self.P, device=self.device).bool(),
                                 _inactive_search=torch.randperm(self.P, device=self.device)[:self.spotlight],
                                 _gumbel=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64),
                                 add_prob=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 weight=torch.tensor(1.0, device=self.device, dtype=self.dtype))

        if self.compute_betas:
            sample.beta = torch.zeros(self.P, device=self.device, dtype=self.dtype)

        if self.include_intercept:
            sample._activeb = torch.tensor([self.P], device=self.device, dtype=torch.int64)

        return sample

    def mcmc_move(self, sample):
        self.t += 1

        search = torch.cat([sample._active, sample._inactive_search])
        pi_search = self.epsilon_asi + (1.0 - 2.0 * self.epsilon_asi) * sigmoid(sample._gumbel[search])
        one = torch.ones(1, device=pi_search.device, dtype=pi_search.dtype)
        A_search = self.zeta * torch.min(one, pi_search / (1.0 - pi_search))
        D_search = self.zeta * torch.min(one, (1.0 - pi_search) / pi_search)

        gamma_curr = sample.gamma[search].double()
        q_curr = Bernoulli(gamma_curr * D_search + (1.0 - gamma_curr) * A_search)
        flips = q_curr.sample()
        flips_bool = flips.bool()

        gamma_prop = gamma_curr.clone().bool()
        gamma_prop[flips_bool] = ~gamma_prop[flips_bool]

        q_prop = Bernoulli(gamma_prop.double() * D_search + (1.0 - gamma_prop.double()) * A_search)

        logq_prop = q_curr.log_prob(flips).sum().item()
        logq_curr = q_prop.log_prob(flips).sum().item()

        gamma_prop_full = sample.gamma.clone()
        gamma_prop_full[search] = gamma_prop

        log_target_curr, _, _, LLZ_curr = self.compute_log_target(sample=sample)
        log_target_prop, active_prop, activeb_prop, LLZ_prop = self.compute_log_target(gamma=gamma_prop_full)

        sample_prop = SimpleNamespace(gamma=gamma_prop_full,
                                      _inactive_search=sample._inactive_search,
                                      _gumbel=sample._gumbel,
                                      _active=active_prop,
                                      add_prob=torch.zeros(self.P, device=self.device, dtype=self.dtype))
        if self.include_intercept:
            sample_prop._activeb = activeb_prop

        sample.add_prob = self._compute_add_prob(sample)
        sample_prop.add_prob = self._compute_add_prob(sample_prop)
        lam = sample.add_prob[search]

        delta_log_gumbel =

        accept = log_target_prop - log_target_curr + logq_curr - logq_prop
        accept = min(1.0, accept.exp().item())
        #print("%.4f" % accept, sample._active.data.numpy(), " => ", active_prop.data.numpy())

        accept_bool = self.uniform_dist.sample().item() < accept

        if accept_bool:
            self.num_accepted += 1
            sample.gamma = gamma_prop_full
            sample._active = active_prop
            if self.include_intercept:
                sample._activeb = activeb_prop
            L, LZ = LLZ_prop
        else:
            L, LZ = LLZ_curr

        if self.compute_betas and self.t >= self.T_burnin:
            beta_active = trisolve(LZ.unsqueeze(-1), L.t(), upper=True)[0].squeeze(-1)
            sample.beta = self.X.new_zeros(self.Pb)
            if self.include_intercept:
                sample.beta[sample._activeb] = beta_active
            else:
                sample.beta[sample._active] = beta_active

        inactive_search = torch.randperm(self.P, device=self.device)[:self.spotlight]
        sample._inactive_search = set_subtract(inactive_search, sample._active)

        if self.t <= self.T_burnin:
            t = self.t
            phi_t = 1.0 / math.pow(t, self.lambda_exponent)
            self.update_zeta(phi_t * (accept - self.acc_target))

        return sample

    def _compute_add_prob(self, sample, return_log_odds=False):
        active = sample._active
        activeb = sample._activeb if self.include_intercept else sample._active
        inactive = torch.nonzero(~sample.gamma).squeeze(-1)
        inactive_search = sample._inactive_search
        num_active = active.size(-1)

        assert num_active < self.P, "The MCMC sampler has been driven into a regime where " +\
            "all covariates have been selected. Are you sure you have chosen a reasonable prior? " +\
            "Are you sure there is signal in your data?"

        X_k = self.X[:, inactive_search]
        Z_k = self.Z[inactive_search]
        if self.XX is None:
            XX_k = norm(X_k, dim=0).pow(2.0)
        else:
            XX_k = self.XX_diag[inactive_search]

        if self.include_intercept or num_active > 0:
            X_activeb = self.X[:, activeb]
            Z_active = self.Z[activeb]
            if self.XX is not None:
                XX_active = self.XX[activeb][:, activeb]
            else:
                XX_active = X_activeb.t() @ X_activeb
            XX_active.diagonal(dim1=-2, dim2=-1).add_(self.tau)
            if self.include_intercept:
                XX_active[-1, -1].add_(self.tau_intercept - self.tau)

            L_active = safe_cholesky(XX_active)

            Zt_active = trisolve(Z_active.unsqueeze(-1), L_active, upper=False)[0].squeeze(-1)
            Xt_active = trisolve(X_activeb.t(), L_active, upper=False)[0].t()
            XtZt_active = einsum("np,p->n", Xt_active, Zt_active)
            # XtZt_active = torch.mv(X_activeb, torch.mv(F, Z_active))

            if self.XX is None:
                G_k_inv = XX_k + self.tau - norm(einsum("ni,nk->ik", Xt_active, X_k), dim=0).pow(2.0)
            else:
                normsq = trisolve(self.XX[activeb][:, inactive_search], L_active, upper=False)[0]
                G_k_inv = XX_k + self.tau - norm(normsq, dim=0).pow(2.0)

            W_k_sq = (einsum("np,n->p", X_k, XtZt_active) - Z_k).pow(2.0) / (G_k_inv + self.epsilon)
            Zt_active_sq = Zt_active.pow(2.0).sum()

            log_det_inactive = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)
        else:
            W_k_sq = Z_k.pow(2.0) / (XX_k + self.tau + self.epsilon)
            Zt_active_sq = 0.0
            log_det_inactive = -0.5 * torch.log1p(XX_k / self.tau)

        if self.compute_betas and (num_active > 0 or self.include_intercept):
            beta_active = trisolve(Zt_active.unsqueeze(-1), L_active.t(), upper=True)[0].squeeze(-1)
            sample.beta = self.X.new_zeros(self.Pb)
            sample.beta[activeb] = beta_active
        elif self.compute_betas and num_active == 0:
            sample.beta = self.X.new_zeros(self.Pb)

        if num_active > 1:
            active_loo = leave_one_out(active)  # I  I-1

            if self.include_intercept:
                active_loob = torch.cat([active_loo,
                                         (self.P * active_loo.new_ones(active_loo.size(0))).long().unsqueeze(-1)],
                                        dim=-1)
            else:
                active_loob = active_loo

            X_active_loo = self.X[:, active_loob].permute(1, 2, 0)  # I I-1 N
            Z_active_loo = self.Z[active_loob]

            F = torch.cholesky_inverse(L_active, upper=False)
            F_loo = get_loo_inverses(F)
            if self.include_intercept:
                F_loo = F_loo[:-1]

            Zt_active_loo = matmul(F_loo, Z_active_loo.unsqueeze(-1)).squeeze(-1)
            Zt_active_loo_sq = einsum("ij,ij->i", Zt_active_loo, Z_active_loo)

            X_active = self.X[:, active]
            X_I_X_k = matmul(X_active_loo, X_active.t().unsqueeze(-1))
            F_X_I_X_k = matmul(F_loo, X_I_X_k).squeeze(-1)
            XXFXX = einsum("ij,ij->i", X_I_X_k.squeeze(-1), F_X_I_X_k)
            XX_active_diag = XX_active.diag() if not self.include_intercept else XX_active.diag()[:-1]
            G_k_inv = XX_active_diag - XXFXX
            log_det_active = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)

        elif num_active == 1:
            if not self.include_intercept:
                Zt_active_loo_sq = 0.0
                log_det_active = -0.5 * torch.log1p(norm(self.X[:, active], dim=0).pow(2.0) / self.tau)
            else:
                Zt_active_loo_sq = norm(self.Z[self.P]).pow(2.0) / (self.tau_intercept + float(self.N))
                G_inv = norm(self.X[:, active], dim=0).pow(2.0) + self.tau \
                    - self.X[:, active].sum().pow(2.0) / (self.tau_intercept + float(self.N))
                log_det_active = -0.5 * G_inv.log() + 0.5 * math.log(self.tau)
        elif num_active == 0:
            Zt_active_loo_sq = 0.0
            log_det_active = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        log_S_ratio = -torch.log1p(- W_k_sq / (self.YY - Zt_active_sq))
        log_odds_inactive = self.log_h_ratio + log_det_inactive + 0.5 * self.N_nu0 * log_S_ratio

        log_S_ratio = (self.YY - Zt_active_loo_sq).log() - (self.YY - Zt_active_sq).log()
        log_odds_active = self.log_h_ratio + log_det_active + 0.5 * self.N_nu0 * log_S_ratio

        sample._gumbel[active] = Gumbel(loc=log_odds_active, scale=1.0).sample()
        sample._gumbel[inactive_search] = Gumbel(loc=log_odds_inactive, scale=1.0).sample()

        pip = sample.gamma.double().clone()
        pip[active] = sigmoid(log_odds_active)
        pip[inactive_search] = sigmoid(log_odds_inactive)

        #pips = pip.data.numpy().tolist()[:3]
        #pips += sample.gamma.double().tolist()[:3]
        #print("%.3f %.3f %.3f   %.1f %.1f %.1f" % tuple(pips))

        return pip

    def compute_log_target(self, sample=None, gamma=None):
        gamma = gamma if gamma is not None else sample.gamma

        active = torch.nonzero(gamma).squeeze(-1)
        activeb = active
        if self.include_intercept:
            activeb = torch.cat([active, torch.tensor([self.P])])

        Xb_active = self.X[:, activeb]
        num_active = activeb.size(-1)
        num_inactive = self.P - num_active

        precision = Xb_active.t() @ Xb_active
        precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)
        if self.include_intercept:
            precision[-1, -1].add_(self.tau_intercept - self.tau)

        L = safe_cholesky(precision)
        LZ = trisolve(self.Z[activeb].unsqueeze(-1), L, upper=False)[0].squeeze(-1)
        logdet = -L.diag().log().sum()
        h_factor = num_active * (self.log_h + 0.5 * math.log(self.tau)) + num_inactive * self.log_1mh
        log_factor = logdet - 0.5 * self.N * (self.YY - norm(LZ).pow(2.0)).log() + h_factor

        return log_factor, active, activeb, (L, LZ)
