import math
from types import SimpleNamespace

import torch
from torch import sigmoid
from torch.distributions import Bernoulli, Uniform
from torch.linalg import norm
from torch.linalg import solve_triangular as trisolve

from .normal import NormalLikelihoodSampler
from .util import safe_cholesky


class ASISampler(NormalLikelihoodSampler):
    def __init__(self, X, Y, S=5.0,
                 prior='isotropic',
                 include_intercept=True,
                 tau=0.01, tau_intercept=1.0e-4,
                 nu0=0.0, lambda0=0.0,
                 precompute_XX=False,
                 always_adaptive=True,
                 compute_betas=False, verbose_constructor=True):

        assert prior == "isotropic"
        assert isinstance(S, float)

        super().__init__(X=X, Y=Y, S=S, prior=prior, include_intercept=include_intercept,
                         tau=tau, tau_intercept=tau_intercept, c=0.0,
                         nu0=nu0, lambda0=lambda0, explore=5.0, precompute_XX=precompute_XX,
                         compute_betas=compute_betas, verbose_constructor=False, subset_size=None)

        self.epsilon_asi = 0.1 / self.P
        self.zeta = torch.tensor(0.90, device=self.device, dtype=self.dtype)
        self.pi = self.h * torch.ones(self.P, device=self.device, dtype=self.dtype)
        self.acc_target = 0.25
        self.lambda_exponent = 0.75
        self.log_h = math.log(self.h)
        self.log_1mh = math.log1p(-self.h)
        self.always_adaptive = always_adaptive

        self.update_zeta_AD(0.0)

        self.uniform_dist = Uniform(0.0, torch.ones(1, device=X.device, dtype=X.dtype))

        if verbose_constructor:
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
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64),
                                 pip=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 _log_h_ratio=self.log_h_ratio,
                                 weight=torch.tensor(1.0, device=self.device, dtype=self.dtype))

        if self.compute_betas:
            sample.beta = torch.zeros(self.P, device=self.device, dtype=self.dtype)

        if self.include_intercept:
            sample._activeb = torch.tensor([self.P], device=self.device, dtype=torch.int64)

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

        log_target_curr, _, _, LLZ_curr = self.compute_log_target(sample=sample)
        log_target_prop, active_prop, activeb_prop, LLZ_prop = self.compute_log_target(gamma=gamma_prop)

        accept = log_target_prop - log_target_curr + logq_curr - logq_prop
        accept = min(1.0, accept.exp().item())

        accept_bool = self.uniform_dist.sample().item() < accept

        if accept_bool:
            sample.gamma = gamma_prop
            sample._active = active_prop
            if self.include_intercept:
                sample._activeb = activeb_prop
            L, LZ = LLZ_prop
        else:
            L, LZ = LLZ_curr

        # BETA STD WRONG BECAUSE NO RANDN
        if self.compute_betas and self.t >= self.T_burnin:
            beta_active = trisolve(L.t(), LZ.unsqueeze(-1), upper=True).squeeze(-1)
            sample.beta = self.X.new_zeros(self.P + self.Pa)
            if self.include_intercept:
                sample.beta[sample._activeb] = beta_active
            else:
                sample.beta[sample._active] = beta_active

        sample.pip = sigmoid(self._compute_add_prob(sample))

        if self.t <= self.T_burnin or self.always_adaptive:
            t = self.t
            self.pi = (t / (t + 1)) * self.pi + sample.pip / (t + 1)
            phi_t = 1.0 / math.pow(t, self.lambda_exponent)
            self.update_zeta_AD(phi_t * (accept - self.acc_target))

        return sample

    def compute_log_target(self, sample=None, gamma=None):
        gamma = gamma if gamma is not None else sample.gamma

        active = torch.nonzero(gamma).squeeze(-1)
        activeb = active
        if self.include_intercept:
            activeb = torch.cat([active, torch.tensor([self.P], device=self.device)])

        Xb_active = self.X[:, activeb]
        num_active = activeb.size(-1)
        num_inactive = self.P - num_active

        precision = Xb_active.t() @ Xb_active
        precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)
        if self.include_intercept:
            precision[-1, -1].add_(self.tau_intercept - self.tau)

        L = safe_cholesky(precision)
        LZ = trisolve(L, self.Z[activeb].unsqueeze(-1), upper=False).squeeze(-1)
        logdet = -L.diag().log().sum()
        h_factor = num_active * (self.log_h + 0.5 * math.log(self.tau)) + num_inactive * self.log_1mh
        log_factor = logdet - 0.5 * self.N * (self.YY - norm(LZ).pow(2.0)).log() + h_factor

        return log_factor, active, activeb, (L, LZ)
