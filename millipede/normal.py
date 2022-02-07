import math
from types import SimpleNamespace

import torch
from torch import einsum, matmul, sigmoid
from torch import triangular_solve as trisolve
from torch.distributions import Categorical
from torch.linalg import norm

from .sampler import MCMCSampler
from .util import get_loo_inverses, leave_one_out, safe_cholesky


class NormalLikelihoodSampler(MCMCSampler):
    r"""
    MCMC sampler for Bayesian variable selection for a linear model with a Normal likelihood.
    The likelihood variance is controlled by a Inverse Gamma prior.
    This class supports continuous-valued responses.

    The details of the available models in `NormalLikelihoodSampler` are as follows.
    The covariates :math:`X` and responses :math:`Y` are defined as follows:

    .. math::

        X \in \mathbb{R}^{N \times P} \qquad \qquad Y \in \mathbb{R}^{N}

    The inclusion of each covariate is governed by a Bernoulli random variable :math:`\gamma_p`.
    In particular :math:`\gamma_p = 0` corresponds to exclusion and :math:`\gamma_p = 1` corresponds to inclusion.
    The prior probability of inclusion is governed by :math:`h` or alternatively :math:`S`:

    .. math::

        h \in [0, 1] \qquad \rm{with} \qquad S \equiv hP

    Putting this together, the model specification for an isotopric prior (with an intercept
    :math:`\beta_0` included) is as follows:

    .. math::

        &\gamma_p \sim \rm{Bernoulli}(h) \qquad \rm{for} \qquad p=1,2,...,P

        &\sigma^2 \sim \rm{InverseGamma}(\nu_0 / 2, \nu_0 \lambda_0 / 2)

        &\beta_0 \sim \rm{Normal}(0, \sigma^2\tau_\rm{intercept}^{-1})

        &\beta_\gamma \sim \rm{Normal}(0, \sigma^2 \tau^{-1} \mathbb{1}_\gamma)

        &Y_n \sim \rm{Normal}(\beta_0 + X_{n, \gamma} \cdot \beta_\gamma, \sigma^2)

    Note that the dimension of :math:`\beta_\gamma` depends on the number of covariates
    included in a particular model (i.e. on the number of non-zero entries in :math:`\gamma`).

    For a gprior the prior over the coefficients is instead specified as follows:

    .. math::

        \beta_{\gamma} \sim \rm{Normal}(0, c \sigma^2 (X_\gamma^{\rm{T}} X_\gamma)^{-1})

    Usage of this class is only recommended for advanced users. For most users it should
    suffice to use :class:`NormalLikelihoodVariableSelector`.

    :param tensor X: A N x P `torch.Tensor` of covariates.
    :param tensor Y: A N-dimensional `torch.Tensor` of continuous responses.
    :param float S: The number of covariates to include in the model a priori. Defaults to 5.
    :param str prior: One of the two supported priors for the coefficients: 'isotropic' or 'gprior'.
        Defaults to 'isotropic'.
    :param bool include_intercept: Whether to include an intercept term. If included the intercept term is
       is included in all models so that the corresponding coefficient does not have a PIP.
    :param float tau: Controls the precision of the coefficients in the isotropic prior. Defaults to 0.01.
    :param float tau_intercept: Controls the precision of the intercept in the isotropic prior. Defaults to 1.0e-4.
    :param float c: Controls the precision of the coefficients in the gprior. Defaults to 100.0.
    :param float nu0: Controls the prior over the precision in the Normal likelihood. Defaults to 0.0.
    :param float lambda0: Controls the prior over the precision in the Normal likelihood. Defaults to 0.0.
    :param float explore: This hyperparameter controls how greedy the MCMC algorithm is. Defaults to 5.0.
    :param bool precompute_XX: Whether the matrix X^t @ X should be pre-computed. Defaults to False. Note
        that setting this to True may result in out-of-memory errors for sufficiently large covariate matrices.
    :param bool verbose_constructor: Whether the class constructor should print some information to
        stdout upon initialization.
    """
    def __init__(self, X, Y, S=5,
                 prior="isotropic", include_intercept=True,
                 tau=0.01, tau_intercept=1.0e-4, c=100.0,
                 nu0=0.0, lambda0=0.0,
                 explore=5, precompute_XX=False,
                 compute_betas=False, verbose_constructor=True,
                 xi_target=0.2):
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
            self.tau_intercept = tau_intercept if include_intercept else tau
        else:
            self.tau_intercept = 0.0

        if include_intercept:
            self.X = torch.cat([X, X.new_ones(X.size(0), 1)], dim=-1)
            self.Pb = self.P + 1
        else:
            self.Pb = self.P

        if not isinstance(S, tuple):
            if S >= self.P or S <= 0:
                raise ValueError("S must satisfy 0 < S < P or must be a tuple.")
        else:
            if len(S) != 2 or not isinstance(S[0], float) or not isinstance(S[1], float) or S[0] <= 0.0 or S[1] <= 0.0:
                raise ValueError("If S is a tuple it must be a tuple of two positive floats (alpha, beta).")
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

        if not isinstance(S, tuple):
            self.h = S / self.P
            self.xi = torch.tensor([0.0], device=X.device)
        else:
            self.S_alpha, self.S_beta = S
            self.h = self.S_alpha / (self.S_alpha + self.S_beta)
            self.xi = torch.tensor([5.0], device=X.device)
            self.xi_target = xi_target

        self.c_one_c = self.c / (1.0 + self.c)
        self.log_one_c_sqrt = 0.5 * math.log(1.0 + self.c)
        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)

        self.explore = explore / self.P
        self.N_nu0 = self.N + nu0

        self.compute_betas = compute_betas
        self.include_intercept = include_intercept
        self.epsilon = 1.0e-18

        if verbose_constructor:
            s2 = " = ({}, {}, {:.1f}, {:.3f})" if not isinstance(S, tuple) else " = ({}, {}, ({:.1f}, {:.1f}), {:.3f})"
            S = S if isinstance(S, tuple) else (S,)
            if self.prior == 'isotropic':
                s1 = "Initialized NormalLikelihoodSampler with isotropic prior and (N, P, S, tau)"
                print((s1 + s2).format(self.N, self.P, *S, self.tau))
            else:
                s1 = "Initialized NormalLikelihoodSampler with gprior and (N, P, S, c)"
                print((s1 + s2).format(self.N, self.P, *S, self.c))

    def initialize_sample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        sample = SimpleNamespace(gamma=torch.zeros(self.P, device=self.device).bool(),
                                 add_prob=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 _i_prob=torch.zeros(self.P, device=self.device, dtype=self.dtype),
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64),
                                 _idx=0, _log_h_ratio=self.log_h_ratio, weight=0)
        if self.compute_betas:
            sample.beta = torch.zeros(self.P, device=self.device, dtype=self.dtype)

        if self.include_intercept:
            sample._activeb = torch.tensor([self.P], device=self.device, dtype=torch.int64)

        sample = self._compute_probs(sample)
        return sample

    def _compute_add_prob(self, sample, return_log_odds=False):
        active = sample._active
        activeb = sample._activeb if self.include_intercept else sample._active
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

        if self.include_intercept or num_active > 0:
            X_activeb = self.X[:, activeb]
            Z_active = self.Z[activeb]
            if self.XX is not None:
                XX_active = self.XX[activeb][:, activeb]
            else:
                XX_active = X_activeb.t() @ X_activeb
            if self.prior == 'isotropic':
                XX_active.diagonal(dim1=-2, dim2=-1).add_(self.tau)
                if self.include_intercept:
                    XX_active[-1, -1].add_(self.tau_intercept - self.tau)

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

        if self.compute_betas and (num_active > 0 or self.include_intercept):
            beta_active = trisolve(Zt_active.unsqueeze(-1), L_active.t(), upper=True)[0].squeeze(-1)
            sample.beta = self.X.new_zeros(self.Pb)
            if self.prior == 'gprior':
                sample.beta[activeb] = self.c_one_c * beta_active
            else:
                sample.beta[activeb] = beta_active
        elif self.compute_betas and num_active == 0:
            sample.beta = self.X.new_zeros(self.Pb)

        if num_active > 1:
            active_loo = leave_one_out(active)  # I I-1

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

            if self.prior == 'isotropic':
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
                if self.prior == 'isotropic':
                    log_det_active = -0.5 * torch.log1p(norm(self.X[:, active], dim=0).pow(2.0) / self.tau)
            else:
                Zt_active_loo_sq = norm(self.Z[self.P]).pow(2.0) / (self.tau_intercept + float(self.N))
                if self.prior == 'isotropic':
                    G_inv = norm(self.X[:, active], dim=0).pow(2.0) + self.tau \
                        - self.X[:, active].sum().pow(2.0) / (self.tau_intercept + float(self.N))
                    log_det_active = -0.5 * G_inv.log() + 0.5 * math.log(self.tau)
        elif num_active == 0:
            Zt_active_loo_sq = 0.0
            log_det_active = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.prior == 'gprior':
            log_S_ratio = -torch.log1p(-self.c_one_c * W_k_sq / (self.YY - self.c_one_c * Zt_active_sq))
            log_odds_inactive = sample._log_h_ratio + self.log_one_c_sqrt + 0.5 * self.N_nu0 * log_S_ratio

            log_S_ratio = torch.log(self.YY - self.c_one_c * Zt_active_loo_sq) -\
                torch.log(self.YY - self.c_one_c * Zt_active_sq)
            log_odds_active = sample._log_h_ratio + self.log_one_c_sqrt + 0.5 * self.N_nu0 * log_S_ratio
        elif self.prior == 'isotropic':
            log_S_ratio = -torch.log1p(- W_k_sq / (self.YY - Zt_active_sq))
            log_odds_inactive = sample._log_h_ratio + log_det_inactive + 0.5 * self.N_nu0 * log_S_ratio

            log_S_ratio = (self.YY - Zt_active_loo_sq).log() - (self.YY - Zt_active_sq).log()
            log_odds_active = sample._log_h_ratio + log_det_active + 0.5 * self.N_nu0 * log_S_ratio

        log_odds = self.X.new_zeros(self.P)
        log_odds[inactive] = log_odds_inactive
        log_odds[active] = log_odds_active

        return log_odds

    def _compute_probs(self, sample):
        sample.add_prob = sigmoid(self._compute_add_prob(sample))

        gamma = sample.gamma.type_as(sample.add_prob)
        prob_gamma_i = gamma * sample.add_prob + (1.0 - gamma) * (1.0 - sample.add_prob)
        i_prob = 0.5 * (sample.add_prob + self.explore) / (prob_gamma_i + self.epsilon)

        if hasattr(self, 'S_alpha') and self.t <= self.T_burnin:  # adapt xi
            self.xi += (self.xi_target - self.xi / (self.xi + i_prob.sum())) / math.sqrt(self.t + 1)

        sample._i_prob = torch.cat([self.xi, i_prob])

        return sample

    def mcmc_move(self, sample):
        self.t += 1

        sample._idx = Categorical(probs=sample._i_prob).sample() - 1

        if sample._idx.item() >= 0:
            sample.gamma[sample._idx] = ~sample.gamma[sample._idx]

            sample._active = torch.nonzero(sample.gamma).squeeze(-1)
            sample._activeb = torch.cat([sample._active, torch.tensor([self.P], device=self.device)]) \
                if self.include_intercept else sample._active
        else:
            sample = self.sample_alpha_beta(sample)

        sample = self._compute_probs(sample)
        sample.weight = sample._i_prob.mean().reciprocal()

        return sample

    def sample_alpha_beta(self, sample):
        return sample
