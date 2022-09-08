import math
from types import SimpleNamespace

import torch
from torch import einsum, matmul, sigmoid
from torch.distributions import Beta, Categorical, Gamma
from torch.linalg import norm
from torch.linalg import solve_triangular as trisolve

from .sampler import MCMCSampler
from .util import (
    arange_complement,
    get_loo_inverses,
    leave_one_out,
    leave_one_out_off_diagonal,
    safe_cholesky,
    sample_active_subset,
)


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

    Alternatively, if :math:`h` is not known a priori we can put a prior on :math:`h`:

    .. math::

        h \sim {\rm Beta}(\alpha, \beta) \qquad \rm{with} \qquad \alpha > 0 \;\;\;\; \beta > 0

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
    :param tensor X_assumed: A N x P' `torch.Tensor` of covariates that are always assumed to be part of the model.
        Defaults to `None`.
    :param tensor sigma_scale_factor: A N-dimensional `torch.Tensor` of positive scale factors that are used
        to scale the standard deviation of the Normal likelihood for each datapoint. For example, specifying
        2.0 for a particular datapoint results in :math:`\sigma \rightarrow 2 \times \sigma`. Defaults to `None`.
    :param S: Controls the expected number of covariates to include in the model a priori. Defaults to 5.0.
        To specify covariate-level prior inclusion probabilities provide a P-dimensional `torch.Tensor` of
        the form `(h_1, ..., h_P)`.
        If a tuple of positive floats `(alpha, beta)` is provided, the a priori inclusion probability is a latent
        variable governed by the corresponding Beta prior so that the sparsity level is inferred from the data.
        Note that for a given choice of `alpha` and `beta` the expected number of covariates to include in the model
        a priori is given by :math:`\frac{\alpha}{\alpha + \beta} \times P`.  Also note that the mean number of
        covariates in the posterior can vary significantly from prior expectations, since the posterior is in
        effect a compromise between the prior and the observed data.
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
    :param float xi_target: This hyperparameter controls how often :math:`h` MCMC updates are made if :math:`h`
        is a latent variable. Defaults to 0.2.
    :param int subset_size: If `subset_size` is not None `subset_size` controls the amount of computational
        resources to use in Subset wTGS. Otherwise if `subset_size` is None vanilla wTGS is used.
        This argument is intended to be used for datasets with a very large number of covariates (e.g.
        tens of thousands or more). A typical value might be ~5-10% of the total number of covariates; smaller values
        result in more MCMC iterations per second but may lead to high variance PIP estimates. Defaults to None.
    :param int anchor_size: If `subset_size` is not None `anchor_size` controls how greedy Subset wTGS is.
        If `anchor_size` is None it defaults to half of `subset_size`. For expert users only. Defaults to None.
    """
    def __init__(self, X, Y, X_assumed=None, sigma_scale_factor=None, S=5.0,
                 prior="isotropic", include_intercept=True,
                 tau=0.01, tau_intercept=1.0e-4, c=100.0,
                 nu0=0.0, lambda0=0.0,
                 explore=5, precompute_XX=False,
                 compute_betas=False, verbose_constructor=True,
                 xi_target=0.2,
                 subset_size=None, anchor_size=None):
        assert prior in ['isotropic', 'gprior']

        self.N, self.P = X.shape
        assert (self.N,) == Y.shape

        assert X.dtype == Y.dtype
        assert X.device == Y.device

        if subset_size is not None and (subset_size <= 1 or subset_size >= self.P):
            raise ValueError("If subset_size is not None must be strictly between 1 and P, the number of covariates.")
        self.subset_size = subset_size

        if anchor_size is not None:
            if subset_size is None:
                raise ValueError("The anchor_size argument should only be used if subset_size is not None.")
            if anchor_size < 1 or anchor_size >= subset_size:
                raise ValueError("anchor_size should be strictly between 0 and subset_size.")

        if X_assumed is not None:
            assert X.dtype == X_assumed.dtype
            assert X.device == X_assumed.device
            if X.size(0) != X_assumed.size(0):
                raise ValueError("X and X_assumed must have the same number of rows.")

        if sigma_scale_factor is not None:
            assert sigma_scale_factor.dtype == X.dtype
            assert sigma_scale_factor.device == X.device
            if sigma_scale_factor.shape != (self.N,):
                raise ValueError("sigma_scale_factor must be a N-dimensional tensor.")
            if sigma_scale_factor.min().item() <= 0.0:
                raise ValueError("All entries in sigma_scale_factor must be positive.")
            if prior != "isotropic":
                raise ValueError("sigma_scale_factor can only be used in conjuction with an isotropic prior.")

        self.device = Y.device
        self.dtype = Y.dtype

        self.prior = prior
        self.X = X
        self.Y = Y
        self.c = c if prior == 'gprior' else 0.0
        self.tau = tau if prior == 'isotropic' else 0.0

        if prior == 'isotropic':
            self.tau_intercept = tau_intercept if include_intercept else tau
        else:
            self.tau_intercept = 0.0

        if X_assumed is not None:
            assert X_assumed.size(-1) > 0
            self.X = torch.cat([self.X, X_assumed], dim=-1)

        if include_intercept:
            self.X = torch.cat([self.X, X.new_ones(X.size(0), 1)], dim=-1)

        S = S if not isinstance(S, int) else float(S)
        if isinstance(S, float):
            if S >= self.P or S <= 0:
                raise ValueError("S must satisfy 0 < S < P or must be a tuple or tensor.")
        elif isinstance(S, tuple):
            if len(S) != 2 or not isinstance(S[0], float) or not isinstance(S[1], float) or S[0] <= 0.0 or S[1] <= 0.0:
                raise ValueError("If S is a tuple it must be a tuple of two positive floats (alpha, beta).")
        elif isinstance(S, torch.Tensor):
            if S.shape != (self.P,) or (S >= 1.0).any().item() or (S <= 0.0).any().item():
                raise ValueError("If S is a tensor it must be P-dimensional and all elements must be strictly" +
                                 " contained in (0, 1).")
        else:
            raise ValueError("S must be a float, tuple or tensor.")

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
        if xi_target <= 0.0 or xi_target >= 1.0:
            raise ValueError("xi_target must be in the interval (0, 1).")

        if sigma_scale_factor is not None:
            self.X = self.X.clone() / sigma_scale_factor.unsqueeze(-1)
            Y_scaled = Y / sigma_scale_factor
            self.YY = Y_scaled.pow(2.0).sum() + nu0 * lambda0
            self.Z = einsum("np,n->p", self.X, Y_scaled)
        else:
            self.YY = Y.pow(2.0).sum() + nu0 * lambda0
            self.Z = einsum("np,n->p", self.X, Y)

        if isinstance(S, float):
            self.h = S / self.P
            self.xi = torch.tensor([0.0], device=self.device)
            self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        elif isinstance(S, tuple):
            self.h_alpha, self.h_beta = S
            self.h = self.h_alpha / (self.h_alpha + self.h_beta)
            self.xi = torch.tensor([5.0], device=self.device)
            self.xi_target = xi_target
            self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        else:
            self.h = S
            self.xi = torch.tensor([0.0], device=self.device)
            self.log_h_ratio = S.log() - torch.log1p(-S)

        if prior == "gprior":
            self.c_one_c = self.c / (1.0 + self.c)
            self.c_one_c_sqrt = math.sqrt(self.c_one_c)
            self.log_one_c_sqrt = 0.5 * math.log(1.0 + self.c)

        if self.subset_size is not None:
            self.anchor_size = subset_size // 2 if anchor_size is None else anchor_size
            self.pi = X.new_ones(self.P) * self.h if isinstance(S, (float, tuple)) else self.h
            self.total_weight = 0.0
            self.comb_factor = (self.subset_size - self.anchor_size) / (self.P - self.anchor_size)
        else:
            self.comb_factor = 1.0

        self.explore = explore / self.P
        self.N_nu0 = self.N + nu0

        self.compute_betas = compute_betas
        self.include_intercept = include_intercept

        if include_intercept and X_assumed is None:
            self.assumed_covariates = torch.tensor([self.P], device=self.device, dtype=torch.int64)
        elif include_intercept and X_assumed is not None:
            self.assumed_covariates = torch.arange(self.P, self.P + X_assumed.size(-1) + 1,
                                                   device=self.device, dtype=torch.int64)
        elif not include_intercept and X_assumed is not None:
            self.assumed_covariates = torch.arange(self.P, self.P + X_assumed.size(-1),
                                                   device=self.device, dtype=torch.int64)
        else:
            self.assumed_covariates = None

        if precompute_XX:
            self.XX = self.X.t() @ self.X
            self.XX_diag = self.XX.diagonal()
        else:
            self.XX = None

        self.Pa = 0 if self.assumed_covariates is None else self.assumed_covariates.size(-1)
        self.epsilon = 1.0e3 * torch.finfo(Y.dtype).tiny

        if verbose_constructor:
            s2 = " = ({}, {}, {:.1f}, {:.3f}, {})" if not isinstance(S, tuple) \
                else " = ({}, {}, ({:.1f}, {:.1f}), {:.3f}, {})"
            if isinstance(S, float):
                S = (S,)
            elif isinstance(S, torch.Tensor):
                S = (S.min().item(), S.max().item())
            if self.prior == 'isotropic':
                s1 = "Initialized NormalLikelihoodSampler with isotropic prior and (N, P, S, tau, subset_size)"
                print((s1 + s2).format(self.N, self.P, *S, self.tau, self.subset_size))
            else:
                s1 = "Initialized NormalLikelihoodSampler with gprior and (N, P, S, c, subset_size)"
                print((s1 + s2).format(self.N, self.P, *S, self.c, self.subset_size))

    def initialize_sample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        sample = SimpleNamespace(gamma=torch.zeros(self.P, device=self.device).bool(),
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64),
                                 _log_h_ratio=self.log_h_ratio)

        if self.Pa > 0:
            sample._activeb = self.assumed_covariates

        if self.subset_size is not None:
            Z_cent = einsum("np,n->p", self.X[:, :self.P], self.Y - self.Y.mean())
            self._update_anchor(Z_cent.abs().argsort()[-self.anchor_size:])
            sample._idx = torch.randint(self.P, (), device=self.device)
            sample._active_subset = sample_active_subset(self.P, self.subset_size, self.anchor_subset,
                                                         self.anchor_subset_set, self.anchor_complement, sample._idx)

        if hasattr(self, "h_alpha"):
            sample.h_alpha = torch.tensor(self.h_alpha, device=self.device)
            sample.h_beta = torch.tensor(self.h_beta, device=self.device)

        sample = self._compute_probs(sample)
        return sample

    def _update_anchor(self, anchor):
        self.anchor_subset = anchor
        self.anchor_subset_set = set(anchor.data.cpu().numpy().tolist())
        self.anchor_complement = arange_complement(self.P, anchor)

    def _compute_add_prob(self, sample, return_log_odds=False):
        active = sample._active
        activeb = sample._activeb if self.Pa > 0 else sample._active

        if self.subset_size is not None:
            inactive = torch.zeros(self.P, device=self.device, dtype=torch.bool)
            inactive[sample._active_subset] = ~(sample.gamma[sample._active_subset])
            inactive = torch.nonzero(inactive).squeeze(-1)
        else:
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

        if self.Pa > 0 or num_active > 0:
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

            Zt_active = trisolve(L_active, Z_active.unsqueeze(-1), upper=False).squeeze(-1)
            Xt_active = trisolve(L_active, X_activeb.t(), upper=False).t()
            XtZt_active = einsum("np,p->n", Xt_active, Zt_active)

            if self.XX is None:
                G_k_inv = XX_k + self.tau - norm(einsum("ni,nk->ik", Xt_active, X_k), dim=0).pow(2.0)
            else:
                normsq = trisolve(L_active, self.XX[activeb][:, inactive], upper=False)
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

        if self.compute_betas and (num_active > 0 or self.Pa > 0):
            beta_active = trisolve(L_active.t(), Zt_active.unsqueeze(-1), upper=True).squeeze(-1)
            sample.beta = self.Y.new_zeros(self.P + self.Pa)
            epsilon = torch.randn(activeb.size(-1), 1, device=self.device, dtype=self.dtype)
            if self.prior == 'gprior':
                sigma_beta = 0.5 * (self.YY - self.c_one_c * Zt_active_sq)
                sample.sigma = Gamma(0.5 * self.N_nu0, sigma_beta).sample().sqrt().reciprocal()
                sample.beta[activeb] = self.c_one_c * beta_active
                sample.beta[activeb] += self.c_one_c_sqrt * sample.sigma * \
                    trisolve(L_active, epsilon, upper=False).squeeze(-1)
            else:
                sigma_beta = 0.5 * (self.YY - Zt_active_sq)
                sample.sigma = Gamma(0.5 * self.N_nu0, sigma_beta).sample().sqrt().reciprocal()
                sample.beta[activeb] = beta_active
                sample.beta[activeb] += sample.sigma * trisolve(L_active, epsilon, upper=False).squeeze(-1)
        elif self.compute_betas and num_active == 0:
            sample.beta = self.Y.new_zeros(self.P + self.Pa)

        if num_active > 1:
            active_loo = leave_one_out(active)  # I I-1

            if self.Pa > 0:
                assumed_covariates = self.assumed_covariates.expand(num_active, -1)
                active_loob = torch.cat([active_loo, assumed_covariates], dim=-1)
            else:
                active_loob = active_loo

            Z_active_loo = self.Z[active_loob]

            F = torch.cholesky_inverse(L_active, upper=False)
            F_loo = get_loo_inverses(F)
            if self.Pa > 0:
                F_loo = F_loo[:-self.Pa]

            Zt_active_loo = matmul(F_loo, Z_active_loo.unsqueeze(-1)).squeeze(-1)
            Zt_active_loo_sq = einsum("ij,ij->i", Zt_active_loo, Z_active_loo)

            if self.prior == 'isotropic':
                X_I_X_k = leave_one_out_off_diagonal(XX_active).unsqueeze(-1)
                X_I_X_k = X_I_X_k if self.Pa == 0 else X_I_X_k[:-self.Pa]

                F_X_I_X_k = matmul(F_loo, X_I_X_k).squeeze(-1)
                XXFXX = einsum("ij,ij->i", X_I_X_k.squeeze(-1), F_X_I_X_k)
                XX_active_diag = XX_active.diag() if self.Pa == 0 else XX_active.diag()[:-self.Pa]
                G_k_inv = XX_active_diag - XXFXX
                log_det_active = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)

        elif num_active == 1:
            if self.Pa == 0:
                Zt_active_loo_sq = 0.0
                if self.prior == 'isotropic':
                    log_det_active = -0.5 * torch.log1p(norm(self.X[:, active], dim=0).pow(2.0) / self.tau)
            else:
                if self.XX is None:
                    XX_assumed = self.X[:, self.assumed_covariates]
                    XX_assumed = XX_assumed.t() @ XX_assumed
                else:
                    XX_assumed = self.XX[self.assumed_covariates][:, self.assumed_covariates]
                if self.prior == "isotropic":
                    XX_assumed.diagonal(dim1=-2, dim2=-1).add_(self.tau)
                    if self.include_intercept:
                        XX_assumed[-1, -1].add_(self.tau_intercept - self.tau)
                L_assumed = safe_cholesky(XX_assumed)
                Zt_active_loo = trisolve(L_assumed, self.Z[self.assumed_covariates].unsqueeze(-1),
                                         upper=False).squeeze(-1)
                Zt_active_loo_sq = norm(Zt_active_loo, dim=0).pow(2.0)
                if self.prior == "isotropic":
                    log_det_active = L_assumed.diagonal().log().sum() - L_active.diagonal().log().sum()
                    log_det_active += 0.5 * math.log(self.tau)

        elif num_active == 0:
            Zt_active_loo_sq = 0.0
            log_det_active = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        log_h_ratio_active = sample._log_h_ratio[active] if isinstance(self.h, torch.Tensor) else sample._log_h_ratio
        log_h_ratio_inactive = sample._log_h_ratio[inactive] if isinstance(self.h, torch.Tensor) \
            else sample._log_h_ratio

        if self.prior == 'gprior':
            log_S_ratio = -torch.log1p(-self.c_one_c * W_k_sq / (self.YY - self.c_one_c * Zt_active_sq))
            log_odds_inactive = log_h_ratio_inactive - self.log_one_c_sqrt + 0.5 * self.N_nu0 * log_S_ratio

            log_S_ratio = torch.log(self.YY - self.c_one_c * Zt_active_loo_sq) -\
                torch.log(self.YY - self.c_one_c * Zt_active_sq)
            log_odds_active = log_h_ratio_active - self.log_one_c_sqrt + 0.5 * self.N_nu0 * log_S_ratio
        elif self.prior == 'isotropic':
            log_S_ratio = -torch.log1p(- W_k_sq / (self.YY - Zt_active_sq))
            log_odds_inactive = log_h_ratio_inactive + log_det_inactive + 0.5 * self.N_nu0 * log_S_ratio

            log_S_ratio = (self.YY - Zt_active_loo_sq).log() - (self.YY - Zt_active_sq).log()
            log_odds_active = log_h_ratio_active + log_det_active + 0.5 * self.N_nu0 * log_S_ratio

        log_odds = self.Y.new_full((self.P,), -torch.inf)
        log_odds[inactive] = log_odds_inactive
        log_odds[active] = log_odds_active

        return log_odds

    def _compute_probs(self, sample):
        sample._add_prob = sigmoid(self._compute_add_prob(sample))

        gamma = sample.gamma.type_as(sample._add_prob)
        prob_gamma_i = gamma * sample._add_prob + (1.0 - gamma) * (1.0 - sample._add_prob)
        _i_prob = 0.5 * (sample._add_prob + self.explore) / (prob_gamma_i + self.epsilon)

        if self.subset_size is not None:
            _i_prob[self.anchor_subset] *= self.comb_factor
            i_prob = torch.zeros_like(_i_prob)
            i_prob[sample._active_subset] = _i_prob[sample._active_subset]
            sample.pip = sample.gamma.type_as(i_prob)
            sample.pip[sample._active_subset] = sample._add_prob[sample._active_subset]
        else:
            i_prob = _i_prob
            sample.pip = sample._add_prob

        if hasattr(self, 'h_alpha') and self.t <= self.T_burnin:  # adapt xi
            xi_comb = self.xi * self.comb_factor
            self.xi += (self.xi_target - xi_comb / (xi_comb + i_prob.sum())) / math.sqrt(self.t + 1)
            self.xi.clamp_(min=0.01)

        sample._i_prob = torch.cat([self.xi * self.comb_factor, i_prob])

        return sample

    def mcmc_move(self, sample):
        self.t += 1

        sample._idx = Categorical(probs=sample._i_prob).sample() - 1

        if sample._idx.item() >= 0:
            sample.gamma[sample._idx] = ~sample.gamma[sample._idx]

            sample._active = torch.nonzero(sample.gamma).squeeze(-1)
            if self.Pa > 0:
                sample._activeb = torch.cat([sample._active, self.assumed_covariates])
        else:
            sample = self.sample_alpha_beta(sample)

        if self.subset_size is not None:
            sample._active_subset = sample_active_subset(self.P, self.subset_size, self.anchor_subset,
                                                         self.anchor_subset_set, self.anchor_complement, sample._idx)

        sample = self._compute_probs(sample)
        sample.weight = sample._i_prob.mean().reciprocal()

        if self.subset_size is not None and self.t <= self.T_burnin:
            self.pi = sample.weight * sample.pip + self.total_weight * self.pi
            self.total_weight += sample.weight
            self.pi /= self.total_weight
            if (self.t > 99 and self.t % 100 == 0) or self.t == self.T_burnin:
                self._update_anchor(self.pi.argsort()[-self.anchor_size:])

        return sample

    def sample_alpha_beta(self, sample):
        num_active = sample._active.size(-1)
        num_inactive = self.P - num_active
        sample.h_alpha = torch.tensor(self.h_alpha + num_active, device=self.device)
        sample.h_beta = torch.tensor(self.h_beta + num_inactive, device=self.device)
        h = Beta(sample.h_alpha, sample.h_beta).sample().item()
        sample._log_h_ratio = math.log(h) - math.log(1.0 - h)
        return sample
