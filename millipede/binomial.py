import math
from types import SimpleNamespace

import numpy as np
import torch
from polyagamma import random_polyagamma
from torch import cholesky_solve as chosolve
from torch import dot, einsum, matmul, sigmoid
from torch.distributions import Beta, Categorical, Uniform
from torch.linalg import norm
from torch.linalg import solve_triangular as trisolve
from torch.nn.functional import softplus

from .sampler import MCMCSampler
from .util import (
    arange_complement,
    get_loo_inverses,
    leave_one_out,
    leave_one_out_off_diagonal,
    safe_cholesky,
    sample_active_subset,
)


class CountLikelihoodSampler(MCMCSampler):
    r"""
    MCMC algorithm for Bayesian variable selection for a generalized linear model with a Binomial or
    Negative Binomial likelihood (note that a Bernoulli likelihood is a special case of a Binomial likelihood).
    This class supports count-valued responses.

    To define a Binomial model specify `TC` but not `psi0`.
    To define a Negative Binomial model specify `psi0` but not `TC`.

    The details of the available models in :class:`CountlLikelihoodSampler` are as follows.
    For both likelihoods the covariates and responses are defined as:

    .. math::

        X \in \mathbb{R}^{N \times P} \qquad \qquad Y \in \mathbb{Z}_{\ge 0}^{N}

    The inclusion of each covariate is governed by a Bernoulli random variable :math:`\gamma_p`.
    In particular :math:`\gamma_p = 0` corresponds to exclusion and :math:`\gamma_p = 1` corresponds to inclusion.
    The prior probability of inclusion is governed by :math:`h` or alternatively :math:`S`:

    .. math::

        h \in [0, 1] \qquad \rm{with} \qquad S \equiv hP

    Alternatively, if :math:`h` is not known a priori we can put a prior on :math:`h`:

    .. math::

        h \sim {\rm Beta}(\alpha, \beta) \qquad \rm{with} \qquad \alpha > 0 \;\;\;\; \beta > 0

    Putting this together, the model specification for the Binomial case is as follows:

    .. math::

        &\gamma_p \sim \rm{Bernoulli}(h) \qquad \rm{for} \qquad p=1,2,...,P

        &\beta_0 \sim \rm{Normal}(0, \tau_\rm{intercept}^{-1})

        &\beta_\gamma \sim \rm{Normal}(0, \tau^{-1} \mathbb{1}_\gamma)

        &Y_n \sim \rm{Binomial}(T_n, \sigma(\beta_0 + X_{n, \gamma} \cdot \beta_\gamma))

    where :math:`\sigma(\cdot)` is the logistic or sigmoid function and :math:`T_n` denotes the
    :math:`N`-dimensional vector of total counts. That is each Binomial likelihood is equivalent
    to :math:`T_n` corresponding Bernoulli likelihoods.

    The Negative Binomial case is similar but includes a latent variable :math:`\nu > 0`
    that governs the dispersion of the Negative Binomial distribution:

    .. math::

        &\log \nu \sim \rm{ImproperPrior}(-\infty, \infty)

        &Y_n \sim \rm{NegBinomial}(\rm{mean}=\rm{exp}(\beta_0 + X_{n, \gamma} \cdot \beta_\gamma + \psi_{0, n}), \nu)

    The vector :math:`\psi_0 \in \mathbb{R}^N` allows the user to supply a datapoint-specific offset.
    We note that we use a parameterization of the Negative Binomial distribution where the variance is given by

    .. math::

        \rm{variance} = \rm{mean} + \rm{mean}^2 / \nu

    so that small values of :math:`\nu` correspond to large dispersion/variance and :math:`\nu \to \infty` recovers
    the Poisson distribution.

    Note that above the dimension of :math:`\beta_\gamma` depends on the number of covariates
    included in a particular model (i.e. on the number of non-zero entries in :math:`\gamma`).

    Usage of this class is only recommended for advanced users. For most users it should
    suffice to use one of :class:`BinomialLikelihoodVariableSelector`, :class:`BernoulliLikelihoodVariableSelector`,
    and :class:`NegativeBinomialLikelihoodVariableSelector`.

    :param tensor X: A N x P `torch.Tensor` of covariates. This is a required argument.
    :param tensor Y: A N-dimensional `torch.Tensor` of non-negative count-valued responses. This is a required argument.
    :param tensor X_assumed: A N x P' `torch.Tensor` of covariates that are always assumed to be part of the model.
        Defaults to `None`.
    :param tensor TC: A N-dimensional `torch.Tensor` of non-negative counts. This is a required argument if
        you wish to specify a Binomial model. Defaults to None.
    :param tensor psi0: A N-dimensional `torch.Tensor` of offsets `psi0`. This is a required argument if
        you wish to specify a Negative Binomial model. If the user specifies a float, `psi0` will be expanded
        to a N-dimensional vector internally. Defaults to None.
    :param S: Controls the expected number of covariates to include in the model a priori. Defaults to 5.0.
        To specify covariate-level prior inclusion probabilities provide a P-dimensional `torch.Tensor` of
        the form `(h_1, ..., h_P)`.
        If a tuple of positive floats `(alpha, beta)` is provided, the a priori inclusion probability is a latent
        variable governed by the corresponding Beta prior so that the sparsity level is inferred from the data.
        Note that for a given choice of `alpha` and `beta` the expected number of covariates to include in the model
        a priori is given by :math:`\frac{\alpha}{\alpha + \beta} \times P`.  Also note that the mean number of
        covariates in the posterior can vary significantly from prior expectations, since the posterior is in
        effect a compromise between the prior and the observed data.
    :param float tau: Controls the precision of the coefficients in the isotropic prior. Defaults to 0.01.
    :param float tau_intercept: Controls the precision of the intercept in the isotropic prior. Defaults to 1.0e-4.
    :param float explore: This hyperparameter controls how greedy the MCMC algorithm is. Defaults to 5.0.
    :param float log_nu_rw_scale: This hyperparameter controls the proposal distribution for `nu` updates.
        Defaults to 0.05. Only applicable to the Negative Binomial case.
    :param bool omega_mh: Whether to include Metropolis-Hastings corrections during Polya-Gamma updates. Defaults
        to True. Only applicable to the Binomial case.
    :param float xi_target: This hyperparameter controls how frequently the MCMC algorithm makes Polya-Gamma updates.
        It also controls how often :math:`h` updates are made if :math:`h` is a latent variable. Defaults to 0.25.
    :param float init_nu: This hyperparameter controls the initial value of the dispersion parameter `nu`.
        Defaults to 5.0. Only applicable to the Negative Binomial case.
    :param bool verbose_constructor: Whether the class constructor should print some information to
        stdout upon initialization.
    :param int subset_size: If `subset_size` is not None `subset_size` controls the amount of computational
        resources to use in Subset PG-wTGS. Otherwise if `subset_size` is None vanilla PG-wTGS is used.
        This argument is intended to be used for datasets with a very large number of covariates (e.g.
        tens of thousands or more). A typical value might be ~5-10% of the total number of covariates; smaller values
        result in more MCMC iterations per second but may lead to high variance PIP estimates. Defaults to None.
    :param int anchor_size: If `subset_size` is not None `anchor_size` controls how greedy Subset PG-wTGS is.
        If `anchor_size` is None it defaults to half of `subset_size`. For expert users only. Defaults to None.
    """
    def __init__(self, X, Y, X_assumed=None, TC=None, psi0=None,
                 S=5.0, tau=0.01, tau_intercept=1.0e-4,
                 explore=5.0, log_nu_rw_scale=0.05, omega_mh=True,
                 xi_target=0.25, init_nu=5.0, verbose_constructor=True,
                 subset_size=None, anchor_size=None):
        super().__init__()
        if not ((TC is None and psi0 is not None) or (TC is not None and psi0 is None)):
            raise ValueError('CountLikelihoodSampler supports two modes of operation. ' +
                             'In order to specify a binomial likelihood the user must provide TC but ~not~ ' +
                             'provide psi0. For a negative binomial likelihood the user must provide psi0 ' +
                             'but ~not~ provide TC.')

        self.dtype = X.dtype
        self.device = X.device
        self.Xb = X
        self.N, self.P = X.shape

        if X_assumed is not None:
            assert self.dtype == X_assumed.dtype
            assert self.device == X_assumed.device
            if X.size(0) != X_assumed.size(0):
                raise ValueError("X and X_assumed must have the same number of rows.")
            assert X_assumed.size(-1) > 0
            self.Xb = torch.cat([self.Xb, X_assumed], dim=-1)

        if subset_size is not None and (subset_size <= 1 or subset_size >= self.P):
            raise ValueError("If subset_size is not None must be strictly between 1 and P, the number of covariates.")
        self.subset_size = subset_size

        if anchor_size is not None:
            if subset_size is None:
                raise ValueError("The anchor_size argument should only be used if subset_size is not None.")
            if anchor_size < 1 or anchor_size >= subset_size:
                raise ValueError("anchor_size should be strictly between 0 and subset_size.")

        self.Xb = torch.cat([self.Xb, X.new_ones(X.size(0), 1)], dim=-1)
        self.Y = Y
        self.Y_float = self.Y.type_as(X)
        self.tau = tau

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

        if X_assumed is not None:
            self.assumed_covariates = torch.arange(self.P, self.P + X_assumed.size(-1) + 1, device=self.device)
        else:
            self.assumed_covariates = torch.tensor([self.P], device=self.device, dtype=torch.int64)

        self.Pa = self.assumed_covariates.size(-1)

        if isinstance(S, float):
            self.h = S / self.P
            self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        elif isinstance(S, tuple):
            self.h_alpha, self.h_beta = S
            self.h = self.h_alpha / (self.h_alpha + self.h_beta)
            self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        else:
            self.h = S
            self.log_h_ratio = S.log() - torch.log1p(-S)

        self.explore = explore / self.P
        self.half_log_tau = 0.5 * math.log(tau)
        self.tau_intercept = tau_intercept

        self.epsilon = 1.0e3 * torch.finfo(X.dtype).tiny
        self.xi = torch.tensor([5.0], device=X.device)
        self.xi_target = xi_target

        self.omega_mh = omega_mh
        self.uniform_dist = Uniform(0.0, X.new_ones(1)[0])

        if self.subset_size is not None:
            self.anchor_size = subset_size // 2 if anchor_size is None else anchor_size
            self.pi = X.new_ones(self.P) * self.h if isinstance(S, (float, tuple)) else self.h
            self.total_weight = 0.0
            self.comb_factor = (self.subset_size - self.anchor_size) / (self.P - self.anchor_size)
        else:
            self.comb_factor = 1.0

        if verbose_constructor:
            s1 = "Initialized CountLikelihoodSampler with {} likelihood and (N, P, S, epsilon, subset_size) = "
            s2 = "({}, {}, {:.1f}, {:.1f}, {})" if not isinstance(S, tuple) \
                else "({}, {}, ({:.1f}, {:.1f}), {:.1f}, {})"
            if isinstance(S, float):
                S = (S,)
            elif isinstance(S, torch.Tensor):
                S = (S.min().item(), S.max().item())
            print((s1 + s2).format("Negative Binomial" if self.negbin
                  else "Binomial", self.N, self.P, *S, explore, subset_size))

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

        sample = SimpleNamespace(gamma=self.Xb.new_zeros(self.P).bool(),
                                 _omega=_omega,
                                 beta=self.Xb.new_zeros(self.P + self.Pa),
                                 beta_mean=self.Xb.new_zeros(self.P + self.Pa),
                                 _psi0=_psi0,
                                 _idx=torch.randint(self.P, (), device=self.device),
                                 weight=0,
                                 log_nu=log_nu,
                                 _kappa=_kappa,
                                 _kappa_omega=_kappa_omega,
                                 _Z=_Z,
                                 _log_h_ratio=self.log_h_ratio,
                                 _active=torch.tensor([], dtype=torch.int64),
                                 _activeb=self.assumed_covariates)

        if self.subset_size is not None:
            Z_cent = einsum("np,n->p", self.Xb[:, :self.P], self.Y - self.Y.mean())
            self._update_anchor(Z_cent.abs().argsort()[-self.anchor_size:])
            sample._active_subset = sample_active_subset(self.P, self.subset_size, self.anchor_subset,
                                                         self.anchor_subset_set, self.anchor_complement, sample._idx)

        if hasattr(self, "h_alpha"):
            sample.h_alpha = torch.tensor(self.h_alpha, device=self.device)
            sample.h_beta = torch.tensor(self.h_beta, device=self.device)

        sample = self.sample_beta(sample)  # populate self._L_active
        sample = self._compute_probs(sample)
        return sample

    def _update_anchor(self, anchor):
        self.anchor_subset = anchor
        self.anchor_subset_set = set(anchor.data.cpu().numpy().tolist())
        self.anchor_complement = arange_complement(self.P, anchor)

    def _compute_add_prob(self, sample):
        active, activeb = sample._active, sample._activeb

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

        # TODO: do we need to compute all of this if subset_size is not None?
        X_omega = self.Xb * sample._omega.sqrt().unsqueeze(-1)
        X_omega_k = X_omega[:, inactive]

        Z_k = sample._Z[inactive]
        X_omega_active = X_omega[:, activeb]
        Z_active = sample._Z[activeb]

        Zt_active = trisolve(self._L_active, Z_active.unsqueeze(-1), upper=False).squeeze(-1)
        Xt_active = trisolve(self._L_active, X_omega_active.t(), upper=False).t()
        XtZt_active = einsum("np,p->n", Xt_active, Zt_active)

        XX_k = norm(X_omega_k, dim=0).pow(2.0)
        G_k_inv = XX_k + self.tau - norm(einsum("ni,nk->ik", Xt_active, X_omega_k), dim=0).pow(2.0)
        W_k_sq = (einsum("np,n->p", X_omega_k, XtZt_active) - Z_k).pow(2.0) / (G_k_inv + self.epsilon)
        log_det_ratio_inactive = -0.5 * G_k_inv.log() + self.half_log_tau

        if num_active > 1:
            active_loo = leave_one_out(active)  # I I-1
            active_loob = torch.cat([active_loo, self.assumed_covariates.expand(num_active, -1)], dim=-1)

            Z_active_loo = sample._Z[active_loob]

            F = torch.cholesky_inverse(self._L_active, upper=False)
            F_loo = get_loo_inverses(F)[:-self.Pa]

            Zt_active_loo = matmul(F_loo, Z_active_loo.unsqueeze(-1)).squeeze(-1)
            Zt_active_loo_sq = einsum("ij,ij->i", Zt_active_loo, Z_active_loo)

            X_I_X_k = leave_one_out_off_diagonal(self._precision).unsqueeze(-1)[:-self.Pa]
            F_X_I_X_k = matmul(F_loo, X_I_X_k).squeeze(-1)

            XXFXX = einsum("ij,ij->i", X_I_X_k.squeeze(-1), F_X_I_X_k)
            G_k_inv = norm(X_omega[:, active], dim=0).pow(2.0) + self.tau - XXFXX
            log_det_ratio_active = -0.5 * G_k_inv.log() + self.half_log_tau
        elif num_active == 1:
            XX_assumed = self._precision[-self.Pa:][:, -self.Pa:]
            L_assumed = safe_cholesky(XX_assumed)
            Zt_active_loo = trisolve(L_assumed, sample._Z[self.assumed_covariates].unsqueeze(-1),
                                     upper=False).squeeze(-1)
            Zt_active_loo_sq = norm(Zt_active_loo, dim=0).pow(2.0)
            log_det_ratio_active = L_assumed.diagonal().log().sum() - self._L_active.diagonal().log().sum()
            log_det_ratio_active += 0.5 * math.log(self.tau)

        elif num_active == 0:
            Zt_active_loo_sq = 0.0  # dummy values since no active covariates
            log_det_ratio_active = torch.tensor(0.0)

        log_h_ratio_active = sample._log_h_ratio[active] if isinstance(self.h, torch.Tensor) else sample._log_h_ratio
        log_h_ratio_inactive = sample._log_h_ratio[inactive] if isinstance(self.h, torch.Tensor) \
            else sample._log_h_ratio

        log_odds_inactive = 0.5 * W_k_sq + log_det_ratio_inactive + log_h_ratio_inactive
        log_odds_active = 0.5 * (Zt_active.pow(2.0).sum() - Zt_active_loo_sq) + \
            log_det_ratio_active + log_h_ratio_active

        log_odds = self.Xb.new_full((self.P,), -torch.inf)
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

        if self.t <= self.T_burnin:  # adapt xi
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
            sample._activeb = torch.cat([sample._active, self.assumed_covariates], dim=-1)
            sample = self.sample_beta(sample)
        else:
            if hasattr(self, 'h_alpha'):
                if torch.rand(1).item() < 0.50:
                    sample = self.sample_alpha_beta(sample)
                    sample = self.sample_omega_nb(sample) if self.negbin else self.sample_omega_binomial(sample)
                else:
                    sample = self.sample_omega_nb(sample) if self.negbin else self.sample_omega_binomial(sample)
                    sample = self.sample_alpha_beta(sample)
            else:
                sample = self.sample_omega_nb(sample) if self.negbin else self.sample_omega_binomial(sample)

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

    def sample_beta(self, sample):
        activeb = sample._activeb
        Xb_active = self.Xb[:, activeb]
        precision = Xb_active.t() @ (sample._omega.unsqueeze(-1) * Xb_active)
        precision.diagonal(dim1=-2, dim2=-1).add_(self.tau)
        precision[-1, -1].add_(self.tau_intercept - self.tau)
        self._L_active = safe_cholesky(precision)
        self._precision = precision

        sample.beta.zero_()
        sample.beta_mean.zero_()

        beta_active = chosolve(sample._Z[activeb].unsqueeze(-1), self._L_active).squeeze(-1)
        sample.beta_mean[activeb] = beta_active
        sample.beta[activeb] = beta_active + \
            trisolve(self._L_active, torch.randn(activeb.size(-1), 1, device=self.device,
                                                 dtype=self.dtype),
                     upper=False).squeeze(-1)

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
            LZ = trisolve(L, sample._Z[activeb].unsqueeze(-1), upper=False).squeeze(-1)
            logdet = L.diag().log().sum() - L.size(-1) * self.half_log_tau

            return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet, L, precision

        log_target_prop, L_prop, precision_prop = compute_log_target(omega_prop)
        beta_mean_prop = chosolve(sample._Z[activeb].unsqueeze(-1), L_prop).squeeze(-1)
        beta_prop = beta_mean_prop + \
            trisolve(L_prop, torch.randn(activeb.size(-1), 1, device=self.device, dtype=self.dtype),
                     upper=False).squeeze(-1)
        psi_prop = torch.mv(Xb_active, beta_mean_prop)

        if self.omega_mh:
            log_target_curr, _, _ = compute_log_target(sample._omega)
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
            self._precision = precision_prop

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
            LZ = trisolve(L, Z[activeb].unsqueeze(-1), upper=False).squeeze(-1)
            logdet = L.diag().log().sum() - L.size(-1) * self.half_log_tau

            return 0.5 * norm(LZ, dim=0).pow(2.0) - logdet, L, precision

        log_target_prop, L_prop, precision_prop = compute_log_target(omega_prop, Z_prop)
        beta_mean_prop = chosolve(Z_prop[activeb].unsqueeze(-1), L_prop).squeeze(-1)
        beta_prop = beta_mean_prop + \
            trisolve(L_prop, torch.randn(activeb.size(-1), 1, device=self.device, dtype=self.dtype),
                     upper=False).squeeze(-1)

        psi_prop = torch.mv(Xb_active, beta_mean_prop)
        log_target_curr, _, _ = compute_log_target(sample._omega, sample._Z)

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
            self._precision = precision_prop
            sample._kappa = kappa_prop
            sample._psi0 = psi0_prop
            sample._kappa_omega = kappa_omega_prop
            sample._Z = Z_prop
            sample.beta_mean.zero_()
            sample.beta_mean[activeb] = beta_mean_prop
            sample.beta.zero_()
            sample.beta[activeb] = beta_prop

        return sample

    def sample_alpha_beta(self, sample):
        num_active = sample._active.size(-1)
        num_inactive = self.P - num_active
        sample.h_alpha = torch.tensor(self.h_alpha + num_active, device=self.Xb.device)
        sample.h_beta = torch.tensor(self.h_beta + num_inactive, device=self.Xb.device)
        h = Beta(sample.h_alpha, sample.h_beta).sample().item()
        sample._log_h_ratio = math.log(h) - math.log(1.0 - h)
        return sample
