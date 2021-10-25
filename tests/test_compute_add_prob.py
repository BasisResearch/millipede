import torch
from torch import zeros
import numpy as np
from types import SimpleNamespace

from common import assert_close
from millipede import NormalLikelihoodSampler


import math


torch.set_default_tensor_type('torch.DoubleTensor')
precompute_XX = 0

N = 35
P = 9
X = torch.randn(N, P)
Y = X[:, 0] + 0.2 * torch.randn(N)

tau = 0.47
sampler = NormalLikelihoodSampler(X, Y, S=1, c=0.73, tau=tau, precompute_XX=precompute_XX, prior="isotropic")
YY = sampler.YY
Z = sampler.Z
hc = sampler.hc_prefactor
log_h_ratio = sampler.log_h_ratio
c_one_c = sampler.c_one_c
print("c_one_c",c_one_c)
print("log_h_ratio ",log_h_ratio )
print("logtau", math.log(tau))

def compute_log_factor_gprior(ind):
    F = torch.inverse(X[:, ind].t() @ X[:, ind] + tau * torch.eye(len(ind)))
    ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
    logdet = torch.logdet(X[:, ind].t() @ X[:, ind] / tau + torch.eye(len(ind)))
    return 0.5 * ZFZ - 0.5 * logdet

def compute_log_factor_isotropic(ind):
    F = torch.inverse(X[:, ind].t() @ X[:, ind] + tau * torch.eye(len(ind)))
    ZFZ = (torch.mv(F, Z[ind]) * Z[ind]).sum(0)
    logdet = 0.5 * F.diag().log().sum()
    return -0.5 * N * (YY - ZFZ).log() + logdet

def compute_log_factor_ratio_isotropic(ind1, ind0):
    return compute_log_factor_isotropic(ind1) - compute_log_factor_isotropic(ind0) + log_h_ratio + 0.5 * math.log(tau)


########################
# TEST GAMMA = 0 0 0 0 #
########################

sample = SimpleNamespace(gamma=zeros(P).bool(), add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
for p in range(P):
    assert_close(compute_log_factor_ratio_isotropic([p], []), log_odds[p], atol=1.0e-10)

########################
# TEST GAMMA = 1 0 0 0 #
########################

gamma = [1] + [0] * (P - 1)
sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                         add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
log_odds = sampler._compute_add_prob(sample, return_log_odds=True)

for p in range(1, P):
    print("p", p)
    assert_close(compute_log_factor_ratio_isotropic([0, p], [0]), log_odds[p], atol=1.0e-3)

import sys; sys.exit()

########################
# TEST GAMMA = 1 1 0 0 #
########################

gamma = [1, 1] + [0] * (P - 2)
sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                         add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
log_odds = sampler._compute_add_prob(sample, return_log_odds=True)

for p in range(2, 5):
    assert_close(compute_log_factor_ratio_isotropic([0, 1, p], [0, 1]), log_odds[p], atol=1.0e-10)

########################
#    VANILLA PRIOR     #
# TEST GAMMA = 0 0 0 0 #
########################

sampler = NormalLikelihoodSampler(X, Y, S=1, precompute_XX=precompute_XX, prior="gprior")
sample = SimpleNamespace(gamma=zeros(P).bool(), add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
for p in range(P):
    assert_close(compute_log_factor_ratio_gprior([p], []), log_odds[p], atol=1.0e-10)


import sys;sys.exit()

########################
# TEST GAMMA = 1 0 0 0 #
########################

gamma = [1] + [0] * (P - 1)
sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                         add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
log_odds = sampler._compute_add_prob(sample, return_log_odds=True)
p = 0
ZFZ = Z[p].pow(2.0) / (X[:, p].pow(2.0).sum(0) + tau)
S0 = YY
S1 = YY - ZFZ
logdet = -0.5 * (X[:, p].pow(2.0).sum(0) / tau + 1).log()
logS = 0.5 * N * (S0.log() - S1.log()) + log_h_ratio + logdet
check(logS, log_odds[p])

ZFZ0 = Z[0].pow(2.0) / (X[:, 0].pow(2.0).sum(0) + tau)
S0 = YY - ZFZ0
for p in range(1, P):
    ind = [0, p]
    F = torch.inverse(X[:, ind].t() @ X[:, ind] + tau * torch.eye(len(ind)))
    Zp = Z[ind]
    ZFZp = (torch.mv(F, Zp) * Zp).sum(0)
    S1 = YY - ZFZp
    logdet = -0.5 * torch.logdet(X[:, ind].t() @ X[:, ind] / tau + torch.eye(len(ind))) +\
             0.5 * (X[:, 0].pow(2.0).sum(0) / tau + 1).log()
    logS = 0.5 * N * (S0.log() - S1.log()) + log_h_ratio + logdet
    check(logS, log_odds[p])

########################
# TEST GAMMA = 1 1 0 0 #
########################

gamma = [1, 1] + [0] * (P - 2)
sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                         add_prob=zeros(P), i_prob=zeros(P), idx=0, weight=0)
log_odds = sampler._compute_add_prob(sample, return_log_odds=True)

p = 0
ZFZ1 = Z[1].pow(2.0) / (X[:, 1].pow(2.0).sum(0) + tau)
S0 = YY - ZFZ1
ind = [p, 1]
F = torch.inverse(X[:, ind].t() @ X[:, ind] + tau * torch.eye(len(ind)))
Zp = Z[ind]
ZFZp = (torch.mv(F, Zp) * Zp).sum(0)
S1 = YY - ZFZp
logdet = -0.5 * torch.logdet(X[:, ind].t() @ X[:, ind] / tau + torch.eye(len(ind))) +\
         0.5 * (X[:, 1].pow(2.0).sum(0) / tau + 1).log()
logS = 0.5 * N * (S0.log() - S1.log()) + logdet + log_h_ratio
check(logS, log_odds[p])

p = 1
ZFZ0 = Z[0].pow(2.0) / (X[:, 0].pow(2.0).sum(0) + tau)
S0 = YY - ZFZ0
ind = [0, p]
F = torch.inverse(X[:, ind].t() @ X[:, ind] + tau * torch.eye(len(ind)))
Zp = Z[ind]
ZFZp = (torch.mv(F, Zp) * Zp).sum(0)
S1 = YY - ZFZp
logdet = -0.5 * torch.logdet(X[:, ind].t() @ X[:, ind] / tau + torch.eye(len(ind))) +\
         0.5 * (X[:, 0].pow(2.0).sum(0) / tau + 1).log()
logS = 0.5 * N * (S0.log() - S1.log()) + logdet + log_h_ratio
check(logS, log_odds[p])

F0 = torch.inverse(X[:, :2].t() @ X[:, :2] + tau * torch.eye(2))
ZFZ0 = (torch.mv(F0, Z[:2]) * Z[:2]).sum(0)
S0 = YY - ZFZ0
for p in range(2, P):
    ind = [0, 1, p]
    F = torch.inverse(X[:, ind].t() @ X[:, ind] + tau * torch.eye(3))
    Zp = Z[ind]
    ZFZp = (torch.mv(F, Zp) * Zp).sum(0)
    S1 = YY - ZFZp
    logdet = -0.5 * torch.logdet(X[:, ind].t() @ X[:, ind] / tau + torch.eye(len(ind))) +\
              0.5 * torch.logdet(X[:, :2].t() @ X[:, :2] / tau + torch.eye(2))
    logS = 0.5 * N * (S0.log() - S1.log()) + logdet + log_h_ratio
    check(logS, log_odds[p])


print("PASSED {} ASSERTS    [max delta: {:.2e}]".format(num_asserts, max_delta))
