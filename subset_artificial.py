import sys
import time

import numpy as np
import pandas
import pytest
import torch

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


def get_data(N=256, K=64, P=1024 * 32, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(N, P).double()
    Y = torch.zeros(N)
    for k in range(K // 2):
        left, right = 2 * k, 2 * (k + 1)
        Z = torch.randn(N).double()
        X[:, left:right] = Z.unsqueeze(-1) + 0.001 * torch.randn(N, 2).double()
        Y += ((-1) ** k) * Z
    Y = Y + 0.1 * torch.randn(N).double()

    X = X.cuda()
    Y = Y.cuda()
    return X, Y


def trial(X, Y, prior="isotropic", precompute_XX=False,
          T=10 * 1000, T_burnin=2000, seed=1, subset_size=None):

    samples = []
    sampler = NormalLikelihoodSampler(X, Y, X_assumed=None,
                                      precompute_XX=precompute_XX, prior=prior,
                                      compute_betas=True, S=10.0, nu0=0.0, lambda0=0.0,
                                      tau=0.01, c=100.0, include_intercept=False,
                                      tau_intercept=1.0e-4,
                                      subset_size=subset_size)

    t0 = time.time()

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    #print("weights", weights.min(), np.median(weights), weights.max())

    pip = np.dot(samples.pip.T, weights)
    print("pip[:6]", pip[:6])

    #pip_rest = pip[2:]
    #print("pip_rest: ", pip_rest.min(), pip_rest.max(), pip_rest.mean())
    print("Elapsed MCMC time: ", time.time() - t0)
    #print("MCMC times", sampler.time2, sampler.time3)

    return pip


subset_size = int(sys.argv[1])
if subset_size == 0:
    subset_size = None
    T = 5000
else:
    T = 20000

num_trials = 10
X, Y = get_data(N=4096, K=16, P=1024 * 32, seed=11)
pips = []
t0 = time.time()

for t in range(num_trials):
    pip = trial(X, Y, subset_size=subset_size, T_burnin=2000, T=T, seed=t)
    pips.append(pip)

t1 = time.time()
print("total elapsed time: ", t1 - t0, "  subset_size:", subset_size)
pips = np.stack(pips)
np.save('pips.{}'.format(subset_size), pips)
