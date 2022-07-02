import numpy as np
import pandas as pd
import pytest
import torch
import time
import sys
import pickle

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


def _get_data(seed=0, p=10, sigma=0.1):
    torch.manual_seed(seed)

    X = torch.from_numpy(pd.read_feather('X.feather').values)
    #Y = torch.from_numpy(pd.read_feather('Y.feather').values[:, 0])
    X = X.cuda().double()
    #Y = Y.cuda().double()
    N, P = X.shape

    idx = torch.randperm(P)
    print("idx", idx[::2][:5].data.cpu().numpy())
    X = X[:, idx[::2]]

    beta = 0.1 + 0.9 * torch.rand(p).type_as(X)
    beta *= (-1) ** torch.bernoulli(0.5 * torch.ones(p)).type_as(X)
    print("Beta: ", beta.data.cpu().numpy())

    Y = torch.mv(X[:, :p], beta)
    Y += sigma * torch.randn(N).type_as(X)

    print("XY", X.shape, Y.shape)
    return X, Y


def get_data():
    X, Y, beta, indices = pickle.load(open('X_Y_beta_indices.semisynth.pkl', 'rb'))

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    X = X.cuda().double()
    Y = Y.cuda().double()
    N, P = X.shape

    print("Beta: ", beta)
    print("indices", indices)

    print("XY", X.shape, Y.shape)
    return X, Y, indices


def trial(X, Y, prior="isotropic", precompute_XX=False,
          T=10 * 1000, T_burnin=2000, seed=1, subset_size=None, indices=None):

    samples = []
    sampler = NormalLikelihoodSampler(X, Y, X_assumed=None,
                                      precompute_XX=precompute_XX, prior=prior,
                                      explore=5.0,
                                      compute_betas=True, S=10.0, nu0=0.0, lambda0=0.0,
                                      tau=0.01, c=100.0, include_intercept=False,
                                      tau_intercept=1.0e-4,
                                      subset_size=subset_size)

    t0 = time.time()

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            samples.append(namespace_to_numpy(s))
            #if t % 100 == 0:
            #    print("[step {}]  {}".format(t, s.gamma.sum().item()))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    #print("weights", weights.min(), np.median(weights), weights.max())

    pip = np.dot(samples.pip.T, weights)
    print("PIP sum: ", pip.sum().item())
    if indices is not None:
        print("Restricted PIP sum: ", pip[indices].sum().item())

    print("Elapsed MCMC time: ", time.time() - t0)
    #print("MCMC times", sampler.time2, sampler.time3)

    return pip


subset_size = int(sys.argv[1])
if subset_size == 0:
    subset_size = None
    T = 4000
    T_burnin = 4000
else:
    T = 4 * 4000
    T_burnin = 4 * 4000

num_trials = 10
X, Y, indices = get_data()
pips = []
t0 = time.time()

for t in range(num_trials):
    pip = trial(X, Y, subset_size=subset_size, T_burnin=T_burnin, T=T, seed=t, indices=indices)
    pips.append(pip)

t1 = time.time()
print("total elapsed time: ", t1 - t0, "  subset_size:", subset_size)
pips = np.stack(pips)
np.save('pips.semisynth.{}.{}.{}'.format(subset_size, T, T_burnin), pips)
