import pickle
import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


def get_data():
    X, Y, beta, indices = pickle.load(open('X_Y_beta_indices.semisynth.pkl', 'rb'))

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    N, P = X.shape

    #print("Beta: ", beta)
    #print("indices", indices)

    print("XY", X.shape, Y.shape)

    columns = ['feat{}'.format(c) for c in range(P)] + ['response']
    XY = torch.cat([X, Y.unsqueeze(-1)], axis=-1)
    dataframe = pd.DataFrame(XY.data.numpy(), columns=columns)

    return dataframe, indices  # X, Y, indices


def trial(dataframe, prior="isotropic", precompute_XX=False,
          T=10 * 1000, T_burnin=2000, seed=1, subset_size=None, indices=None):

    selector = NormalLikelihoodVariableSelector(dataframe, 'response', assumed_columns=[],
                                                tau=1.0e-4, c=0.0,
                                                precompute_XX=False,
                                                include_intercept=False, prior='isotropic',
                                                S=10, nu0=0.0, lambda0=0.0, precision='double',
                                                explore=5.0,
                                                device='gpu',
                                                subset_size=subset_size)

    selector.run(T=T, T_burnin=T_burnin, verbosity='bar', streaming=True, seed=seed)

    print("[selector.stats]\n", selector.stats)

    weights = selector.weights
    weights = weights.shape[0] * weights / weights.sum()
    print("Normalized weights standard deviation: {:.4f}".format(weights.std().item()))

    pip = selector.pip.values
    print("PIP sum: ", pip.sum().item())
    if indices is not None:
        print("Restricted PIP sum: ", pip[indices].sum().item())

    return pip


subset_size = int(sys.argv[1])
if subset_size == 0:
    subset_size = None
    T = 10 * 4000
    T_burnin = 2000
else:
    T = 10 * 4 * 4000
    T_burnin = 4 * 2000

num_trials = 1
dataframe, indices = get_data()
#X, Y, indices = get_data()
pips = []
t0 = time.time()

for t in range(num_trials):
    pip = trial(dataframe, subset_size=subset_size, T_burnin=T_burnin, T=T, seed=t, indices=indices)
    pips.append(pip)

t1 = time.time()
print("total elapsed time: ", t1 - t0, "  subset_size:", subset_size)
pips = np.stack(pips)
np.save('pips.semisynth.{}.{}.{}'.format(subset_size, T, T_burnin), pips)
