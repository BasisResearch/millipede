import pickle
import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


def get_data(mixed=False):
    X, Y, beta, indices = pickle.load(open('X_Y_beta_indices.semisynth.p20binary.500k.pkl', 'rb'))

    X = torch.from_numpy(X)
    print(X[:10, :10])
    Y = torch.from_numpy(Y)
    N, P = X.shape
    print("N/P", N, P)

    print("Beta: ", beta)
    print("indices", indices)

    columns = ['feat{}'.format(c) for c in range(P)]
    if mixed:
        dataframe = pd.DataFrame(np.array(X.data.numpy(), dtype=np.byte), columns=columns)
    else:
        dataframe = pd.DataFrame(np.array(X.data.numpy(), dtype=np.float64), columns=columns)
    dataframe['response'] = Y.data.numpy()

    print("XY shapes", X.shape, Y.shape, "dataframe XY dtypes",
          dataframe['feat0'].values.dtype, dataframe['response'].values.dtype)

    return dataframe, indices, beta


def trial(dataframe, prior="isotropic", precompute_XX=False,
          T=10 * 1000, T_burnin=2000, seed=1, subset_size=None, indices=None,
          precision='double', beta=None):

    selector = NormalLikelihoodVariableSelector(dataframe, 'response', assumed_columns=[],
                                                tau=1.0e-4, c=0.0,
                                                precompute_XX=precompute_XX,
                                                include_intercept=False, prior='isotropic',
                                                S=10, nu0=0.0, lambda0=0.0, precision=precision,
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

    if beta is not None:
        beta_full = np.zeros(dataframe.values.shape[1] - 1)
        beta_full[indices] = beta
        beta_rmse = np.sqrt(np.sum(np.square(beta_full - selector.beta.values)))
        print("beta rmse: ", beta_rmse)

    return pip


subset_size = int(sys.argv[1])
if subset_size == 0:
    subset_size = None
    T = 1000
    T_burnin = 500
else:
    T = 5000
    T_burnin = 1000

mixed = 0
precision = "double" if not mixed else "mixeddouble"
print("mixed", mixed, "precision", precision)

num_trials = 1
dataframe, indices, beta = get_data(mixed=mixed)
pips = []
t0 = time.time()

for t in range(num_trials):
    pip = trial(dataframe, subset_size=subset_size, T_burnin=T_burnin, T=T, seed=t, indices=indices,
                precision=precision, beta=beta)
    pips.append(pip)

t1 = time.time()
print("total elapsed time: ", t1 - t0, "  subset_size:", subset_size)
pips = np.stack(pips)
#np.save('pips.semisynth.p8.{}.{}.{}'.format(subset_size, T, T_burnin), pips)
