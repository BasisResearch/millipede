import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces

p = 8
torch.manual_seed(0)

X = torch.from_numpy(pd.read_feather('X.feather').values)
X = X.cuda().double()
N, P = X.shape
print("X.shape: ", X.shape)

idx = torch.randperm(P)
keep = p * ((P // 4) // p)
block_size = keep // p
X = X[:, idx[:keep]]
print("X.shape: ", X.shape, "block_size", block_size)
diag = torch.arange(block_size)

indices = []

for i in range(p):
    left = i * block_size
    right = (i + 1) * block_size
    XX = X[:, left:right]
    XX = XX.T @ XX / X.size(0)
    assert XX.shape == (block_size, block_size)
    XX[diag, diag] = 0.0
    XX = XX.abs()
    pairs = torch.nonzero((XX > 0.5) & (XX < 0.9))
    pair = pairs[torch.randperm(pairs.size(0))[0]]
    pair = pair[torch.randperm(2)]
    indices.append(left + pair[0].item())

assert len(indices) == p
print("indices", indices)

beta = 0.1 + 0.9 * torch.rand(p).type_as(X)
beta *= (-1) ** torch.bernoulli(0.5 * torch.ones(p)).type_as(X)
print("Beta: ", beta.data.cpu().numpy())

Y = torch.zeros(N).type_as(X)
for i in range(p):
    Y += X[:, indices[i]] * beta[i]

sigma = 0.1
Y += sigma * torch.randn(N).type_as(X)

print("XY", X.shape, Y.shape)

pickle.dump((X.data.cpu().numpy(), Y.data.cpu().numpy(), beta.data.cpu().numpy(), indices),
            open('X_Y_beta_indices.semisynth.p8.pkl', 'wb'))
