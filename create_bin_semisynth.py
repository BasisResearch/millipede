import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from millipede import NormalLikelihoodSampler, NormalLikelihoodVariableSelector
from millipede.util import namespace_to_numpy, stack_namespaces


p = 20
torch.manual_seed(0)

X = torch.from_numpy(pd.read_feather('X.feather').values).cuda()
X = (X == X.min(0)[0])
X = X.to(dtype=torch.uint8)
assert X.min().item() == 0
assert X.max().item() == 1

N, P = X.shape
print("X.shape: ", X.shape)

idx = torch.randperm(P, device=X.device)
X = X[:, idx]
block_size = P // p
indices = []

for i in range(p):
    left = i * block_size
    right = (i + 1) * block_size if i < p - 1 else left + (P - (p - 1) * block_size)
    print(i, left, right, right - left)
    XX = X[:, left:right].double()
    XX -= XX.mean(0)
    XX /= XX.std(0)
    XX = XX.T @ XX / X.size(0)
    diag = torch.arange(right - left)
    XX[diag, diag] = 0.0
    XX = XX.abs()
    pairs = torch.nonzero((XX > 0.5) & (XX < 0.9))
    pair = pairs[torch.randperm(pairs.size(0))[0]]
    pair = pair[torch.randperm(2)]
    indices.append(left + pair[0].item())

assert len(indices) == p
print("indices", indices)

beta = 0.1 + 0.9 * torch.rand(p, device=X.device).double()
beta *= (-1) ** torch.bernoulli(0.5 * torch.ones(p)).to(dtype=torch.double, device=X.device)
print("Beta: ", beta.data.cpu().numpy())

Y = torch.zeros(N).cuda().double()
for i in range(p):
    Y += X[:, indices[i]].double() * beta[i]

sigma = 0.1
Y += sigma * torch.randn(N).cuda().double()

assert X.shape == (N, P) and Y.shape == (N,)

target = 750 * 1000
if target > P:
    X = torch.cat([X, torch.bernoulli(0.1 * torch.ones(N, target - P, device=X.device, dtype=X.dtype))], dim=-1)
    print("New X: ", X.shape)

pickle.dump((X.data.cpu().numpy(), Y.data.cpu().numpy(), beta.data.cpu().numpy(), indices),
            open('X_Y_beta_indices.semisynth.p20binary.750k.pkl', 'wb'))
