from types import SimpleNamespace

import numpy as np
import torch


def safe_cholesky(A, epsilon=1.0e-8):
    """
    Equivalent of torch.linalg.cholesky that progressively adds
    diagonal jitter to avoid cholesky errors.
    """
    try:
        return torch.linalg.cholesky(A)
    except RuntimeError as e:
        Aprime = A.clone()
        jitter_prev = 0.0
        for i in range(5):
            jitter_new = epsilon * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                return torch.linalg.cholesky(Aprime)
            except RuntimeError:
                continue
        raise e


def get_loo_inverses(F):
    N = F.size(-1)

    mask = torch.ones(N, N, N).bool()
    idx = torch.arange(N)
    mask[idx, idx] = 0
    mask[idx, :, idx] = 0
    F_sub = F.expand(N, N, N)[mask].reshape(N, N - 1, N - 1)

    mask = torch.zeros(N, N, N).bool()
    mask[idx, idx] = 1
    mask[idx, idx, idx] = 0
    F_top = F.expand(N, N, N)[mask].reshape(N, 1, N - 1)

    mask = torch.zeros(N, N, N).bool()
    mask[idx, :, idx] = 1
    mask[idx, idx, idx] = 0
    F_left = F.expand(N, N, N)[mask].reshape(N, N - 1, 1)

    F_corner = F[idx, idx]

    F_loo = F_sub - torch.matmul(F_left, F_top) / F_corner.unsqueeze(-1).unsqueeze(-1)

    return F_loo


def leave_one_out(x):
    N = x.size(-1)
    mask = ~torch.eye(N, N, device=x.device).bool()
    return x.expand(N, N)[mask].reshape(N, N - 1)


def leave_one_out_off_diagonal(x):
    N = x.size(-1)
    N_arange = torch.arange(N, device=x.device)
    mask = N_arange.expand(N, N, N) != N_arange.unsqueeze(-1).unsqueeze(-1)
    mask = ~mask & mask.transpose(dim0=-1, dim1=-2)
    return x.expand(N, N, N)[mask].reshape(N, N - 1)


def namespace_to_numpy(namespace, filter_sites=True, keep_sites=[]):
    attributes = list(namespace.__dict__.keys())
    d = {}
    for attr in attributes:
        val = namespace.__getattribute__(attr)
        filter_site = filter_sites and attr[0] == '_' and attr not in keep_sites
        if val is not None and hasattr(val, 'data') and not filter_site:
            d[attr] = val.data.cpu().numpy().copy()
    return SimpleNamespace(**d)


def stack_namespaces(namespaces):
    attributes = list(namespaces[0].__dict__.keys())
    d = {}
    for attr in attributes:
        val = namespaces[0].__getattribute__(attr)
        if val is not None:
            d[attr] = np.stack([ns.__getattribute__(attr) for ns in namespaces])
    return SimpleNamespace(**d)
