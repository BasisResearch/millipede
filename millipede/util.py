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


def leave_one_out(x):
    N = x.size(-1)
    mask = ~torch.eye(N, N, device=x.device).bool()
    return x.expand(N, N)[mask].reshape(N, N - 1)


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
