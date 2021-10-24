from types import SimpleNamespace

import numpy as np
import torch


class Timer(object):
    def __init__(self, name="TimerEvent"):
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __call__(self):
        self.end.record()
        torch.cuda.synchronize()
        print("[{}]  {:.5f}".format(self.name, self.start.elapsed_time(self.end)))


def divide_int(x, divisor):
    remainder = x % divisor
    ints = [x // divisor + int(n < remainder) for n in range(divisor)]
    assert sum(ints) == x
    return ints


def subdivide_data(X, Y, T, max_count=100):
    N = X.size(0)
    assert Y.shape == T.shape == (N,)

    def get_rows(Xn, Yn, Tn):
        if Tn <= max_count:
            return [Xn], [Yn], [Tn]
        else:
            num_rows = Tn // max_count + int(Tn % max_count > 0)
            Ts = divide_int(Tn, num_rows)
            Ys = divide_int(Yn, num_rows)
            return [Xn] * num_rows, Ys, Ts

    X_new, Y_new, T_new = [], [], []
    for n in range(N):
        X_rows, Y_rows, T_rows = get_rows(X[n], Y[n], T[n])
        X_new.extend(X_rows)
        Y_new.extend(Y_rows)
        T_new.extend(T_rows)

    X_new = torch.stack(X_new)
    Y_new = torch.stack(Y_new)
    T_new = torch.stack(T_new)

    assert T_new.sum() == T.sum()
    assert Y_new.sum() == Y.sum()

    return X_new, Y_new, T_new


def safe_cholesky(A, epsilon=1.0e-8):
    try:
        return torch.linalg.cholesky(A)
    except RuntimeError as e:
        Aprime = A.clone()
        jitter_prev = 0.0
        for i in range(3):
            jitter_new = epsilon * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                return torch.linalg.cholesky(Aprime, upper=False)
            except RuntimeError:
                continue
        raise e


def check(t, s):
    nan = torch.isnan(t).sum().item()
    inf = torch.isinf(t).sum().item()
    print("[{}]  nans: {}  infs: {}".format(s, nan, inf))


def leave_one_out(x):
    N = x.size(-1)
    mask = ~torch.eye(N, N, device=x.device).bool()
    return x.expand(N, N)[mask].reshape(N, N - 1)


def compute_submatrices(XX, X, Z):
    N = XX.size(-1)
    num_submatrices = 2**N - 1

    s = torch.arange(num_submatrices)
    n = torch.arange(N)

    XX_sub = torch.zeros(XX.shape[:-2] + s.shape + XX.shape[-2:])
    X_sub = torch.zeros(X.shape[:-2] + s.shape + X.shape[-2:])
    Z_sub = torch.zeros(Z.shape[:-1] + s.shape + X.shape[-1:])

    source_mask1 = ~((s.unsqueeze(-1) >> n) % 2).bool()
    source_mask2 = source_mask1.unsqueeze(-1) & source_mask1.unsqueeze(-2)

    destin_mask1 = n < source_mask1.sum(-1).unsqueeze(-1)
    destin_mask2 = destin_mask1.unsqueeze(-1) & destin_mask1.unsqueeze(-2)

    dXX = XX - torch.eye(N, dtype=XX.dtype, device=XX.device)
    XX_sub[..., destin_mask2] = dXX.unsqueeze(-3).expand_as(XX_sub)[..., source_mask2]
    XX_sub.reshape(XX_sub.shape[:-2] + (N * N,))[..., ::N + 1] += 1  # Add 1 to diagonal

    X_sub[destin_mask1.unsqueeze(-2).expand_as(X_sub)] = X.expand_as(X_sub)[source_mask1.unsqueeze(-2).expand_as(X_sub)]
    Z_sub[destin_mask1.expand_as(Z_sub)] = Z.expand_as(Z_sub)[source_mask1.expand_as(Z_sub)]

    indices = -torch.ones(num_submatrices, N).long()
    indices[destin_mask1] = torch.arange(N).expand(num_submatrices, N)[source_mask1]

    return XX_sub, X_sub, Z_sub, indices


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
