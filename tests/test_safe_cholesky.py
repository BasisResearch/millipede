import torch

from millipede.util import safe_cholesky


def test_safe_cholesky_smoke_test(D=10):
    X = torch.randn(D, 1)
    XX = X.t() @ X - 3.0e-5 * torch.eye(D)
    safe_cholesky(XX)
