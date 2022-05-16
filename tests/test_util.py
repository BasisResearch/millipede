import torch

from millipede.util import safe_cholesky, set_subtract


def test_safe_cholesky_smoke_test(D=10):
    X = torch.randn(D, 1)
    XX = X.t() @ X - 3.0e-5 * torch.eye(D)
    safe_cholesky(XX)


def test_set_subtract():

    def _test(a, b):
        result = set_subtract(torch.tensor(a), torch.tensor(b)).data.numpy().tolist()
        assert len(set(result)) == len(result)
        assert set(result) == set(a) - set(b)

    for a in ([], [0], [0, 1], [0, 1, 2]):
        for b in ([], [0], [0, 1], [0, 1, 2]):
            _test(a, b)
