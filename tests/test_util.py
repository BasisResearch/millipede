import torch

from millipede.util import safe_cholesky, set_subtract, set_intersect


def test_safe_cholesky_smoke_test(D=10):
    X = torch.randn(D, 1)
    XX = X.t() @ X - 3.0e-5 * torch.eye(D)
    safe_cholesky(XX)


def test_set_arithmetic():

    def _test(a, b):
        result = set_subtract(a, b).data.numpy().tolist()
        assert len(set(result)) == len(result)
        assert set(result) == set(a.data.numpy().tolist()) - set(b.data.numpy().tolist())

        result = set_intersect(a, b).data.numpy().tolist()
        assert len(set(result)) == len(result)
        assert set(result) == set(a.data.numpy().tolist()).intersection(set(b.data.numpy().tolist()))

    for _ in range(5):
        for a in range(10):
            for b in range(10):
                _a = torch.randperm(12)[:a] if a > 0 else torch.tensor([])
                _b = torch.randperm(12)[:b] if b > 0 else torch.tensor([])
                _test(_a, _b)
