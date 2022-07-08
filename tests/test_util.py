import pytest

import torch

from millipede.util import safe_cholesky, set_subtract, set_intersect, sample_active_subset, arange_complement


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

    for _ in range(50):
        for a in range(10):
            for b in range(10):
                _a = torch.randperm(12)[:a] if a > 0 else torch.tensor([])
                _b = torch.randperm(12)[:b] if b > 0 else torch.tensor([])
                _test(_a, _b)


@pytest.mark.parametrize("subset_size", [2, 3, 4, 5, 6, 7, 9])
def test_sample_active_subset(subset_size, P=10):
    A = subset_size // 2
    anchor_subset = torch.randperm(P)[:A]
    anchor_subset_set = set(anchor_subset.data.cpu().numpy().tolist())
    anchor_complement = arange_complement(P, anchor_subset)
    idx = torch.randint(P, ())
    active_subset = sample_active_subset(P, subset_size, anchor_subset, anchor_subset_set, anchor_complement, idx)
    for i in anchor_subset:
        assert i in active_subset
    assert idx.item() in active_subset
    assert active_subset.size(0) == subset_size
