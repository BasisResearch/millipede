import pytest
import torch

from millipede.util import (
    arange_complement,
    safe_cholesky,
    sample_active_subset,
)


def test_safe_cholesky_smoke_test(D=10):
    X = torch.randn(D, 1)
    XX = X.t() @ X - 3.0e-5 * torch.eye(D)
    safe_cholesky(XX)


@pytest.mark.parametrize("subset_size", [2, 3, 4, 5, 6, 7, 9])
def test_sample_active_subset(subset_size, P=10):
    A = subset_size // 2
    anchor_subset = torch.randperm(P)[:A]
    anchor_subset_set = set(anchor_subset.data.cpu().numpy().tolist())
    anchor_complement = arange_complement(P, anchor_subset)

    # test arange_complement
    anchor_complement_direct = list(range(P))
    for i in anchor_subset_set:
        anchor_complement_direct.remove(i)
    assert set(anchor_complement_direct) == set(anchor_complement.data.cpu().numpy().tolist())

    idx = torch.randint(P, ())
    active_subset = sample_active_subset(P, subset_size, anchor_subset, anchor_subset_set, anchor_complement, idx)
    for i in anchor_subset:
        assert i in active_subset
    assert idx.item() in active_subset
    assert active_subset.size(0) == subset_size
