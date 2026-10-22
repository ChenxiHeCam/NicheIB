import torch

from nicheib.baselines import matched_budget_edge_mask, prune_edge_index


def test_matched_budget_edge_mask_keeps_exact_count():
    scores = torch.arange(10, dtype=torch.float32)

    mask = matched_budget_edge_mask(scores, 0.2)

    assert mask.dtype == torch.bool
    assert mask.sum().item() == 2
    assert mask.tolist() == [False, False, False, False, False, False, False, False, True, True]


def test_matched_budget_edge_mask_handles_extremes():
    scores = torch.arange(5, dtype=torch.float32)

    assert matched_budget_edge_mask(scores, 0.0).sum().item() == 0
    assert matched_budget_edge_mask(scores, -1.0).sum().item() == 0
    assert matched_budget_edge_mask(scores, 1.0).sum().item() == 5
    assert matched_budget_edge_mask(scores, 2.0).sum().item() == 5


def test_prune_edge_index_selects_columns():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    mask = torch.tensor([True, False, True, False])

    pruned = prune_edge_index(edge_index, mask)

    expected = torch.tensor([[0, 2], [1, 3]])
    assert torch.equal(pruned, expected)
