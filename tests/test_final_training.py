"""Small CPU-only tests for the camera-ready final training and export path."""

from types import SimpleNamespace

import numpy as np
import torch

from experiments.tsb.dlpfc_matched_mask_refit import coverage_penalty, topk, train_tsb


def test_expected_in_degree_coverage():
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    probabilities = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    penalty = coverage_penalty(probabilities, edges, 3, 1.0)
    assert torch.allclose(penalty, torch.tensor(0.25))
    penalty.backward()
    assert torch.isfinite(probabilities.grad).all()


def test_final_training_and_exact_export_count():
    torch.manual_seed(4)
    n = 12
    src = torch.arange(n).repeat_interleave(2)
    dst = torch.stack(((torch.arange(n) + 1) % n,
                       (torch.arange(n) - 1) % n), dim=1).flatten()
    data = SimpleNamespace(x=torch.randn(n, 5), y=torch.arange(n) % 2,
                           edge_index=torch.stack((src, dst)))
    train = torch.arange(n) < 8
    _, logits, scores, rate = train_tsb(data, train, 2, seed=3,
                                       epochs=3, budget=0.2, variant="final")
    assert logits.shape == (n, 2)
    assert torch.isfinite(logits).all()
    assert np.isfinite(scores).all()
    assert 0 <= rate <= 1
    count = max(1, round(rate * len(scores)))
    mask = topk(scores, count)
    assert mask.dtype == bool
    assert int(mask.sum()) == count
    # This smoke test intentionally does not claim budget convergence in 3 epochs.
