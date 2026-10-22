import pytest
import torch

pytest.importorskip("torch_geometric")

from nicheib.loss import tsb_loss
from nicheib.tsb import TargetedSpatialBottleneck


def test_tsb_loss_is_finite():
    x = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    model = TargetedSpatialBottleneck(4, hidden_dim=16, latent_dim=8, output_dim=2)
    out = model(x, edge_index)
    loss = tsb_loss(out, y, target_edge_rate=0.2, lambda_l0=0.1)
    assert torch.isfinite(loss["total"])
    assert torch.isfinite(loss["task"])
    assert torch.isfinite(loss["l0"])
