import pytest
import torch

pytest.importorskip("torch_geometric")

from nicheib.tsb import TargetedSpatialBottleneck


def test_tsb_node_forward():
    x = torch.randn(12, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]])
    model = TargetedSpatialBottleneck(5, hidden_dim=16, latent_dim=8, output_dim=3)
    out = model(x, edge_index)
    assert out["logits"].shape == (12, 3)
    assert out["z_node"].shape == (12, 8)
    assert out["edge_prob"].numel() == edge_index.shape[1]
    assert torch.isfinite(out["edge_rate"])


def test_tsb_graph_forward():
    x = torch.randn(12, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 6, 7, 8, 9], [1, 2, 3, 4, 7, 8, 9, 10]])
    batch = torch.tensor([0] * 6 + [1] * 6)
    model = TargetedSpatialBottleneck(
        5,
        hidden_dim=16,
        latent_dim=8,
        output_dim=2,
        task_level="graph",
        readout="attention",
    )
    out = model(x, edge_index, batch=batch)
    assert out["logits"].shape == (2, 2)
    assert out["z_graph"].shape == (2, 8)
