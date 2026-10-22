"""Readout layers for node-level and graph-level TSB tasks."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class GatedAttentionReadout(nn.Module):
    """Gated MIL-style graph readout for sample-level targets."""

    def __init__(self, in_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.value = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            batch = x.new_zeros(x.shape[0], dtype=torch.long)
        h = self.value(x) * self.gate(x)
        logits = self.score(h).squeeze(-1)
        weights = torch.zeros_like(logits)
        for graph_id in batch.unique(sorted=True):
            mask = batch == graph_id
            weights[mask] = torch.softmax(logits[mask], dim=0)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        out = x.new_zeros(n_graphs, x.shape[-1])
        out.index_add_(0, batch, x * weights.unsqueeze(-1))
        return out


class GraphReadout(nn.Module):
    """Graph readout wrapper with mean or gated-attention pooling."""

    def __init__(self, in_dim: int, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.attention = GatedAttentionReadout(in_dim)
        elif mode != "mean":
            raise ValueError(f"Unknown readout mode: {mode}")

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            batch = x.new_zeros(x.shape[0], dtype=torch.long)
        if self.mode == "mean":
            return global_mean_pool(x, batch)
        return self.attention(x, batch)
