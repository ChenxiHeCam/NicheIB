"""Small baselines for TSB experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except ModuleNotFoundError:
    GCNConv = None

    def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ModuleNotFoundError("torch_geometric is required for graph-level pooling")


class FeatureMLP(nn.Module):
    """Feature-only baseline without spatial message passing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, task_level: str = "node"):
        super().__init__()
        self.task_level = task_level
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> dict:
        if self.task_level == "graph":
            if batch is None:
                batch = x.new_zeros(x.shape[0], dtype=torch.long)
            x = global_mean_pool(x, batch)
        return {"logits": self.net(x)}


class PlainGCN(nn.Module):
    """No-bottleneck spatial GNN baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        task_level: str = "node",
        dropout: float = 0.1,
    ):
        super().__init__()
        if GCNConv is None:
            raise ModuleNotFoundError("torch_geometric is required for PlainGCN")
        self.task_level = task_level
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.Tensor | None = None,
    ) -> dict:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        if self.task_level == "graph":
            if batch is None:
                batch = x.new_zeros(x.shape[0], dtype=torch.long)
            h = global_mean_pool(h, batch)
        return {"logits": self.head(h), "z_node": h}


class PlainGAT(nn.Module):
    """GAT baseline with same architecture as TSB but without edge gates.

    Uses GatedGraphConv with all-ones edge mask, matching TSB's 4-head
    attention + skip connections + encoder + head structure. This isolates
    the contribution of edge gating from architectural differences.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        heads: int = 4,
        task_level: str = "node",
        dropout: float = 0.1,
    ):
        super().__init__()
        from nicheib.tsb import GatedGraphConv as _GGC

        self.task_level = task_level
        self.dropout = dropout
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.conv1 = _GGC(hidden_dim, hidden_dim, heads=heads, gate_mode="multiplicative")
        self.conv2 = _GGC(hidden_dim, latent_dim, heads=heads, gate_mode="multiplicative")
        self.head = nn.Linear(latent_dim, output_dim)
        if task_level == "graph":
            self._pool = True
        else:
            self._pool = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.Tensor | None = None,
    ) -> dict:
        ones = x.new_ones(edge_index.shape[1])
        h = F.relu(self.encoder(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv1(h, edge_index, ones))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, ones)
        if self._pool:
            if batch is None:
                batch = x.new_zeros(x.shape[0], dtype=torch.long)
            h = global_mean_pool(h, batch)
        return {"logits": self.head(h), "z_node": h}


def random_edge_scores(n_edges: int, device: torch.device | None = None) -> torch.Tensor:
    """Random matched-budget explanation baseline."""
    return torch.rand(n_edges, device=device)


def matched_budget_edge_mask(edge_scores: torch.Tensor, target_edge_rate: float) -> torch.Tensor:
    """Keep the exact top-scoring edge budget as a boolean mask."""
    n_edges = edge_scores.numel()
    if target_edge_rate <= 0 or n_edges == 0:
        return torch.zeros(n_edges, dtype=torch.bool, device=edge_scores.device)
    if target_edge_rate >= 1:
        return torch.ones(n_edges, dtype=torch.bool, device=edge_scores.device)

    k = int(round(target_edge_rate * n_edges))
    k = max(1, min(k, n_edges))
    top_idx = torch.topk(edge_scores, k=k, largest=True).indices
    mask = torch.zeros(n_edges, dtype=torch.bool, device=edge_scores.device)
    mask[top_idx] = True
    return mask


def prune_edge_index(edge_index: torch.LongTensor, edge_mask: torch.Tensor) -> torch.LongTensor:
    """Select edge columns according to a boolean edge mask."""
    if edge_mask.dtype != torch.bool:
        edge_mask = edge_mask.bool()
    return edge_index[:, edge_mask]
