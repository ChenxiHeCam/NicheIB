"""Targeted Spatial Bottleneck model."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from nicheib.gates import HardConcreteSampler
from nicheib.readout import GraphReadout

TaskLevel = Literal["node", "graph"]
TaskType = Literal["classification", "regression", "survival"]


class EdgeConditionedGate(nn.Module):
    """Computes target-relevant edge retention probabilities from endpoint states."""

    def __init__(self, node_dim: int, hidden_dim: int, target_dim: int = 0,
                 n_edge_types: int = 1, edge_type_dim: int = 8,
                 init_bias: float = 0.0):
        super().__init__()
        self.n_edge_types = n_edge_types
        in_dim = 4 * node_dim + target_dim
        if n_edge_types > 1:
            self.edge_type_emb = nn.Embedding(n_edge_types, edge_type_dim)
            in_dim += edge_type_dim
        else:
            self.edge_type_emb = None
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if init_bias != 0.0:
            nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        target_context: torch.Tensor = None,
        edge_type: torch.Tensor = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        edge_feat = [x[src], x[dst], (x[src] - x[dst]).abs(), x[src] * x[dst]]
        if target_context is not None:
            if target_context.dim() == 1:
                target_context = target_context.unsqueeze(0).expand(src.shape[0], -1)
            edge_feat.append(target_context)
        if self.edge_type_emb is not None and edge_type is not None:
            edge_feat.append(self.edge_type_emb(edge_type))
        return self.net(torch.cat(edge_feat, dim=-1)).squeeze(-1)


GateMode = Literal["multiplicative", "additive"]


class GatedGraphConv(MessagePassing):
    """GAT-style message passing with externally supplied edge masks."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 gate_mode: GateMode = "multiplicative"):
        super().__init__(aggr="add", node_dim=0)
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.gate_mode = gate_mode
        self.lin = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.randn(1, heads, self.head_dim) * 0.01)
        self.att_dst = nn.Parameter(torch.randn(1, heads, self.head_dim) * 0.01)
        self.skip = nn.Linear(in_dim, out_dim)
        if gate_mode == "additive":
            self.gate_proj = nn.Linear(1, heads, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_mask: torch.Tensor):
        x_proj = self.lin(x).view(-1, self.heads, self.head_dim)
        alpha_src = (x_proj * self.att_src).sum(dim=-1)
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)
        out = self.propagate(
            edge_index,
            x=x_proj,
            alpha_src=alpha_src,
            alpha_dst=alpha_dst,
            edge_mask=edge_mask,
            size=None,
        )
        return out.reshape(-1, self.heads * self.head_dim) + self.skip(x)

    def message(
        self,
        x_j: torch.Tensor,
        alpha_src_i: torch.Tensor,
        alpha_dst_j: torch.Tensor,
        edge_mask: torch.Tensor,
        index: torch.Tensor,
        ptr,
        size_i: int,
    ) -> torch.Tensor:
        if self.gate_mode == "additive":
            gate_bias = self.gate_proj(edge_mask.view(-1, 1))
            alpha = F.leaky_relu(alpha_src_i + alpha_dst_j + gate_bias, negative_slope=0.2)
            alpha = softmax(alpha, index, ptr, size_i)
            return x_j * alpha.unsqueeze(-1)
        else:
            alpha = F.leaky_relu(alpha_src_i + alpha_dst_j, negative_slope=0.2)
            alpha = softmax(alpha, index, ptr, size_i)
            return x_j * alpha.unsqueeze(-1) * edge_mask.view(-1, 1, 1)


class TargetedSpatialBottleneck(nn.Module):
    """Sparse target-conditioned spatial graph encoder.

    The model learns edge gates from endpoint embeddings and optimizes explicit
    expected-L0 budget losses. It supports node-level and graph-level targets.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        output_dim: int = 2,
        task_level: TaskLevel = "node",
        task_type: TaskType = "classification",
        heads: int = 4,
        dropout: float = 0.1,
        use_xib: bool = True,
        readout: str = "mean",
        target_context_dim: int = 0,
        hc_temperature: float = 2.0 / 3.0,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
        n_edge_types: int = 1,
        edge_type_dim: int = 8,
        gate_init_bias: float = 3.0,
        gate_mode: GateMode = "multiplicative",
    ):
        super().__init__()
        self.task_level = task_level
        self.task_type = task_type
        self.dropout = dropout
        self.use_xib = use_xib

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gate1 = EdgeConditionedGate(hidden_dim, hidden_dim, target_context_dim,
                                         n_edge_types, edge_type_dim,
                                         init_bias=gate_init_bias)
        self.conv1 = GatedGraphConv(hidden_dim, hidden_dim, heads=heads,
                                    gate_mode=gate_mode)
        self.gate2 = EdgeConditionedGate(hidden_dim, hidden_dim, target_context_dim,
                                         n_edge_types, edge_type_dim,
                                         init_bias=gate_init_bias)
        self.conv2 = GatedGraphConv(hidden_dim, latent_dim, heads=heads,
                                    gate_mode=gate_mode)
        self.hc_sampler = HardConcreteSampler(hc_temperature, hc_gamma, hc_zeta)

        if use_xib:
            self.mu_head = nn.Linear(latent_dim, latent_dim)
            self.logvar_head = nn.Linear(latent_dim, latent_dim)

        if task_level == "graph":
            self.readout = GraphReadout(latent_dim, mode=readout)
        elif task_level != "node":
            raise ValueError(f"Unknown task level: {task_level}")

        if task_type == "classification":
            self.head = nn.Linear(latent_dim, output_dim)
        elif task_type in {"regression", "survival"}:
            self.head = nn.Linear(latent_dim, output_dim)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _edge_mask(self, logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Hard Concrete gating to edge logits.

        Returns:
            edge_mask: Sampled (training) or deterministic (eval) gate values.
                       Exact zeros are possible thanks to the HC stretch+clamp.
            edge_prob: Deterministic gate values for visualization/fidelity.
            expected_l0: Per-edge probability of being non-zero (for budget loss).
        """
        edge_mask, expected_l0 = self.hc_sampler(logits)
        edge_prob = self.hc_sampler.deterministic(logits)
        return edge_mask, edge_prob, expected_l0

    def _xib(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_xib:
            return z, z.new_tensor(0.0)
        mu = self.mu_head(z)
        logvar = self.logvar_head(z).clamp(-10.0, 10.0)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        xib = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        return z, xib

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.Tensor | None = None,
        target_context: torch.Tensor = None,
        temperature: float = 0.5,
        override_edge_mask: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> dict:
        h = F.relu(self.encoder(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        if override_edge_mask is not None:
            # Bypass gate computation — used for fidelity evaluation.
            mask1 = mask2 = override_edge_mask
            prob = override_edge_mask
            el0 = override_edge_mask
        else:
            logits1 = self.gate1(h, edge_index, target_context, edge_type)
            mask1, prob1, el0_1 = self._edge_mask(logits1, temperature)

        h = F.relu(self.conv1(h, edge_index, mask1))
        h = F.dropout(h, p=self.dropout, training=self.training)

        if override_edge_mask is None:
            logits2 = self.gate2(h, edge_index, target_context, edge_type)
            mask2, prob2, el0_2 = self._edge_mask(logits2, temperature)
            prob = (prob1 + prob2) / 2
            el0 = (el0_1 + el0_2) / 2

        z = self.conv2(h, edge_index, mask2)
        z, xib = self._xib(z)

        pred_input = z
        z_graph = None
        if self.task_level == "graph":
            z_graph = self.readout(z, batch)
            pred_input = z_graph

        raw_pred = self.head(pred_input)
        out = {
            "z_node": z,
            "z_graph": z_graph,
            "edge_prob": prob,
            "edge_mask": (mask1 + mask2) / 2,
            "expected_l0": el0,
            "edge_rate": el0.mean(),
            "xib": xib,
        }
        if self.task_type == "classification":
            out["logits"] = raw_pred
        elif self.task_type == "survival":
            out["risk"] = raw_pred.squeeze(-1)
            out["prediction"] = raw_pred.squeeze(-1)
        else:
            out["prediction"] = raw_pred
        return out
