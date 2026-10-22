"""Sparse gates for Targeted Spatial Bottleneck."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteGate(nn.Module):
    """Hard Concrete / L0 gate for sparse edge selection.

    The gate stores one learnable log-alpha parameter per edge. During training it
    samples differentiable masks; during evaluation it returns deterministic
    clipped probabilities. The expected L0 value gives an explicit sparsity
    budget, unlike KL-to-Bernoulli(0.5) gates.
    """

    def __init__(
        self,
        n_gates: int,
        init_keep_prob: float = 0.5,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ):
        super().__init__()
        if not 0 < init_keep_prob < 1:
            raise ValueError("init_keep_prob must be in (0, 1)")
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta
        init_logit = math.log(init_keep_prob) - math.log1p(-init_keep_prob)
        self.log_alpha = nn.Parameter(torch.full((n_gates,), init_logit))

    def sample(self) -> torch.Tensor:
        """Sample relaxed hard-concrete masks."""
        u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.temperature)
        stretched = s * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def deterministic(self) -> torch.Tensor:
        """Return deterministic gate values for evaluation."""
        s = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
        return s.clamp(0.0, 1.0)

    def expected_l0(self) -> torch.Tensor:
        """Probability that each gate is non-zero."""
        offset = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - offset)

    def forward(self) -> torch.Tensor:
        return self.sample() if self.training else self.deterministic()

    def edge_rate(self) -> torch.Tensor:
        return self.expected_l0().mean()


class HardConcreteSampler(nn.Module):
    """Stateless Hard Concrete sampler for input-dependent logits.

    Unlike HardConcreteGate which stores fixed per-edge log-alpha parameters,
    this module takes dynamic logits (e.g. from EdgeConditionedGate) and applies
    the Hard Concrete distribution mechanics: stretch to [gamma, zeta], then
    hard-clamp to [0, 1].  This produces exact zeros and ones, enabling true
    sparsity.  The expected L0 correctly accounts for the probability that the
    stretched sample lands above 0.

    Reference: Louizos, Welling & Kingma, "Learning Sparse Neural Networks
    through L0 Regularization", ICLR 2018.
    """

    def __init__(
        self,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample relaxed hard-concrete masks with exact zeros."""
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid(
            (torch.log(u) - torch.log1p(-u) + logits) / self.temperature
        )
        stretched = s * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def deterministic(self, logits: torch.Tensor) -> torch.Tensor:
        """Return deterministic gate values for evaluation."""
        s = torch.sigmoid(logits) * (self.zeta - self.gamma) + self.gamma
        return s.clamp(0.0, 1.0)

    def expected_l0(self, logits: torch.Tensor) -> torch.Tensor:
        """Probability that each gate is non-zero (correct L0 estimate)."""
        offset = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(logits - offset)

    def forward(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mask, expected_l0)."""
        mask = self.sample(logits) if self.training else self.deterministic(logits)
        el0 = self.expected_l0(logits)
        return mask, el0


def budget_loss(
    expected_l0: torch.Tensor,
    target_rate: float | None = None,
    mode: str = "hinge",
) -> torch.Tensor:
    """Compute sparsity or target-budget loss from expected L0 probabilities."""
    edge_rate = expected_l0.mean()
    if target_rate is None:
        return edge_rate
    target = torch.as_tensor(target_rate, dtype=edge_rate.dtype, device=edge_rate.device)
    if mode == "hinge":
        return F.relu(edge_rate - target).pow(2)
    if mode == "barrier":
        return edge_rate + F.relu(edge_rate - target).pow(2)
    if mode == "squared":
        return (edge_rate - target).pow(2)
    if mode == "absolute":
        return (edge_rate - target).abs()
    raise ValueError(f"Unknown budget loss mode: {mode}")
