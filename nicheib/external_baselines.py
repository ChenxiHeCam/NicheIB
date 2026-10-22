"""Baseline edge-importance scoring methods for fidelity comparison.

Provides alternative edge-score sources for deletion/insertion fidelity
evaluation. All functions return a numpy array of per-edge scores compatible
with evaluate_fidelity.run_deletion_curve / run_insertion_curve.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def random_edge_scores(n_edges: int, seed: int = 0) -> np.ndarray:
    """Uniform random edge importance (null baseline)."""
    rng = np.random.RandomState(seed)
    return rng.rand(n_edges).astype(np.float32)


def degree_edge_scores(edge_index: torch.Tensor) -> np.ndarray:
    """Score each edge by sum of endpoint degrees (structural baseline)."""
    src, dst = edge_index
    n_nodes = int(edge_index.max()) + 1
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    deg.scatter_add_(0, src, torch.ones(src.shape[0]))
    scores = (deg[src] + deg[dst]).numpy()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


@torch.no_grad()
def attention_edge_scores(model, data) -> np.ndarray:
    """Extract attention weights from GatedGraphConv layers as edge scores.

    Works with TargetedSpatialBottleneck (uses conv1/conv2 attention coefficients).
    Falls back to uniform if no attention coefficients are available.
    """
    model.eval()
    # Run a forward pass to populate intermediate computations
    h = F.relu(model.encoder(data.x))
    logits1 = model.gate1(h, data.edge_index)
    # We skip the HC sampler; just use sigmoid of raw logits as a score
    score1 = torch.sigmoid(logits1)
    h = F.relu(model.conv1(h, data.edge_index, torch.ones(data.edge_index.shape[1], device=data.x.device)))
    logits2 = model.gate2(h, data.edge_index)
    score2 = torch.sigmoid(logits2)
    scores = ((score1 + score2) / 2).cpu().numpy()
    return scores


def gradient_edge_scores(
    model,
    data,
    task_type: str = "classification",
) -> np.ndarray:
    """Gradient-based edge saliency: |d(loss)/d(edge_mask)|.

    Runs a single forward+backward with a differentiable all-ones mask
    and measures how much each edge affects the loss.
    """
    model.eval()
    n_edges = data.edge_index.shape[1]
    edge_mask = torch.ones(n_edges, device=data.x.device, requires_grad=True)

    out = model(
        data.x, data.edge_index,
        batch=getattr(data, "batch", None),
        override_edge_mask=edge_mask,
    )

    # Compute a proxy loss
    if task_type == "classification":
        y = data.y
        valid = getattr(data, "valid_mask", None)
        logits = out["logits"]
        if valid is not None:
            logits = logits[valid]
            y = y[valid]
        loss = F.cross_entropy(logits, y.long())
    elif task_type == "regression":
        pred = out["prediction"].squeeze()
        target = data.y.float().squeeze()
        valid = getattr(data, "valid_mask", None)
        if valid is not None:
            pred = pred[valid]
            target = target[valid]
        loss = F.mse_loss(pred, target)
    else:
        raise NotImplementedError(f"gradient_edge_scores not implemented for {task_type}")

    loss.backward()
    scores = edge_mask.grad.abs().cpu().numpy()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def gnnexplainer_edge_scores(
    model,
    data,
    task_type: str = "classification",
    epochs: int = 100,
    lr: float = 0.01,
) -> np.ndarray:
    """GNNExplainer-style edge mask optimization.

    Learns a soft edge mask that preserves the model's original prediction
    while being as sparse as possible. Uses the standard mutual-information
    objective from Ying et al., NeurIPS 2019.
    """
    model.eval()
    n_edges = data.edge_index.shape[1]

    # Get original prediction as the target
    with torch.no_grad():
        out_orig = model(
            data.x, data.edge_index,
            batch=getattr(data, "batch", None),
        )
        if task_type == "classification":
            target_probs = F.softmax(out_orig["logits"], dim=-1)
        else:
            target_pred = out_orig["prediction"]

    # Learnable edge mask
    mask_logits = torch.zeros(n_edges, device=data.x.device, requires_grad=True)
    optimizer = torch.optim.Adam([mask_logits], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        mask = torch.sigmoid(mask_logits)
        out = model(
            data.x, data.edge_index,
            batch=getattr(data, "batch", None),
            override_edge_mask=mask,
        )

        # Prediction fidelity: KL divergence from original
        if task_type == "classification":
            pred_probs = F.log_softmax(out["logits"], dim=-1)
            valid = getattr(data, "valid_mask", None)
            if valid is not None:
                fidelity_loss = F.kl_div(pred_probs[valid], target_probs[valid],
                                         log_target=False, reduction="batchmean")
            else:
                fidelity_loss = F.kl_div(pred_probs, target_probs,
                                         log_target=False, reduction="batchmean")
        else:
            fidelity_loss = F.mse_loss(out["prediction"], target_pred)

        # Sparsity: encourage mask to be sparse
        sparsity_loss = mask.mean()

        # Entropy: encourage mask to be discrete (0 or 1)
        ent = -mask * torch.log(mask + 1e-8) - (1 - mask) * torch.log(1 - mask + 1e-8)
        entropy_loss = ent.mean()

        loss = fidelity_loss + 0.01 * sparsity_loss + 0.01 * entropy_loss
        loss.backward()
        optimizer.step()

    scores = torch.sigmoid(mask_logits).detach().cpu().numpy()
    return scores
