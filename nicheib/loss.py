"""Loss functions for Targeted Spatial Bottleneck."""

import torch
import torch.nn.functional as F

from nicheib.gates import budget_loss


def cox_partial_likelihood(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood for graph-level survival tasks."""
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order].float()
    log_cumsum_hazard = torch.logcumsumexp(risk, dim=0)
    observed = event.sum().clamp_min(1.0)
    return -((risk - log_cumsum_hazard) * event).sum() / observed


def _pred_loss(output: dict, target: torch.Tensor, task_type: str,
               valid_mask: torch.Tensor | None, event: torch.Tensor | None) -> torch.Tensor:
    """Compute task prediction loss (shared by tsb_loss and counterfactual)."""
    if task_type == "classification":
        logits = output["logits"]
        if valid_mask is not None:
            logits = logits[valid_mask]
            target = target[valid_mask]
        return F.cross_entropy(logits, target.long())
    elif task_type == "regression":
        prediction = output["prediction"]
        if valid_mask is not None:
            prediction = prediction[valid_mask]
            target = target[valid_mask]
        return F.mse_loss(prediction.squeeze(), target.float().squeeze())
    elif task_type == "survival":
        if event is None:
            raise ValueError("event must be provided for survival tasks")
        return cox_partial_likelihood(output["risk"], target.float(), event.float())
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def tsb_loss(
    output: dict,
    target: torch.Tensor,
    task_type: str = "classification",
    valid_mask: torch.Tensor | None = None,
    lambda_l0: float = 1.0,
    target_edge_rate: float | None = None,
    beta_xib: float = 0.0,
    budget_mode: str = "barrier",
    event: torch.Tensor | None = None,
    edge_target: torch.Tensor | None = None,
    lambda_edge: float = 0.0,
    # Stability regularization
    output_aug: dict | None = None,
    lambda_stab: float = 0.0,
    # Counterfactual fidelity
    output_cf: dict | None = None,
    lambda_cf: float = 0.0,
    cf_margin: float = 0.1,
    cf_target: torch.Tensor | None = None,
) -> dict:
    """Targeted Spatial Bottleneck loss.

    Combines task prediction with:
      - Expected-L0 edge budget (Hard Concrete)
      - Optional feature bottleneck (XIB)
      - Optional edge supervision
      - Optional stability regularization (gate consistency under perturbation)
      - Optional counterfactual fidelity (deleting top edges should hurt prediction)
    """
    pred_loss = _pred_loss(output, target, task_type, valid_mask, event)

    l0 = budget_loss(output["expected_l0"], target_rate=target_edge_rate, mode=budget_mode)
    xib = output.get("xib", pred_loss.new_tensor(0.0))
    edge_supervision = pred_loss.new_tensor(0.0)
    if edge_target is not None and lambda_edge > 0:
        edge_supervision = F.binary_cross_entropy(
            output["edge_prob"].clamp(1e-6, 1 - 1e-6), edge_target.float()
        )

    total = pred_loss + lambda_l0 * l0 + beta_xib * xib + lambda_edge * edge_supervision

    # Stability: encourage consistent edge gates across augmented forward passes
    stab_loss = pred_loss.new_tensor(0.0)
    if output_aug is not None and lambda_stab > 0:
        stab_loss = F.mse_loss(output["edge_prob"], output_aug["edge_prob"])
        total = total + lambda_stab * stab_loss

    # Counterfactual fidelity: deleting high-score edges should increase loss
    cf_loss = pred_loss.new_tensor(0.0)
    if output_cf is not None and lambda_cf > 0:
        cf_pred_loss = _pred_loss(
            output_cf,
            cf_target if cf_target is not None else target,
            task_type, valid_mask, event,
        )
        # We want: cf_pred_loss - pred_loss >= margin (deleting top edges hurts)
        cf_loss = F.relu(cf_margin - (cf_pred_loss - pred_loss))
        total = total + lambda_cf * cf_loss

    return {
        "total": total,
        "task": pred_loss,
        "l0": l0,
        "xib": xib,
        "edge_supervision": edge_supervision,
        "stability": stab_loss,
        "counterfactual": cf_loss,
        "edge_rate": output["edge_rate"],
    }
