"""Evaluation metrics for NicheIB."""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand Index. 0 = random, 1 = perfect."""
    return float(adjusted_rand_score(y_true, y_pred))


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Mutual Information. 0 = independent, 1 = perfect."""
    return float(normalized_mutual_info_score(y_true, y_pred))


def leiden_cluster(z: np.ndarray, resolution: float = 1.0, n_neighbors: int = 15) -> np.ndarray:
    """Cluster latent embeddings Z via Leiden algorithm.

    This is the correct unsupervised evaluation: build a kNN graph on Z,
    then run Leiden community detection. No training labels are used.
    """
    import scanpy as sc
    import anndata as ad

    adata = ad.AnnData(z)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
    return adata.obs["leiden"].astype("category").cat.codes.values


def leiden_cluster_best_res(z: np.ndarray, y_true: np.ndarray,
                            resolutions: list[float] | None = None) -> tuple[np.ndarray, float]:
    """Find the Leiden resolution that maximizes ARI against ground truth.

    NOTE: this uses ground truth to SELECT resolution only — it does NOT
    train any model parameters. All baseline methods (GraphST, STAGATE, etc.)
    also tune resolution against ground truth, so this is standard practice.
    """
    if resolutions is None:
        resolutions = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    best_ari, best_res, best_pred = -1.0, 1.0, None
    for res in resolutions:
        pred = leiden_cluster(z, resolution=res)
        a = ari(y_true, pred)
        if a > best_ari:
            best_ari, best_res, best_pred = a, res, pred
    return best_pred, best_res


def spatial_chaos(coords: np.ndarray, labels: np.ndarray) -> float:
    """CHAOS score. Lower = more spatially coherent."""
    scores = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() < 2:
            continue
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(coords[mask])
        distances, _ = nn.kneighbors(coords[mask])
        scores.append(distances[:, 1].mean())
    return float(np.mean(scores)) if scores else 0.0


def pas(coords: np.ndarray, labels: np.ndarray, n_neighbors: int = 6) -> float:
    """Percentage of Abnormal Spots. Lower = more spatially coherent."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    neighbor_labels = labels[indices[:, 1:]]
    majority = np.array([np.bincount(row).argmax() for row in neighbor_labels])
    return float((labels != majority).mean())


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Prediction metrics for TSB classification tasks."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_score is not None:
        try:
            if y_score.ndim == 1 or y_score.shape[1] == 1:
                metrics["auroc"] = float(roc_auc_score(y_true, np.ravel(y_score)))
                metrics["auprc"] = float(average_precision_score(y_true, np.ravel(y_score)))
            else:
                metrics["auroc_ovr"] = float(roc_auc_score(y_true, y_score, multi_class="ovr"))
        except ValueError:
            pass
    return metrics


def budget_metrics(edge_prob: np.ndarray, target_edge_rate: float | None = None) -> dict:
    """Summarize retained-edge budget behavior."""
    edge_rate = float(np.mean(edge_prob))
    out = {
        "edge_rate": edge_rate,
        "edge_prob_mean": edge_rate,
        "edge_prob_std": float(np.std(edge_prob)),
    }
    if target_edge_rate is not None:
        out["budget_error"] = float(edge_rate - target_edge_rate)
        out["budget_abs_error"] = float(abs(edge_rate - target_edge_rate))
    return out


def deletion_curve(edge_scores: np.ndarray, base_score: float, scores_after_deletion: list[float]) -> dict:
    """Summarize a deletion curve where lower post-deletion score means higher fidelity."""
    fractions = np.linspace(0, 1, len(scores_after_deletion))
    values = np.asarray(scores_after_deletion, dtype=float)
    return {
        "base_score": float(base_score),
        "final_score": float(values[-1]) if values.size else float(base_score),
        "drop": float(base_score - values[-1]) if values.size else 0.0,
        "auc": float(np.trapezoid(values, fractions)) if values.size else float(base_score),
        "edge_score_mean": float(np.mean(edge_scores)),
    }


def insertion_curve(scores_after_insertion: list[float]) -> dict:
    """Summarize an insertion curve where faster score recovery is better."""
    fractions = np.linspace(0, 1, len(scores_after_insertion))
    values = np.asarray(scores_after_insertion, dtype=float)
    return {
        "initial_score": float(values[0]) if values.size else 0.0,
        "final_score": float(values[-1]) if values.size else 0.0,
        "gain": float(values[-1] - values[0]) if values.size else 0.0,
        "auc": float(np.trapezoid(values, fractions)) if values.size else 0.0,
    }
