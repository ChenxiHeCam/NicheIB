'Psoriasis graph-level classification benchmark for Targeted Spatial Bottleneck.'

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nicheib.baselines import (
    FeatureMLP,
    PlainGCN,
    matched_budget_edge_mask,
    prune_edge_index,
    random_edge_scores,
)
from nicheib.data import load_psoriasis
from nicheib.eval import budget_metrics, classification_metrics
from nicheib.loss import tsb_loss

try:
    from torch_geometric.data import Batch
    from nicheib.tsb import TargetedSpatialBottleneck
except ModuleNotFoundError:
    Batch = None
    TargetedSpatialBottleneck = None


def _require_pyg():
    if Batch is None or TargetedSpatialBottleneck is None:
        raise ModuleNotFoundError("torch_geometric required")


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _graph_labels(graphs: list) -> np.ndarray:
    return np.asarray([int(g.y.item()) for g in graphs], dtype=np.int64)


def _batch(graphs: list, device: torch.device):
    return Batch.from_data_list(graphs).to(device)


def _batch_y(batch) -> torch.Tensor:
    return batch.y.view(-1).long()


def _classification_record(y, logits):
    prob = logits.softmax(dim=-1).detach().cpu().numpy()
    pred = prob.argmax(axis=1)
    y_true = y.detach().cpu().numpy()
    return classification_metrics(y_true, pred, prob)


def _make_lopo_splits(graphs: list) -> list[dict]:
    """Leave-one-patient-out splits."""
    patients = [str(getattr(g, "patient", "unknown")) for g in graphs]
    unique_patients = sorted(set(patients))
    splits = []
    for held_out in unique_patients:
        test_idx = [i for i, p in enumerate(patients) if p == held_out]
        train_idx = [i for i, p in enumerate(patients) if p != held_out]
        if not test_idx or not train_idx:
            continue
        # Check that train has both classes
        train_labels = set(_graph_labels([graphs[i] for i in train_idx]))
        if len(train_labels) < 2:
            continue
        splits.append({
            "split_id": f"lopo_{held_out}",
            "train_idx": train_idx,
            "test_idx": test_idx,
            "held_out_patient": held_out,
        })
    return splits


def _fit_graph_model(
    model, train_graphs, epochs, lr, weight_decay, device,
    temperature=None, lambda_l0=None, target_edge_rate=None,
    beta_xib=0.0, budget_mode="squared", uses_edges=True,
    warmup_fraction=0.2,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_train = -1.0
    train_batch = _batch(train_graphs, device)
    y = _batch_y(train_batch)
    warmup_epochs = max(1, int(epochs * warmup_fraction))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if temperature is None:
            if uses_edges:
                out = model(train_batch.x, train_batch.edge_index, train_batch.batch)
            else:
                out = model(train_batch.x, train_batch.batch)
            loss = F.cross_entropy(out["logits"], y)
        else:
            warmup_scale = min(1.0, (epoch + 1) / warmup_epochs)
            effective_lambda = float(lambda_l0) * warmup_scale
            out = model(
                train_batch.x, train_batch.edge_index,
                batch=train_batch.batch, temperature=temperature,
            )
            loss = tsb_loss(
                out, y, task_type="classification",
                lambda_l0=effective_lambda,
                target_edge_rate=target_edge_rate,
                beta_xib=beta_xib, budget_mode=budget_mode,
            )["total"]
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)
            train_acc = (pred == y).float().mean().item()
            if train_acc > best_train:
                best_train = train_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_train)


def _evaluate_model(model, graphs, device, temperature=None, uses_edges=True):
    batch = _batch(graphs, device)
    model.eval()
    with torch.no_grad():
        if temperature is None:
            if uses_edges:
                out = model(batch.x, batch.edge_index, batch.batch)
            else:
                out = model(batch.x, batch.batch)
        else:
            out = model(batch.x, batch.edge_index, batch=batch.batch, temperature=temperature)
    return out, _batch_y(batch)


def _train_tsb(args, graphs, train_graphs, test_graphs, seed, split_id,
               target_edge_rate, lambda_l0):
    _seed_everything(seed)
    device = torch.device(args.device)
    model = TargetedSpatialBottleneck(
        input_dim=graphs[0].x.shape[1],
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        output_dim=args.n_classes, task_level="graph",
        task_type="classification", dropout=args.dropout,
        use_xib=False, gate_init_bias=args.gate_init_bias,
    ).to(device)
    model, best_train = _fit_graph_model(
        model, train_graphs, args.epochs, args.lr, args.weight_decay, device,
        temperature=args.temperature, lambda_l0=lambda_l0,
        target_edge_rate=target_edge_rate, budget_mode=args.budget_mode,
        warmup_fraction=args.warmup_fraction,
    )
    out, y = _evaluate_model(model, test_graphs, device, temperature=args.temperature)

    record = {
        "method": "tsb", "seed": seed, "split_id": split_id,
        "target_edge_rate": target_edge_rate, "lambda_l0": lambda_l0,
        "n_train": len(train_graphs), "n_test": len(test_graphs),
        "test_samples": [getattr(g, "sample_id", "?") for g in test_graphs],
        "test_conditions": [getattr(g, "condition", "?") for g in test_graphs],
        "best_train_accuracy": best_train,
    }
    record.update(_classification_record(y, out["logits"]))
    record.update(budget_metrics(out["edge_prob"].detach().cpu().numpy(), target_edge_rate))
    return record


def _train_plain_gcn(args, graphs, train_graphs, test_graphs, seed, split_id):
    _seed_everything(seed)
    device = torch.device(args.device)
    model = PlainGCN(
        graphs[0].x.shape[1], args.hidden_dim, args.n_classes,
        task_level="graph", dropout=args.dropout,
    ).to(device)
    model, best_train = _fit_graph_model(
        model, train_graphs, args.epochs, args.lr, args.weight_decay, device,
    )
    out, y = _evaluate_model(model, test_graphs, device)
    record = {
        "method": "plain_gcn", "seed": seed, "split_id": split_id,
        "n_train": len(train_graphs), "n_test": len(test_graphs),
        "test_samples": [getattr(g, "sample_id", "?") for g in test_graphs],
        "best_train_accuracy": best_train,
    }
    record.update(_classification_record(y, out["logits"]))
    return record


def _train_feature_mlp(args, graphs, train_graphs, test_graphs, seed, split_id):
    _seed_everything(seed)
    device = torch.device(args.device)
    model = FeatureMLP(
        graphs[0].x.shape[1], args.hidden_dim, args.n_classes, task_level="graph",
    ).to(device)
    model, best_train = _fit_graph_model(
        model, train_graphs, args.epochs, args.lr, args.weight_decay, device,
        uses_edges=False,
    )
    out, y = _evaluate_model(model, test_graphs, device, uses_edges=False)
    record = {
        "method": "feature_mlp", "seed": seed, "split_id": split_id,
        "n_train": len(train_graphs), "n_test": len(test_graphs),
        "test_samples": [getattr(g, "sample_id", "?") for g in test_graphs],
        "best_train_accuracy": best_train,
    }
    record.update(_classification_record(y, out["logits"]))
    return record


def _train_random_gcn(args, graphs, train_graphs, test_graphs, seed, split_id,
                      target_edge_rate, repeat):
    _seed_everything(seed + 1009 * (repeat + 1))
    device = torch.device(args.device)

    def _prune(gs):
        pruned = []
        for g in gs:
            scores = random_edge_scores(g.edge_index.shape[1])
            mask = matched_budget_edge_mask(scores, target_edge_rate).cpu()
            gc = copy.copy(g)
            gc.edge_index = prune_edge_index(g.edge_index, mask)
            if getattr(g, "edge_attr", None) is not None:
                gc.edge_attr = g.edge_attr[mask]
            pruned.append(gc)
        return pruned

    train_pruned = _prune(train_graphs)
    test_pruned = _prune(test_graphs)
    model = PlainGCN(
        graphs[0].x.shape[1], args.hidden_dim, args.n_classes,
        task_level="graph", dropout=args.dropout,
    ).to(device)
    model, best_train = _fit_graph_model(
        model, train_pruned, args.epochs, args.lr, args.weight_decay, device,
    )
    out, y = _evaluate_model(model, test_pruned, device)
    record = {
        "method": "random_gcn", "seed": seed, "split_id": split_id,
        "target_edge_rate": target_edge_rate, "random_repeat": repeat,
        "n_train": len(train_graphs), "n_test": len(test_graphs),
        "best_train_accuracy": best_train,
    }
    record.update(_classification_record(y, out["logits"]))
    return record


def _summarize(records):
    groups = defaultdict(list)
    for r in records:
        key = (r.get("method"), r.get("target_edge_rate"))
        groups[key].append(r)

    summary = []
    for (method, ter), vals in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {"method": method, "target_edge_rate": ter, "n": len(vals)}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "auroc_ovr", "edge_rate"]:
            present = [float(v[metric]) for v in vals if metric in v and v[metric] is not None]
            if present:
                row[f"{metric}_mean"] = float(np.mean(present))
                row[f"{metric}_std"] = float(np.std(present))
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--label_mode", choices=["binary", "3class"], default="binary")
    parser.add_argument("--n_neighbors", type=int, default=6)
    parser.add_argument("--n_hvgs", type=int, default=1000)
    parser.add_argument("--n_pcs", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lambda_l0", type=float, default=5.0)
    parser.add_argument("--target_edge_rate", type=float, default=0.2)
    parser.add_argument("--budget_mode", default="squared")
    parser.add_argument("--gate_init_bias", type=float, default=3.0)
    parser.add_argument("--warmup_fraction", type=float, default=0.2)
    parser.add_argument("--random_repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+",
                        default=["tsb", "plain_gcn", "feature_mlp", "random_gcn"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--permute_labels_seed", type=int, default=None,
                        help="Shuffle graph labels before splitting, to build a "
                             "permutation null for the n=23 cohort")
    parser.add_argument("--permute_scope", choices=["global", "within_patient"],
                        default="global",
                        help="'global' shuffles labels across the whole cohort; "
                             "'within_patient' shuffles only among sections of the "
                             "same patient, which is the exchangeability the paired "
                             "lesional/non-lesional design actually implies")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading psoriasis data...", flush=True)
    graphs = load_psoriasis(
        str(args.data_root), n_neighbors=args.n_neighbors,
        n_hvgs=args.n_hvgs, n_pcs=args.n_pcs, label_mode=args.label_mode,
    )
    permutation_info = {"scope": None, "seed": args.permute_labels_seed}
    if args.permute_labels_seed is not None:
        # Label-permutation null: the LOPO pipeline is left untouched, only the
        # section-to-label assignment is destroyed, so the resulting score
        # distribution is what this cohort size yields under no real signal.
        rng = np.random.default_rng(args.permute_labels_seed)
        labels = np.asarray([int(g.y.item()) for g in graphs], dtype=np.int64)
        patients = np.asarray([str(getattr(g, "patient", "unknown")) for g in graphs])
        if args.permute_scope == "within_patient":
            # Lesional and non-lesional sections are paired biopsies from the same
            # patient, so under H0 a section's label is exchangeable only with the
            # other sections of that patient. A global shuffle additionally destroys
            # each patient's class composition, which changes what leave-one-patient-out
            # is even asked to do and so mis-calibrates the null.
            shuffled = labels.copy()
            for p in np.unique(patients):
                idx = np.flatnonzero(patients == p)
                shuffled[idx] = rng.permutation(labels[idx])
        else:
            shuffled = rng.permutation(labels)
        for g, lab in zip(graphs, shuffled):
            g.y = torch.tensor([int(lab)], dtype=torch.long)
        permutation_info.update({
            "scope": args.permute_scope,
            "original_labels": [int(v) for v in labels],
            "permuted_labels": [int(v) for v in shuffled],
            "patients": [str(p) for p in patients],
            "n_moved": int((labels != shuffled).sum()),
        })
        print(f"PERMUTED labels ({args.permute_scope}, seed={args.permute_labels_seed}): "
              f"{list(shuffled)} ({permutation_info['n_moved']} of {len(labels)} moved)",
              flush=True)

    n_classes = len(set(int(g.y.item()) for g in graphs))
    args.n_classes = n_classes
    print(f"Loaded {len(graphs)} graphs, {n_classes} classes", flush=True)
    for g in graphs:
        print(f"  {g.sample_id}: {g.n_spots} spots, {g.condition}, patient={g.patient}", flush=True)

    splits = _make_lopo_splits(graphs)
    print(f"LOPO splits: {len(splits)}", flush=True)

    # Incremental checkpoint: a LOPO sweep over 3 seeds is long enough that a
    # timeout part-way through would otherwise discard everything. Each finished
    # (method, seed, split) is appended immediately and replayed on restart.
    partial_path = args.output_dir / "records_partial.jsonl"
    records = []
    done: set[tuple] = set()
    if partial_path.is_file():
        with partial_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a killed job
                records.append(r)
                done.add((r.get("method"), r.get("seed"), r.get("split_id"),
                          r.get("random_repeat")))
        print(f"resuming: {len(records)} records already on disk", flush=True)

    def _emit(record):
        key = (record.get("method"), record.get("seed"), record.get("split_id"),
               record.get("random_repeat"))
        if key in done:
            return False
        records.append(record)
        done.add(key)
        with partial_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return True

    for seed in args.seeds:
        for split in splits:
            train_graphs = [graphs[i] for i in split["train_idx"]]
            test_graphs = [graphs[i] for i in split["test_idx"]]
            split_id = split["split_id"]

            if "tsb" in args.methods and ("tsb", seed, split_id, None) not in done:
                print(f"  TSB seed={seed} {split_id}", flush=True)
                r = _train_tsb(args, graphs, train_graphs, test_graphs, seed,
                               split_id, args.target_edge_rate, args.lambda_l0)
                _emit(r)
                print(f"    F1={r['macro_f1']:.4f} ER={r.get('edge_rate', -1):.4f}", flush=True)

            if "plain_gcn" in args.methods and ("plain_gcn", seed, split_id, None) not in done:
                print(f"  GCN seed={seed} {split_id}", flush=True)
                r = _train_plain_gcn(args, graphs, train_graphs, test_graphs, seed, split_id)
                _emit(r)
                print(f"    F1={r['macro_f1']:.4f}", flush=True)

            if "feature_mlp" in args.methods and ("feature_mlp", seed, split_id, None) not in done:
                print(f"  MLP seed={seed} {split_id}", flush=True)
                r = _train_feature_mlp(args, graphs, train_graphs, test_graphs, seed, split_id)
                _emit(r)
                print(f"    F1={r['macro_f1']:.4f}", flush=True)

            if "random_gcn" in args.methods:
                for rep in range(args.random_repeats):
                    if ("random_gcn", seed, split_id, rep) in done:
                        continue
                    r = _train_random_gcn(args, graphs, train_graphs, test_graphs, seed,
                                          split_id, args.target_edge_rate, rep)
                    _emit(r)

    summary = _summarize(records)
    print("\n=== Summary ===")
    for row in summary:
        print(f"  {row['method']}: bal_acc={row.get('balanced_accuracy_mean', 0):.4f}"
              f"±{row.get('balanced_accuracy_std', 0):.4f}"
              f"  F1={row.get('macro_f1_mean', 0):.4f}±{row.get('macro_f1_std', 0):.4f}")

    with (args.output_dir / "psoriasis_results.json").open("w") as f:
        json.dump({"summary": summary, "records": records,
                   "permutation": permutation_info}, f, indent=2)
    print(f"\nSaved to {args.output_dir / 'psoriasis_results.json'}")


if __name__ == "__main__":
    main()
