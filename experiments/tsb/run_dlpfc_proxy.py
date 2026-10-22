'DLPFC layer proxy benchmark for Targeted Spatial Bottleneck.'

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from nicheib.baselines import (
    FeatureMLP,
    PlainGCN,
    matched_budget_edge_mask,
    prune_edge_index,
    random_edge_scores,
)
from nicheib.constants import DLPFC_SAMPLES
from nicheib.data import load_dlpfc
from nicheib.eval import budget_metrics, classification_metrics
from nicheib.loss import tsb_loss
from nicheib.tsb import TargetedSpatialBottleneck

SCHEMA_VERSION = "dlpfc_proxy_sweep.v1"
METHODS = {"tsb", "random_gcn_matched_budget", "plain_gcn", "feature_mlp"}


def _split_valid_indices(y: torch.Tensor, valid_mask: torch.Tensor, seed: int, test_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.where(valid_mask & (y >= 0))[0].cpu().numpy()
    labels = y[idx].cpu().numpy()
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long)


def _resolve_methods(methods: list[str]) -> list[str]:
    if "all" in methods:
        return ["tsb", "random_gcn_matched_budget", "plain_gcn", "feature_mlp"]
    unknown = sorted(set(methods) - METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def _resolve_list(value, fallback):
    return value if value is not None else [fallback]


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _train_mask(n_nodes: int, train_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[train_idx] = True
    return mask


def _base_record(
    args: argparse.Namespace,
    sample_id: str,
    seed: int,
    method: str,
    data,
    n_classes: int,
    train_mask: torch.Tensor,
    test_idx: torch.Tensor,
    target_edge_rate: float | None,
    lambda_l0: float | None,
    random_repeat: int | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "dlpfc_layer_proxy",
        "method": method,
        "baseline_family": None,
        "sample_id": sample_id,
        "seed": seed,
        "random_repeat": random_repeat,
        "target_edge_rate": target_edge_rate,
        "lambda_l0": lambda_l0,
        "budget_mode": args.budget_mode,
        "uses_spatial_edges": method != "feature_mlp",
        "budget_matched": method == "random_gcn_matched_budget",
        "is_upper_bound": method == "plain_gcn",
        "n_nodes": int(data.x.shape[0]),
        "n_edges": int(data.edge_index.shape[1]),
        "n_features": int(data.x.shape[1]),
        "n_classes": int(n_classes),
        "n_train": int(train_mask.sum().item()),
        "n_test": int(test_idx.numel()),
        "label_names": getattr(data, "label_names", None),
    }


def _classification_record(y: torch.Tensor, test_idx: torch.Tensor, logits: torch.Tensor) -> dict:
    prob = logits.softmax(dim=-1).detach().cpu().numpy()
    pred = prob.argmax(axis=1)
    y_true = y.detach().cpu().numpy()
    test_np = test_idx.detach().cpu().numpy()
    return classification_metrics(y_true[test_np], pred[test_np], prob[test_np])


def _fit_baseline(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    edge_index: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_train = -1.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        if edge_index is None:
            out = model(x)
        else:
            out = model(x, edge_index)
        loss = F.cross_entropy(out["logits"][train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)
            train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
            if train_acc > best_train:
                best_train = train_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_train)


def _train_tsb(
    args: argparse.Namespace,
    data,
    sample_id: str,
    seed: int,
    n_classes: int,
    train_mask_cpu: torch.Tensor,
    test_idx_cpu: torch.Tensor,
    target_edge_rate: float,
    lambda_l0: float,
) -> dict:
    _seed_everything(seed)
    device = torch.device(args.device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = train_mask_cpu.to(device)
    test_idx = test_idx_cpu.to(device)

    model = TargetedSpatialBottleneck(
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_dim=n_classes,
        task_level="node",
        task_type="classification",
        dropout=args.dropout,
        use_xib=args.use_xib,
        gate_init_bias=getattr(args, "gate_init_bias", 3.0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_frac = getattr(args, "warmup_fraction", 0.2)
    warmup_epochs = max(1, int(args.epochs * warmup_frac))
    best_state = None
    best_train = -1.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        # Lambda warmup: linearly ramp from 0 to lambda_l0 over warmup_epochs
        warmup_scale = min(1.0, (epoch + 1) / warmup_epochs)
        effective_lambda = lambda_l0 * warmup_scale
        out = model(x, edge_index, temperature=args.temperature)
        loss = tsb_loss(
            out,
            y,
            valid_mask=train_mask,
            lambda_l0=effective_lambda,
            target_edge_rate=target_edge_rate,
            beta_xib=args.beta_xib,
            budget_mode=args.budget_mode,
        )
        loss["total"].backward()
        optimizer.step()

        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)
            train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
            if train_acc > best_train:
                best_train = train_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, temperature=args.temperature)

    record = _base_record(
        args,
        sample_id,
        seed,
        "tsb",
        data,
        n_classes,
        train_mask_cpu,
        test_idx_cpu,
        target_edge_rate,
        lambda_l0,
    )
    record.update(_classification_record(y, test_idx, out["logits"]))
    record.update(budget_metrics(out["edge_prob"].detach().cpu().numpy(), target_edge_rate))
    record["best_train_accuracy"] = float(best_train)

    # Save checkpoint for fidelity evaluation
    if getattr(args, "save_checkpoint", False):
        ckpt_dir = args.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"tsb_{sample_id}_s{seed}_er{target_edge_rate}_l{lambda_l0}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_kwargs": {
                "input_dim": x.shape[1],
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "output_dim": n_classes,
                "task_level": "node",
                "task_type": "classification",
                "dropout": args.dropout,
                "use_xib": args.use_xib,
                "gate_init_bias": getattr(args, "gate_init_bias", 3.0),
            },
            "sample_id": sample_id,
            "seed": seed,
            "target_edge_rate": target_edge_rate,
            "lambda_l0": lambda_l0,
        }, ckpt_path)
        record["checkpoint_path"] = str(ckpt_path)

    return record


def _train_plain_gcn(
    args: argparse.Namespace,
    data,
    sample_id: str,
    seed: int,
    n_classes: int,
    train_mask_cpu: torch.Tensor,
    test_idx_cpu: torch.Tensor,
) -> dict:
    _seed_everything(seed)
    device = torch.device(args.device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = train_mask_cpu.to(device)
    test_idx = test_idx_cpu.to(device)

    model = PlainGCN(x.shape[1], args.hidden_dim, n_classes, task_level="node", dropout=args.dropout).to(device)
    model, best_train = _fit_baseline(model, x, y, train_mask, args.epochs, args.lr, args.weight_decay, edge_index)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)

    record = _base_record(args, sample_id, seed, "plain_gcn", data, n_classes, train_mask_cpu, test_idx_cpu, None, None)
    record.update(_classification_record(y, test_idx, out["logits"]))
    record.update(budget_metrics(np.ones(data.edge_index.shape[1], dtype=np.float32), None))
    record["best_train_accuracy"] = best_train
    return record


def _train_feature_mlp(
    args: argparse.Namespace,
    data,
    sample_id: str,
    seed: int,
    n_classes: int,
    train_mask_cpu: torch.Tensor,
    test_idx_cpu: torch.Tensor,
) -> dict:
    _seed_everything(seed)
    device = torch.device(args.device)
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = train_mask_cpu.to(device)
    test_idx = test_idx_cpu.to(device)

    model = FeatureMLP(x.shape[1], args.hidden_dim, n_classes, task_level="node").to(device)
    model, best_train = _fit_baseline(model, x, y, train_mask, args.epochs, args.lr, args.weight_decay)

    model.eval()
    with torch.no_grad():
        out = model(x)

    record = _base_record(args, sample_id, seed, "feature_mlp", data, n_classes, train_mask_cpu, test_idx_cpu, None, None)
    record.update(_classification_record(y, test_idx, out["logits"]))
    record.update(budget_metrics(np.zeros(data.edge_index.shape[1], dtype=np.float32), None))
    record["best_train_accuracy"] = best_train
    return record


def _train_random_gcn(
    args: argparse.Namespace,
    data,
    sample_id: str,
    seed: int,
    n_classes: int,
    train_mask_cpu: torch.Tensor,
    test_idx_cpu: torch.Tensor,
    target_edge_rate: float,
    random_repeat: int,
) -> dict:
    repeat_seed = seed + 1009 * (random_repeat + 1)
    _seed_everything(repeat_seed)
    device = torch.device(args.device)
    x = data.x.to(device)
    y = data.y.to(device)
    full_edge_index = data.edge_index.to(device)
    train_mask = train_mask_cpu.to(device)
    test_idx = test_idx_cpu.to(device)

    edge_scores = random_edge_scores(full_edge_index.shape[1], device=device)
    edge_mask = matched_budget_edge_mask(edge_scores, target_edge_rate)
    edge_index = prune_edge_index(full_edge_index, edge_mask)

    model = PlainGCN(x.shape[1], args.hidden_dim, n_classes, task_level="node", dropout=args.dropout).to(device)
    model, best_train = _fit_baseline(model, x, y, train_mask, args.epochs, args.lr, args.weight_decay, edge_index)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)

    record = _base_record(
        args,
        sample_id,
        seed,
        "random_gcn_matched_budget",
        data,
        n_classes,
        train_mask_cpu,
        test_idx_cpu,
        target_edge_rate,
        None,
        random_repeat,
    )
    record["baseline_family"] = "random_budget"
    record.update(_classification_record(y, test_idx, out["logits"]))
    record.update(budget_metrics(edge_mask.detach().cpu().float().numpy(), target_edge_rate))
    record["best_train_accuracy"] = best_train
    return record


def _summarize_records(records: list[dict]) -> list[dict]:
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "auroc_ovr", "edge_rate", "budget_abs_error"]
    groups = defaultdict(list)
    for record in records:
        key = (
            record.get("method"),
            record.get("target_edge_rate"),
            record.get("lambda_l0"),
        )
        groups[key].append(record)

    summary = []
    for (method, target_edge_rate, lambda_l0), values in sorted(groups.items(), key=lambda item: str(item[0])):
        row = {
            "method": method,
            "target_edge_rate": target_edge_rate,
            "lambda_l0": lambda_l0,
            "n": len(values),
            "n_sections": len({v["sample_id"] for v in values}),
            "n_seeds": len({v["seed"] for v in values}),
        }
        for key in metric_keys:
            present = np.asarray([float(v[key]) for v in values if key in v and v[key] is not None], dtype=float)
            if present.size:
                row[f"{key}_mean"] = float(present.mean())
                row[f"{key}_std"] = float(present.std())
                row[f"{key}_min"] = float(present.min())
                row[f"{key}_max"] = float(present.max())
        summary.append(row)
    return summary


def _record_key(r: dict) -> tuple:
    """Unique key for a record to detect already-completed runs."""
    return (
        r.get("method"),
        r.get("sample_id"),
        r.get("seed"),
        r.get("target_edge_rate"),
        r.get("lambda_l0"),
        r.get("random_repeat"),
    )


def _load_existing_records(output_dir: Path) -> list[dict]:
    """Load previously saved records for resume support."""
    records_path = output_dir / "records.jsonl"
    if not records_path.exists():
        return []
    records = []
    with records_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_sweep(args: argparse.Namespace) -> list[dict]:
    methods = _resolve_methods(args.methods)
    seeds = _resolve_list(args.seeds, args.seed)
    target_edge_rates = _resolve_list(args.target_edge_rates, args.target_edge_rate)
    lambda_l0_values = _resolve_list(args.lambda_l0_values, args.lambda_l0)

    # Resume: load existing records and skip completed runs
    existing = _load_existing_records(args.output_dir)
    done_keys = {_record_key(r) for r in existing}
    records = list(existing)
    if existing:
        print(f"Resuming: {len(existing)} records already completed", flush=True)

    records_path = args.output_dir / "records.jsonl"
    # Open in append mode for incremental writes
    records_file = records_path.open("a")

    total_sections = len(args.sample_ids) * len(seeds)
    section_idx = 0
    for sample_id in args.sample_ids:
        data = load_dlpfc(
            sample_id,
            data_dir=str(args.data_root),
            n_neighbors=args.n_neighbors,
            n_hvgs=args.n_hvgs,
            n_pcs=args.n_pcs,
        )
        if not hasattr(data, "y"):
            raise ValueError(f"DLPFC sample {sample_id} has no layer labels")
        valid_mask = getattr(data, "valid_mask", torch.ones(data.x.shape[0], dtype=torch.bool))
        labeled_mask = valid_mask & (data.y >= 0)
        n_classes = int(data.y[labeled_mask].max().item() + 1)

        for seed in seeds:
            section_idx += 1
            print(f"[{section_idx}/{total_sections}] section={sample_id} seed={seed}", flush=True)
            train_idx, test_idx = _split_valid_indices(data.y, valid_mask, seed, args.test_size)
            train_mask = _train_mask(data.x.shape[0], train_idx)

            if "feature_mlp" in methods:
                key = ("feature_mlp", sample_id, seed, None, None, None)
                if key not in done_keys:
                    r = _train_feature_mlp(args, data, sample_id, seed, n_classes, train_mask, test_idx)
                    records.append(r)
                    records_file.write(json.dumps(r) + "\n")
                    records_file.flush()
            if "plain_gcn" in methods:
                key = ("plain_gcn", sample_id, seed, None, None, None)
                if key not in done_keys:
                    r = _train_plain_gcn(args, data, sample_id, seed, n_classes, train_mask, test_idx)
                    records.append(r)
                    records_file.write(json.dumps(r) + "\n")
                    records_file.flush()
            if "random_gcn_matched_budget" in methods:
                for target_edge_rate in target_edge_rates:
                    for random_repeat in range(args.random_repeats):
                        key = ("random_gcn_matched_budget", sample_id, seed, target_edge_rate, None, random_repeat)
                        if key not in done_keys:
                            r = _train_random_gcn(
                                args, data, sample_id, seed, n_classes,
                                train_mask, test_idx, target_edge_rate, random_repeat,
                            )
                            records.append(r)
                            records_file.write(json.dumps(r) + "\n")
                            records_file.flush()
            if "tsb" in methods:
                for target_edge_rate in target_edge_rates:
                    for lambda_l0 in lambda_l0_values:
                        key = ("tsb", sample_id, seed, target_edge_rate, lambda_l0, None)
                        if key not in done_keys:
                            r = _train_tsb(
                                args, data, sample_id, seed, n_classes,
                                train_mask, test_idx, target_edge_rate, lambda_l0,
                            )
                            records.append(r)
                            records_file.write(json.dumps(r) + "\n")
                            records_file.flush()
                            print(f"  TSB er={target_edge_rate} l0={lambda_l0} -> actual_er={r['edge_rate']:.4f} f1={r['macro_f1']:.4f}", flush=True)

    records_file.close()
    return records


def _legacy_sections(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("method") == "tsb"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_ids", nargs="+", default=[DLPFC_SAMPLES[0]])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--n_neighbors", type=int, default=6)
    parser.add_argument("--n_hvgs", type=int, default=1000)
    parser.add_argument("--n_pcs", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lambda_l0", type=float, default=0.1)
    parser.add_argument("--lambda_l0_values", nargs="+", type=float)
    parser.add_argument("--target_edge_rate", type=float, default=0.2)
    parser.add_argument("--target_edge_rates", nargs="+", type=float)
    parser.add_argument("--methods", nargs="+", default=["tsb"])
    parser.add_argument("--random_repeats", type=int, default=1)
    parser.add_argument("--budget_mode", default="squared")
    parser.add_argument("--beta_xib", type=float, default=0.0)
    parser.add_argument("--use_xib", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Save TSB model checkpoints for fidelity evaluation")
    parser.add_argument("--gate_init_bias", type=float, default=3.0,
                        help="Initial bias for gate logits (positive = start open)")
    parser.add_argument("--warmup_fraction", type=float, default=0.2,
                        help="Fraction of epochs for lambda warmup (0-1)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = run_sweep(args)
    summary = _summarize_records(records)
    sections = _legacy_sections(records)

    # Rewrite final clean records file (run_sweep already wrote incrementally)
    with (args.output_dir / "records.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    for record in sections:
        if len([r for r in sections if r["sample_id"] == record["sample_id"]]) == 1:
            with (args.output_dir / f"{record['sample_id']}_metrics.json").open("w") as f:
                json.dump(record, f, indent=2)

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "task": "dlpfc_layer_proxy",
        "sample_ids": args.sample_ids,
        "seeds": _resolve_list(args.seeds, args.seed),
        "methods": _resolve_methods(args.methods),
        "target_edge_rates": _resolve_list(args.target_edge_rates, args.target_edge_rate),
        "lambda_l0_values": _resolve_list(args.lambda_l0_values, args.lambda_l0),
        "n_records": len(records),
        "n_sections": len({r["sample_id"] for r in records}),
        "summary": summary,
        "records": records,
        "sections": sections,
    }
    if sections:
        aggregate.update(
            {
                "accuracy_mean": float(np.mean([m["accuracy"] for m in sections])),
                "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in sections])),
                "macro_f1_mean": float(np.mean([m["macro_f1"] for m in sections])),
                "edge_rate_mean": float(np.mean([m["edge_rate"] for m in sections])),
                "budget_abs_error_mean": float(np.mean([m["budget_abs_error"] for m in sections])),
            }
        )

    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(aggregate, f, indent=2)
    with (args.output_dir / "config_resolved.json").open("w") as f:
        config = vars(args).copy()
        config["data_root"] = str(args.data_root)
        config["output_dir"] = str(args.output_dir)
        json.dump(config, f, indent=2)
    print(json.dumps({k: v for k, v in aggregate.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
