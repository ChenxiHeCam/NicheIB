'Run the DLPFC transfer protocol on the mouse-brain headline task.'

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.tsb.dlpfc_matched_mask_refit import (
    fit,
    macro_f1,
    make_split,
    seed_everything,
    topk,
    train_tsb,
)
from nicheib.baselines import FeatureMLP, PlainGCN
from nicheib.data import load_mouse_brain


def isolated_fraction(edge_index, keep, n_nodes):
    retained = edge_index[:, torch.tensor(keep)]
    degree = np.bincount(retained.reshape(-1).numpy(), minlength=n_nodes)
    return retained, float((degree == 0).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--budget", type=float, default=0.2)
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if output.exists():
        records = json.loads(output.read_text())
    completed = {record["seed"] for record in records}

    data = load_mouse_brain(
        data_dir=args.data_dir, n_hvgs=1000, n_pcs=64, n_neighbors=6
    )
    n_nodes = data.x.shape[0]
    n_edges = data.edge_index.shape[1]
    n_classes = len(data.label_names)
    print(f"Mouse brain: {n_nodes} nodes, {n_edges} edges, {n_classes} classes",
          flush=True)

    for seed in args.seeds:
        if seed in completed:
            print(f"Seed {seed}: already complete", flush=True)
            continue
        train_mask, test_idx = make_split(data.y, seed)
        seed_records = []

        gated = {}
        for variant in ("submitted", "final"):
            model, logits, scores, edge_rate = train_tsb(
                data, train_mask, n_classes, seed, args.epochs, args.budget,
                variant,
            )
            gated[variant] = {"scores": scores, "edge_rate": edge_rate}
            seed_records.append({
                "seed": seed,
                "selector": f"tsb_{variant}",
                "protocol": "end_to_end",
                "macro_f1": macro_f1(data.y, test_idx, logits),
                "edge_rate": edge_rate,
                "budget_error": abs(edge_rate - args.budget),
            })

        for selector, model, edge_index in (
            ("plain_gcn_full", PlainGCN(data.x.shape[1], 64, n_classes,
                                         task_level="node"), data.edge_index),
            ("feature_mlp", FeatureMLP(data.x.shape[1], 64, n_classes,
                                        task_level="node"), None),
        ):
            seed_everything(seed)
            logits = fit(
                model, data.x, data.y, train_mask, args.epochs, edge_index
            )
            seed_records.append({
                "seed": seed,
                "selector": selector,
                "protocol": "reference",
                "macro_f1": macro_f1(data.y, test_idx, logits),
                "edge_rate": 1.0 if edge_index is not None else 0.0,
                "frac_isolated": 0.0 if edge_index is not None else 1.0,
            })

        k = max(1, int(round(gated["final"]["edge_rate"] * n_edges)))
        rng = np.random.default_rng(7919 + seed)
        selector_scores = {
            "tsb_final": gated["final"]["scores"],
            "tsb_submitted": gated["submitted"]["scores"],
            "random": rng.random(n_edges),
        }
        for selector, scores in selector_scores.items():
            retained, frac_isolated = isolated_fraction(
                data.edge_index, topk(scores, k), n_nodes
            )
            seed_everything(seed)
            logits = fit(
                PlainGCN(data.x.shape[1], 64, n_classes, task_level="node"),
                data.x,
                data.y,
                train_mask,
                args.epochs,
                retained,
            )
            seed_records.append({
                "seed": seed,
                "selector": selector,
                "protocol": "transfer_gcn",
                "macro_f1": macro_f1(data.y, test_idx, logits),
                "edge_rate": k / n_edges,
                "frac_isolated": frac_isolated,
            })

        records.extend(seed_records)
        output.write_text(json.dumps(records, indent=2))
        print(
            f"Seed {seed}: " + " ".join(
                f"{record['selector']}={record['macro_f1']:.4f}"
                for record in seed_records
                if record["protocol"] == "transfer_gcn"
            ),
            flush=True,
        )

    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
