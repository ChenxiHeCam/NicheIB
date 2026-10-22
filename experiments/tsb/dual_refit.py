'Matched-mask refit of TSB-dual on the archived DLPFC and mouse-brain protocols.'

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
from nicheib.baselines import PlainGCN


def refit(data, train_mask, test_idx, n_classes, seed, epochs, scores, k):
    n_nodes = data.x.shape[0]
    retained = data.edge_index[:, torch.tensor(topk(scores, k))]
    degree = np.bincount(retained.reshape(-1).numpy(), minlength=n_nodes)
    seed_everything(seed)
    logits = fit(PlainGCN(data.x.shape[1], 64, n_classes, task_level="node"),
                 data.x, data.y, train_mask, epochs, retained)
    return macro_f1(data.y, test_idx, logits), float((degree == 0).mean())


def archived(rows, seed, budget, selector):
    match = [r for r in rows if r["protocol"] == "transfer_gcn" and r["selector"] == selector
             and r["seed"] == seed and (budget is None or r.get("budget") == budget)]
    assert len(match) == 1, (seed, budget, selector, len(match))
    return match[0]


def run_unit(data, rows, seed, budgets, epochs, unit):
    n_edges = data.edge_index.shape[1]
    n_classes = len(data.label_names)
    train_mask, test_idx = make_split(data.y, seed)
    records = []
    for budget in budgets:
        reference = archived(rows, seed, budget if unit["task"] == "dlpfc" else None, "tsb_final")
        k = int(round(reference["edge_rate"] * n_edges))
        _, logits, scores, edge_rate = train_tsb(
            data, train_mask, n_classes, seed, epochs, budget, "dual")
        base = dict(unit, seed=seed, budget=budget)
        records.append(dict(base, selector="tsb_dual", protocol="end_to_end",
                            macro_f1=macro_f1(data.y, test_idx, logits), edge_rate=edge_rate,
                            budget_error=abs(edge_rate - budget), frac_isolated=None))
        f1, isolated = refit(data, train_mask, test_idx, n_classes, seed, epochs, scores, k)
        records.append(dict(base, selector="tsb_dual", protocol="transfer_gcn", macro_f1=f1,
                            edge_rate=k / n_edges, frac_isolated=isolated))
        rng = np.random.default_rng(7919 + seed)
        f1, isolated = refit(data, train_mask, test_idx, n_classes, seed, epochs,
                             rng.random(n_edges), k)
        old = archived(rows, seed, budget if unit["task"] == "dlpfc" else None, "random")
        records.append(dict(base, selector="random_check", protocol="transfer_gcn",
                            macro_f1=f1, edge_rate=k / n_edges, frac_isolated=isolated,
                            archived_macro_f1=old["macro_f1"]))
        print(f"{unit} seed {seed} budget {budget}: dual refit {records[-2]['macro_f1']:.4f} "
              f"random {f1:.4f} (archived {old['macro_f1']:.4f})", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dlpfc", "mouse"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reference", required=True,
                        help="Archived refit records: unified_<section>.json (DLPFC) or a "
                             "directory of records_seed*.json (mouse).")
    parser.add_argument("--sections", nargs="+", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out.read_text()) if out.exists() else []
    done = {(r.get("section"), r["seed"]) for r in records}

    if args.task == "dlpfc":
        from nicheib.data import load_dlpfc
        for section in args.sections:
            rows = json.loads(Path(args.reference.format(section=section)).read_text())
            data = load_dlpfc(section, data_dir=args.data_dir, n_hvgs=1000, n_pcs=64,
                              n_neighbors=6)
            for seed in args.seeds:
                if (section, seed) in done:
                    continue
                records += run_unit(data, rows, seed, args.budgets, args.epochs,
                                    dict(task="dlpfc", section=section))
                out.write_text(json.dumps(records, indent=1))
    else:
        from nicheib.data import load_mouse_brain
        rows = [r for p in sorted(Path(args.reference).glob("records_seed*.json"))
                for r in json.loads(p.read_text())]
        data = load_mouse_brain(data_dir=args.data_dir, n_hvgs=1000, n_pcs=64, n_neighbors=6)
        for seed in args.seeds:
            if (None, seed) in done:
                continue
            records += run_unit(data, rows, seed, args.budgets, args.epochs,
                                dict(task="mouse", section=None))
            out.write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
