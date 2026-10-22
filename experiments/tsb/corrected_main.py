'The corrected main result: both method fixes applied, with matched baselines.'

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from nicheib.baselines import FeatureMLP, PlainGAT, PlainGCN
from nicheib.tsb import TargetedSpatialBottleneck


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def coverage_penalty(expected_l0, edge_index, n_nodes, d_min):
    """Squared shortfall of each node's expected retained degree below d_min."""
    deg = torch.zeros(n_nodes, device=expected_l0.device)
    deg = deg.scatter_add(0, edge_index[1], expected_l0)
    return F.relu(d_min - deg).pow(2).mean()


def train_tsb(data, train_mask, n_classes, seed, epochs, target_er, mode,
              lambda_l0=5.0, rho=20.0, dual_lr=0.1, lambda_cov=5.0, d_min=1.5,
              lr=1e-3):
    seed_everything(seed)
    n_nodes = data.x.shape[0]
    model = TargetedSpatialBottleneck(
        input_dim=data.x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, int(0.2 * epochs))
    mu = 0.0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        ws = min(1.0, (epoch + 1) / warmup)
        pred = F.cross_entropy(out["logits"][train_mask], data.y[train_mask])
        el0 = out["expected_l0"]
        gap = el0.mean() - target_er

        if mode == "submitted":
            total = pred + lambda_l0 * ws * gap.pow(2)
        else:
            total = pred + ws * (mu * gap + 0.5 * rho * gap.pow(2))
        if mode == "dual_cov":
            total = total + lambda_cov * ws * coverage_penalty(
                el0, data.edge_index, n_nodes, d_min)

        total.backward()
        if mode != "submitted":
            torch.nn.utils.clip_grad_norm_(
                list(model.gate1.parameters()) + list(model.gate2.parameters()), 1.0)
        opt.step()
        if mode != "submitted" and epoch >= warmup:
            mu = float(np.clip(mu + dual_lr * float(gap.detach()), 0.0, 50.0))

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return (out["logits"], out["edge_prob"].cpu().numpy(), float(out["edge_rate"]))


def fit_plain(model, x, y, train_mask, epochs, edge_index=None, lr=1e-3, wd=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x) if edge_index is None else model(x, edge_index)
        F.cross_entropy(out["logits"][train_mask], y[train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return (model(x) if edge_index is None else model(x, edge_index))["logits"]


def macro_f1(y, test_idx, logits):
    from sklearn.metrics import f1_score

    return float(f1_score(y[test_idx].cpu().numpy(),
                          logits.argmax(dim=-1)[test_idx].cpu().numpy(),
                          average="macro"))


def make_split(y, seed, test_size=0.3):
    from sklearn.model_selection import train_test_split

    valid = (y >= 0).nonzero(as_tuple=True)[0].numpy()
    tr, te = train_test_split(valid, test_size=test_size, random_state=seed,
                              stratify=y.numpy()[valid])
    mask = torch.zeros(y.shape[0], dtype=torch.bool)
    mask[tr] = True
    return mask, torch.tensor(te, dtype=torch.long)


def topk(scores, k, largest=True):
    order = np.argsort(scores)
    keep = np.zeros(len(scores), dtype=bool)
    keep[order[-k:] if largest else order[:k]] = True
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        default="data/dlpfc")
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-random-repeats", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from nicheib.data import load_dlpfc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for section in args.sections:
        data = load_dlpfc(section, data_dir=args.data_dir,
                          n_hvgs=1000, n_pcs=64, n_neighbors=6)
        n_nodes, n_edges = data.x.shape[0], data.edge_index.shape[1]
        n_classes = len(data.label_names)
        print(f"\n=== {section}: {n_nodes} spots, {n_edges} edges ===", flush=True)

        for seed in args.seeds:
            train_mask, test_idx = make_split(data.y, seed)

            def add(method, f1, er, extra=None):
                r = {"section": section, "seed": seed, "method": method,
                     "macro_f1": f1, "edge_rate": er}
                r.update(extra or {})
                records.append(r)

            # Full-graph references, once per seed.
            for name, model, ei in (
                    ("feature_mlp", FeatureMLP(data.x.shape[1], 64, n_classes,
                                               task_level="node"), None),
                    ("plain_gcn", PlainGCN(data.x.shape[1], 64, n_classes,
                                           task_level="node"), data.edge_index),
                    ("plain_gat", PlainGAT(data.x.shape[1], 64, 32, n_classes,
                                           task_level="node"), data.edge_index)):
                seed_everything(seed)
                add(name, macro_f1(data.y, test_idx, fit_plain(
                    model, data.x, data.y, train_mask, args.epochs, ei)),
                    0.0 if ei is None else 1.0)

            for budget in args.budgets:
                for mode in ("submitted", "dual", "dual_cov"):
                    logits, scores, er = train_tsb(
                        data, train_mask, n_classes, seed, args.epochs, budget, mode)
                    f1 = macro_f1(data.y, test_idx, logits)
                    k = max(1, int(round(er * n_edges)))
                    ei_top = data.edge_index[:, torch.tensor(topk(scores, k))]
                    deg = np.bincount(ei_top.reshape(-1).numpy(), minlength=n_nodes)
                    add(f"tsb_{mode}", f1, er,
                        {"budget": budget, "budget_error": abs(er - budget),
                         "frac_isolated": float((deg == 0).mean())})

                    # A plain GCN retrained on the retained subgraph, to test whether
                    # the subgraph itself is informative independently of the gate.
                    seed_everything(seed)
                    add(f"gcn_on_{mode}_top", macro_f1(data.y, test_idx, fit_plain(
                        PlainGCN(data.x.shape[1], 64, n_classes, task_level="node"),
                        data.x, data.y, train_mask, args.epochs, ei_top)), er,
                        {"budget": budget})

                    # Size-matched random baselines, both architectures.
                    for rep in range(args.n_random_repeats):
                        rng = np.random.default_rng(7919 * (rep + 1) + seed)
                        ei_r = data.edge_index[:, torch.tensor(topk(rng.random(n_edges), k))]
                        seed_everything(seed)
                        add(f"gcn_random_at_{mode}", macro_f1(data.y, test_idx, fit_plain(
                            PlainGCN(data.x.shape[1], 64, n_classes, task_level="node"),
                            data.x, data.y, train_mask, args.epochs, ei_r)), er,
                            {"budget": budget, "random_repeat": rep})
                        seed_everything(seed)
                        add(f"gat_random_at_{mode}", macro_f1(data.y, test_idx, fit_plain(
                            PlainGAT(data.x.shape[1], 64, 32, n_classes,
                                     task_level="node"),
                            data.x, data.y, train_mask, args.epochs, ei_r)), er,
                            {"budget": budget, "random_repeat": rep})

                    iso = next(r["frac_isolated"] for r in reversed(records)
                               if "frac_isolated" in r)
                    print(f"  seed {seed} budget {budget} {mode:9s}: "
                          f"ER {er:.3f} (err {abs(er - budget):.3f}) iso {iso:.3f} "
                          f"| TSB {f1:.4f}", flush=True)

        with (out_dir / f"corrected_{section}.json").open("w") as f:
            json.dump([r for r in records if r["section"] == section], f, indent=2)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
