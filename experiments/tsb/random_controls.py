'E4: Honest sparsity controls for the TSB-vs-random comparison on DLPFC.'

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
from nicheib.loss import tsb_loss


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a degree sequence (0 = uniform, 1 = fully concentrated)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.sum() == 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_split(y: torch.Tensor, seed: int, test_size: float = 0.3):
    """Stratified-by-availability 70/30 split over annotated spots."""
    from sklearn.model_selection import train_test_split

    valid = (y >= 0).nonzero(as_tuple=True)[0].numpy()
    train_idx, test_idx = train_test_split(
        valid, test_size=test_size, random_state=seed, stratify=y.numpy()[valid])
    train_mask = torch.zeros(y.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True
    return train_mask, torch.tensor(test_idx, dtype=torch.long)


def _maybe_checkpoint(model, logits, y, train_mask, state):
    """Track the highest-training-accuracy state, as run_dlpfc_proxy.py does.

    The submission selects, for every method alike, the epoch with the best training
    accuracy rather than the final epoch. Reproducing that here keeps these controls
    numerically comparable to Table 1; --no-best-checkpoint reruns the same
    comparison at the final epoch to show whether the margin depends on it.
    """
    with torch.no_grad():
        acc = (logits.argmax(dim=-1)[train_mask] == y[train_mask]).float().mean().item()
    if acc > state["best"]:
        state["best"] = acc
        state["sd"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit_gcn(x, y, edge_index, train_mask, n_classes, epochs, lr=1e-3,
            weight_decay=1e-4, hidden=64, dropout=0.1, best_checkpoint=True):
    model = PlainGCN(x.shape[1], hidden, n_classes, task_level="node", dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    state = {"best": -1.0, "sd": None}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        F.cross_entropy(out["logits"][train_mask], y[train_mask]).backward()
        opt.step()
        if best_checkpoint:
            _maybe_checkpoint(model, out["logits"], y, train_mask, state)
    if best_checkpoint and state["sd"] is not None:
        model.load_state_dict(state["sd"])
    model.eval()
    with torch.no_grad():
        return model(x, edge_index)["logits"]


def fit_mlp(x, y, train_mask, n_classes, epochs, lr=1e-3, weight_decay=1e-4,
            hidden=64, best_checkpoint=True):
    model = FeatureMLP(x.shape[1], hidden, n_classes, task_level="node")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    state = {"best": -1.0, "sd": None}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x)
        F.cross_entropy(out["logits"][train_mask], y[train_mask]).backward()
        opt.step()
        if best_checkpoint:
            _maybe_checkpoint(model, out["logits"], y, train_mask, state)
    if best_checkpoint and state["sd"] is not None:
        model.load_state_dict(state["sd"])
    model.eval()
    with torch.no_grad():
        return model(x)["logits"]


def train_tsb(data, train_mask, n_classes, seed, epochs, target_er,
              lambda_l0=5.0, lr=1e-3, best_checkpoint=True):
    seed_everything(seed)
    model = TargetedSpatialBottleneck(
        input_dim=data.x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    state = {"best": -1.0, "sd": None}
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        ws = min(1.0, (epoch + 1) / warmup)
        tsb_loss(out, data.y, task_type="classification", valid_mask=train_mask,
                 lambda_l0=lambda_l0 * ws, target_edge_rate=target_er,
                 budget_mode="squared")["total"].backward()
        opt.step()
        if best_checkpoint:
            _maybe_checkpoint(model, out["logits"], data.y, train_mask, state)
    if best_checkpoint and state["sd"] is not None:
        model.load_state_dict(state["sd"])
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return out["logits"], out["edge_prob"].cpu().numpy(), float(out["edge_rate"])


def topk_mask(scores: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    order = np.argsort(scores)
    keep = order[-k:] if largest else order[:k]
    mask = np.zeros(len(scores), dtype=bool)
    mask[keep] = True
    return mask


def rewired_edge_index(n_nodes: int, n_edges: int, rng) -> torch.Tensor:
    """Size-matched graph of uniformly random node pairs — geometry destroyed."""
    src = rng.integers(0, n_nodes, size=n_edges * 2)
    dst = rng.integers(0, n_nodes, size=n_edges * 2)
    keep = src != dst
    src, dst = src[keep][:n_edges], dst[keep][:n_edges]
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def macro_f1(y, test_idx, logits) -> float:
    from sklearn.metrics import f1_score

    pred = logits.argmax(dim=-1)[test_idx].cpu().numpy()
    return float(f1_score(y[test_idx].cpu().numpy(), pred, average="macro"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="data/dlpfc")
    parser.add_argument("--out-dir", type=str,
                        default="results/rebuttal_controls")
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-random-repeats", type=int, default=3)
    parser.add_argument("--no-best-checkpoint", action="store_true",
                        help="Evaluate the final epoch instead of the best-training-"
                             "accuracy epoch that the submission selects")
    args = parser.parse_args()

    from nicheib.data import load_dlpfc

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    bc = not args.no_best_checkpoint

    for section in args.sections:
        data = load_dlpfc(section, data_dir=args.data_dir,
                          n_hvgs=1000, n_pcs=64, n_neighbors=6)
        n_nodes, n_edges = data.x.shape[0], data.edge_index.shape[1]
        n_classes = len(data.label_names)
        print(f"\n=== {section}: {n_nodes} spots, {n_edges} edges ===", flush=True)

        for seed in args.seeds:
            train_mask, test_idx = make_split(data.y, seed)

            # Reference points that do not depend on the budget.
            seed_everything(seed)
            f1_full = macro_f1(data.y, test_idx,
                               fit_gcn(data.x, data.y, data.edge_index, train_mask,
                                       n_classes, args.epochs,
                                       best_checkpoint=bc))
            seed_everything(seed)
            f1_mlp = macro_f1(data.y, test_idx,
                              fit_mlp(data.x, data.y, train_mask, n_classes, args.epochs,
                                      best_checkpoint=bc))
            records += [
                {"section": section, "seed": seed, "budget": None,
                 "method": "plain_gcn_full", "macro_f1": f1_full, "edge_rate": 1.0},
                {"section": section, "seed": seed, "budget": None,
                 "method": "feature_mlp", "macro_f1": f1_mlp, "edge_rate": 0.0},
            ]
            print(f"  seed {seed}: full GCN {f1_full:.4f} | MLP {f1_mlp:.4f}", flush=True)

            for budget in args.budgets:
                logits, gate_scores, actual_er = train_tsb(
                    data, train_mask, n_classes, seed, args.epochs, budget,
                    best_checkpoint=bc)
                f1_tsb = macro_f1(data.y, test_idx, logits)
                # Every control below retains exactly this many edges.
                k = int(round(actual_er * n_edges))
                records.append(
                    {"section": section, "seed": seed, "budget": budget, "method": "tsb",
                     "macro_f1": f1_tsb, "edge_rate": actual_er, "n_kept": k,
                     "best_checkpoint": bc})
                print(f"  seed {seed} budget {budget}: TSB {f1_tsb:.4f} "
                      f"(actual ER {actual_er:.3f}, k={k})", flush=True)

                def gcn_on(mask_or_index, name, repeat=None):
                    seed_everything(seed if repeat is None else seed + 1009 * (repeat + 1))
                    ei = mask_or_index if torch.is_tensor(mask_or_index) else \
                        data.edge_index[:, torch.tensor(mask_or_index)]
                    f1 = macro_f1(data.y, test_idx,
                                  fit_gcn(data.x, data.y, ei, train_mask,
                                          n_classes, args.epochs,
                                          best_checkpoint=bc))
                    # Node coverage matters: a GCN cannot aggregate for isolated spots,
                    # so a selector that concentrates edges is penalised independently
                    # of whether the edges it picked are the informative ones.
                    deg = np.bincount(ei.reshape(-1).cpu().numpy(), minlength=n_nodes)
                    records.append({"section": section, "seed": seed, "budget": budget,
                                    "method": name, "macro_f1": f1,
                                    "edge_rate": ei.shape[1] / n_edges,
                                    "n_kept": ei.shape[1], "random_repeat": repeat,
                                    "best_checkpoint": bc,
                                    "frac_isolated": float((deg == 0).mean()),
                                    "mean_degree": float(deg.mean()),
                                    "degree_gini": float(_gini(deg))})
                    return f1

                # Architecture control: TSB is a gated GAT with a skip connection, so
                # comparing it against a plain GCN on random edges confounds edge
                # selection with architecture. This arm keeps the architecture fixed.
                seed_everything(seed)
                rng0 = np.random.default_rng(31337 + seed)
                keep_r = topk_mask(rng0.random(n_edges), k, largest=True)
                ei_r = data.edge_index[:, torch.tensor(keep_r)]
                gat = PlainGAT(data.x.shape[1], 64, 32, n_classes, task_level="node")
                opt_g = torch.optim.Adam(gat.parameters(), lr=1e-3, weight_decay=1e-4)
                for _ in range(args.epochs):
                    gat.train()
                    opt_g.zero_grad()
                    o = gat(data.x, ei_r)
                    F.cross_entropy(o["logits"][train_mask], data.y[train_mask]).backward()
                    opt_g.step()
                gat.eval()
                with torch.no_grad():
                    f1_gat_r = macro_f1(data.y, test_idx, gat(data.x, ei_r)["logits"])
                records.append({"section": section, "seed": seed, "budget": budget,
                                "method": "gat_random", "macro_f1": f1_gat_r,
                                "edge_rate": k / n_edges, "n_kept": k})
                print(f"      architecture control: GAT@random {f1_gat_r:.4f}", flush=True)

                f1_top = gcn_on(topk_mask(gate_scores, k, largest=True), "gcn_tsb_top")
                f1_bot = gcn_on(topk_mask(gate_scores, k, largest=False), "gcn_tsb_bottom")

                rnd, rew = [], []
                for rep in range(args.n_random_repeats):
                    rng = np.random.default_rng(7919 * (rep + 1) + seed)
                    rnd.append(gcn_on(
                        topk_mask(rng.random(n_edges), k, largest=True),
                        "gcn_random", repeat=rep))
                    rew.append(gcn_on(
                        rewired_edge_index(n_nodes, k, rng), "gcn_rewired", repeat=rep))
                print(f"      GCN on subgraph — TSB-top {f1_top:.4f} | "
                      f"random {np.mean(rnd):.4f} | TSB-bottom {f1_bot:.4f} | "
                      f"rewired {np.mean(rew):.4f}", flush=True)

        with (outdir / f"controls_{section}.json").open("w") as f:
            json.dump([r for r in records if r["section"] == section], f, indent=2)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
