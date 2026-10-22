'E9: Does the "spatial graph hurts" regime generalise, and does TSB scale to'

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from nicheib.baselines import FeatureMLP, PlainGAT, PlainGCN
from nicheib.tsb import TargetedSpatialBottleneck
from nicheib.loss import tsb_loss

SQUIDPY_DIR = "data/squidpy"

# label_key: the annotation to predict. batch_key splits one file into independent
# sections (MERFISH ships 12 hypothalamic slices in a single object).
DATASETS = {
    # The section used in the submission, re-run here so the TSB-vs-random
    # comparison can be matched on achieved rather than nominal retention.
    "visium_hne": {"path": f"{SQUIDPY_DIR}/visium_hne.h5ad",
                   "label_key": "cluster", "batch_key": None, "k": 6,
                   "platform": "Visium (55 um)"},
    "visium_fluo": {"path": f"{SQUIDPY_DIR}/visium_fluo_adata.h5ad",
                    "label_key": "cluster", "batch_key": None, "k": 6,
                    "platform": "Visium (55 um)"},
    "slideseqv2": {"path": f"{SQUIDPY_DIR}/slideseqv2.h5ad",
                   "label_key": "cluster", "batch_key": None, "k": 10,
                   "platform": "Slide-seqV2 (10 um)"},
    "merfish": {"path": f"{SQUIDPY_DIR}/merfish.h5ad",
                "label_key": "Cell_class", "batch_key": "batch", "k": 10,
                "platform": "MERFISH (single cell)"},
    "seqfish": {"path": f"{SQUIDPY_DIR}/seqfish.h5ad",
                "label_key": "celltype_mapped_refined", "batch_key": None, "k": 10,
                "platform": "seqFISH (single cell)"},
}


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_sections(spec, n_hvgs=1000, n_pcs=64, min_spots=500, max_sections=4):
    import scanpy as sc
    from nicheib.data import build_spatial_graph

    adata = sc.read_h5ad(spec["path"])
    groups = [(None, adata)] if spec["batch_key"] is None else [
        (str(b), adata[adata.obs[spec["batch_key"]] == b].copy())
        for b in adata.obs[spec["batch_key"]].unique()
    ]
    groups = [(b, a) for b, a in groups if a.n_obs >= min_spots][:max_sections]

    out = []
    for batch, sub in groups:
        labels = sub.obs[spec["label_key"]]
        valid = labels.notna()
        sub = sub[valid].copy()
        if sub.n_obs < min_spots:
            continue
        data = build_spatial_graph(sub, n_neighbors=spec["k"],
                                   n_hvgs=n_hvgs, n_pcs=n_pcs)
        cat = sub.obs[spec["label_key"]].astype("category")
        data.y = torch.tensor(cat.cat.codes.values.astype(np.int64))
        data.label_names = list(cat.cat.categories)
        out.append((batch or "all", data))
    return out


def make_split(y, seed, test_size=0.3):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    counts = np.bincount(y.numpy())
    # train_test_split cannot stratify singleton classes.
    strat = y.numpy() if (counts[counts > 0] >= 2).all() else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    mask = torch.zeros(len(y), dtype=torch.bool)
    mask[tr] = True
    return mask, torch.tensor(te, dtype=torch.long)


def macro_f1(y, test_idx, logits):
    from sklearn.metrics import f1_score

    return float(f1_score(y[test_idx].cpu().numpy(),
                          logits.argmax(dim=-1)[test_idx].cpu().numpy(),
                          average="macro"))


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


def spatial_tiles(coords, tile_nodes):
    """Partition spots into roughly square spatial tiles of ~tile_nodes each."""
    c = np.asarray(coords, dtype=np.float64)
    n_tiles = max(1, int(np.ceil(len(c) / tile_nodes)))
    grid = int(np.ceil(np.sqrt(n_tiles)))
    gx = np.clip(((c[:, 0] - c[:, 0].min()) / (np.ptp(c[:, 0]) + 1e-9) * grid)
                 .astype(int), 0, grid - 1)
    gy = np.clip(((c[:, 1] - c[:, 1].min()) / (np.ptp(c[:, 1]) + 1e-9) * grid)
                 .astype(int), 0, grid - 1)
    return gy * grid + gx


def fit_tsb_tiled(data, y_cpu, train_mask_cpu, n_classes, epochs, target_er,
                  device, tile_nodes=4000, lambda_l0=5.0, lr=1e-3):
    """Train one shared TSB over bounded-size spatial tiles.

    This is the "subgraph batching" the submission lists as future work. Peak memory
    then depends on the tile, not the section, so the same model trains on graphs
    that do not fit whole on the device.
    """
    coords = data.coords.cpu().numpy()
    tiles = spatial_tiles(coords, tile_nodes)
    edge_index = data.edge_index
    n = data.x.shape[0]

    subs = []
    for t in np.unique(tiles):
        node_mask = torch.tensor(tiles == t)
        if node_mask.sum() < 20:
            continue
        keep = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        remap = torch.full((n,), -1, dtype=torch.long)
        idx = node_mask.nonzero(as_tuple=True)[0]
        remap[idx] = torch.arange(len(idx))
        subs.append((data.x[idx].to(device), remap[edge_index[:, keep]].to(device),
                     y_cpu[idx].to(device), train_mask_cpu[idx].to(device), idx))

    model = TargetedSpatialBottleneck(
        input_dim=data.x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    for epoch in range(epochs):
        ws = min(1.0, (epoch + 1) / warmup)
        for sx, sei, sy, stm, _ in subs:
            if stm.sum() == 0:
                continue
            model.train()
            opt.zero_grad()
            out = model(sx, sei)
            tsb_loss(out, sy, task_type="classification", valid_mask=stm,
                     lambda_l0=lambda_l0 * ws, target_edge_rate=target_er,
                     budget_mode="squared")["total"].backward()
            opt.step()

    model.eval()
    logits = torch.zeros(n, n_classes)
    ers = []
    with torch.no_grad():
        for sx, sei, _, _, idx in subs:
            out = model(sx, sei)
            logits[idx] = out["logits"].cpu()
            ers.append(float(out["edge_rate"]))
    return logits, float(np.mean(ers)), len(subs)


def fit_tsb(data, x, y, edge_index, train_mask, n_classes, epochs, target_er,
            device, lambda_l0=5.0, lr=1e-3):
    model = TargetedSpatialBottleneck(
        input_dim=x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        ws = min(1.0, (epoch + 1) / warmup)
        tsb_loss(out, y, task_type="classification", valid_mask=train_mask,
                 lambda_l0=lambda_l0 * ws, target_edge_rate=target_er,
                 budget_mode="squared")["total"].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    return out["logits"], out["edge_prob"].cpu().numpy(), float(out["edge_rate"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--target-er", type=float, nargs="+", default=[0.2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-sections", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tiled", action="store_true",
                        help="Also train with spatial subgraph batching")
    parser.add_argument("--tile-nodes", type=int, default=4000)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ({spec['platform']})", flush=True)
    sections = load_sections(spec, max_sections=args.max_sections)
    print(f"  {len(sections)} sections", flush=True)

    records = []
    for name, data in sections:
        n, m = data.x.shape[0], data.edge_index.shape[1]
        n_classes = len(data.label_names)
        print(f"\n=== {args.dataset}/{name}: {n} nodes, {m} edges, "
              f"{n_classes} classes ===", flush=True)
        x, ei, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)

        for seed in args.seeds:
            train_mask_cpu, test_idx = make_split(data.y, seed)
            train_mask = train_mask_cpu.to(device)

            def record(method, f1, er, secs, extra=None):
                r = {"dataset": args.dataset, "platform": spec["platform"],
                     "section": name, "seed": seed, "method": method,
                     "macro_f1": f1, "edge_rate": er, "n_nodes": n, "n_edges": m,
                     "n_classes": n_classes, "seconds": secs}
                r.update(extra or {})
                records.append(r)
                return r

            seed_everything(seed)
            t0 = time.perf_counter()
            f1_mlp = macro_f1(y, test_idx, fit_plain(
                FeatureMLP(x.shape[1], 64, n_classes, task_level="node").to(device),
                x, y, train_mask, args.epochs))
            record("feature_mlp", f1_mlp, 0.0, time.perf_counter() - t0)

            seed_everything(seed)
            t0 = time.perf_counter()
            f1_gcn = macro_f1(y, test_idx, fit_plain(
                PlainGCN(x.shape[1], 64, n_classes, task_level="node").to(device),
                x, y, train_mask, args.epochs, ei))
            record("plain_gcn", f1_gcn, 1.0, time.perf_counter() - t0)

            seed_everything(seed)
            t0 = time.perf_counter()
            f1_gat = macro_f1(y, test_idx, fit_plain(
                PlainGAT(x.shape[1], 64, 32, n_classes, task_level="node").to(device),
                x, y, train_mask, args.epochs, ei))
            record("plain_gat", f1_gat, 1.0, time.perf_counter() - t0)

            for target_er in args.target_er:
                seed_everything(seed)
                t0 = time.perf_counter()
                logits, gate_scores, actual_er = fit_tsb(
                    data, x, y, ei, train_mask, n_classes, args.epochs,
                    target_er, device)
                f1_tsb = macro_f1(y, test_idx, logits)
                secs_tsb = time.perf_counter() - t0
                peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                        if device.type == "cuda" else None)
                record("tsb", f1_tsb, actual_er, secs_tsb,
                       {"peak_mem_mb": peak, "target_er": target_er})

                # Random subgraph with exactly TSB's *achieved* edge count. Matching
                # on the nominal budget instead would compare graphs of different
                # size whenever the budget loss fails to bind.
                k = int(round(actual_er * m))
                rng = np.random.default_rng(7919 + seed)
                keep = np.zeros(m, dtype=bool)
                keep[np.argsort(rng.random(m))[-k:]] = True
                seed_everything(seed)
                t0 = time.perf_counter()
                f1_rnd = macro_f1(y, test_idx, fit_plain(
                    PlainGCN(x.shape[1], 64, n_classes, task_level="node").to(device),
                    x, y, train_mask, args.epochs,
                    ei[:, torch.tensor(keep).to(device)]))
                record("random_matched", f1_rnd, k / m, time.perf_counter() - t0,
                       {"target_er": target_er})

                # Architecture control: the same gate-free GAT that TSB is built on,
                # given a random subgraph of the same size. Comparing TSB against a
                # GCN on random edges would otherwise confound edge selection with
                # the attention-plus-skip architecture.
                seed_everything(seed)
                t0 = time.perf_counter()
                f1_gat_rnd = macro_f1(y, test_idx, fit_plain(
                    PlainGAT(x.shape[1], 64, 32, n_classes,
                             task_level="node").to(device),
                    x, y, train_mask, args.epochs,
                    ei[:, torch.tensor(keep).to(device)]))
                record("gat_random_matched", f1_gat_rnd, k / m,
                       time.perf_counter() - t0, {"target_er": target_er})
                print(f"      architecture control: GAT@random-ER {f1_gat_rnd:.4f}",
                      flush=True)
                print(f"  seed {seed} budget {target_er}: TSB {f1_tsb:.4f} "
                      f"(achieved ER {actual_er:.3f}, {secs_tsb:.0f}s) | "
                      f"random@achieved-ER {f1_rnd:.4f}"
                      + (f" | peak {peak:.0f} MB" if peak else ""), flush=True)

            # Subgraph batching on the same section, so the accuracy cost of tiling
            # is measured on real data rather than asserted.
            if args.tiled and n > args.tile_nodes:
                seed_everything(seed)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                logits_t, er_t, n_tiles = fit_tsb_tiled(
                    data, data.y, train_mask_cpu, n_classes, args.epochs,
                    args.target_er[0], device, tile_nodes=args.tile_nodes)
                f1_tiled = macro_f1(data.y.cpu(), test_idx, logits_t)
                peak_t = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                          if device.type == "cuda" else None)
                record("tsb_tiled", f1_tiled, er_t, time.perf_counter() - t0,
                       {"peak_mem_mb": peak_t, "n_tiles": n_tiles,
                        "tile_nodes": args.tile_nodes})
                print(f"  seed {seed}: TSB tiled {f1_tiled:.4f} "
                      f"({n_tiles} tiles, ER {er_t:.3f}"
                      + (f", peak {peak_t:.0f} MB" if peak_t else "") + ")", flush=True)

            print(f"  seed {seed}: MLP {f1_mlp:.4f} | GCN {f1_gcn:.4f} | "
                  f"GAT {f1_gat:.4f}", flush=True)

        with (out_dir / f"{args.dataset}.json").open("w") as f:
            json.dump(records, f, indent=2)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
