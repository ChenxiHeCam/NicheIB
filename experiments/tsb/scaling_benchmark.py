'E8: How do memory and runtime scale with graph size?'

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from nicheib.tsb import TargetedSpatialBottleneck
from nicheib.loss import tsb_loss


def build_lattice(side: int, n_features: int = 64, n_bands: int = 6, k: int = 6,
                  snr: float = 0.5, seed: int = 0):
    """Square lattice with band labels and spatially autocorrelated features."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    n = coords.shape[0]
    labels = (yy.ravel() * n_bands // side).astype(np.int64)

    signal = np.zeros((n, n_features), dtype=np.float32)
    centres = rng.normal(size=(n_bands, n_features)).astype(np.float32)
    signal += centres[labels]
    x = snr * signal + rng.normal(size=(n, n_features)).astype(np.float32)

    # k-NN on the lattice is the 4/8-neighbourhood; build it analytically to avoid
    # an O(n^2) neighbour search at the largest sizes.
    idx = np.arange(n).reshape(side, side)
    src, dst = [], []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)][:k]
    for dy, dx in offsets:
        a = idx[max(0, -dy):side - max(0, dy), max(0, -dx):side - max(0, dx)]
        b = idx[max(0, dy):side - max(0, -dy), max(0, dx):side - max(0, -dx)]
        src.append(a.ravel())
        dst.append(b.ravel())
    edge_index = torch.tensor(
        np.stack([np.concatenate(src), np.concatenate(dst)]), dtype=torch.long)
    return (torch.tensor(x), edge_index, torch.tensor(labels),
            torch.tensor(coords), n_bands)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def time_epochs(model, x, edge_index, y, train_mask, device, n_epochs=10,
                target_er=0.2, lambda_l0=5.0):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # One untimed epoch so lazy allocations do not land in the measurement.
    for i in range(n_epochs + 1):
        if i == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        tsb_loss(out, y, task_type="classification", valid_mask=train_mask,
                 lambda_l0=lambda_l0, target_edge_rate=target_er,
                 budget_mode="squared")["total"].backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_epochs


def spatial_tiles(coords: torch.Tensor, tile_nodes: int):
    """Partition nodes into roughly square spatial tiles of ~tile_nodes each."""
    n = coords.shape[0]
    n_tiles = max(1, int(np.ceil(n / tile_nodes)))
    grid = int(np.ceil(np.sqrt(n_tiles)))
    c = coords.numpy()
    gx = np.clip(((c[:, 0] - c[:, 0].min()) /
                  (np.ptp(c[:, 0]) + 1e-9) * grid).astype(int), 0, grid - 1)
    gy = np.clip(((c[:, 1] - c[:, 1].min()) /
                  (np.ptp(c[:, 1]) + 1e-9) * grid).astype(int), 0, grid - 1)
    return gy * grid + gx


def induced_subgraph(edge_index, node_mask):
    """Edges with both endpoints inside the tile, reindexed to local node ids."""
    keep = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    sub = edge_index[:, keep]
    remap = torch.full((node_mask.shape[0],), -1, dtype=torch.long)
    remap[node_mask.nonzero(as_tuple=True)[0]] = torch.arange(int(node_mask.sum()))
    return remap[sub]


def run_full(side, device, args):
    x, edge_index, y, coords, n_classes = build_lattice(side, k=args.k)
    n, m = x.shape[0], edge_index.shape[1]
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[torch.randperm(n, device=device)[: int(0.7 * n)]] = True

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = TargetedSpatialBottleneck(
        input_dim=x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0).to(device)
    sec = time_epochs(model, x, edge_index, y, train_mask, device, args.n_epochs)
    return {"mode": "full", "side": side, "n_nodes": n, "n_edges": m,
            "sec_per_epoch": sec, "peak_mem_mb": peak_memory_mb(device)}


def run_patched(side, device, args):
    x, edge_index, y, coords, n_classes = build_lattice(side, k=args.k)
    n, m = x.shape[0], edge_index.shape[1]
    tiles = spatial_tiles(coords, args.tile_nodes)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = TargetedSpatialBottleneck(
        input_dim=x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    subs = []
    for t in np.unique(tiles):
        node_mask = torch.tensor(tiles == t)
        if node_mask.sum() < 10:
            continue
        sub_ei = induced_subgraph(edge_index, node_mask)
        idx = node_mask.nonzero(as_tuple=True)[0]
        subs.append((x[idx].to(device), sub_ei.to(device), y[idx].to(device)))

    t0 = time.perf_counter()
    for _ in range(args.n_epochs):
        for sx, sei, sy in subs:
            tm = torch.zeros(sx.shape[0], dtype=torch.bool, device=device)
            tm[torch.randperm(sx.shape[0], device=device)[: int(0.7 * sx.shape[0])]] = True
            model.train()
            opt.zero_grad()
            out = model(sx, sei)
            tsb_loss(out, sy, task_type="classification", valid_mask=tm,
                     lambda_l0=5.0, target_edge_rate=0.2,
                     budget_mode="squared")["total"].backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / args.n_epochs
    return {"mode": "patched", "side": side, "n_nodes": n, "n_edges": m,
            "n_tiles": len(subs), "tile_nodes": args.tile_nodes,
            "sec_per_epoch": sec, "peak_mem_mb": peak_memory_mb(device)}


def accuracy_full_vs_patched(side, device, args):
    """What does tiling cost in accuracy, at a size where both modes are feasible?"""
    from sklearn.metrics import f1_score

    x, edge_index, y, coords, n_classes = build_lattice(side, k=args.k)
    n = x.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[: int(0.7 * n)]] = True
    test_idx = perm[int(0.7 * n):]

    def evaluate(model, xs, eis, index_sets):
        model.eval()
        pred = torch.empty(n, dtype=torch.long)
        with torch.no_grad():
            for xi, ei, idx in zip(xs, eis, index_sets):
                pred[idx] = model(xi.to(device), ei.to(device))["logits"] \
                    .argmax(dim=-1).cpu()
        keep = test_idx
        return float(f1_score(y[keep].numpy(), pred[keep].numpy(), average="macro"))

    # Full graph.
    torch.manual_seed(0)
    model = TargetedSpatialBottleneck(
        input_dim=x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = int(0.2 * args.accuracy_epochs)
    xd, eid, yd, tmd = (x.to(device), edge_index.to(device), y.to(device),
                        train_mask.to(device))
    for epoch in range(args.accuracy_epochs):
        model.train()
        opt.zero_grad()
        out = model(xd, eid)
        ws = min(1.0, (epoch + 1) / warmup)
        tsb_loss(out, yd, task_type="classification", valid_mask=tmd,
                 lambda_l0=5.0 * ws, target_edge_rate=0.2,
                 budget_mode="squared")["total"].backward()
        opt.step()
    f1_full = evaluate(model, [x], [edge_index], [torch.arange(n)])

    # Tiled.
    tiles = spatial_tiles(coords, args.tile_nodes)
    subs = []
    for t in np.unique(tiles):
        node_mask = torch.tensor(tiles == t)
        if node_mask.sum() < 10:
            continue
        idx = node_mask.nonzero(as_tuple=True)[0]
        subs.append((x[idx], induced_subgraph(edge_index, node_mask), idx))

    torch.manual_seed(0)
    model = TargetedSpatialBottleneck(
        input_dim=x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(args.accuracy_epochs):
        ws = min(1.0, (epoch + 1) / warmup)
        for sx, sei, idx in subs:
            model.train()
            opt.zero_grad()
            out = model(sx.to(device), sei.to(device))
            tsb_loss(out, y[idx].to(device), task_type="classification",
                     valid_mask=train_mask[idx].to(device),
                     lambda_l0=5.0 * ws, target_edge_rate=0.2,
                     budget_mode="squared")["total"].backward()
            opt.step()
    f1_patched = evaluate(model, [s[0] for s in subs], [s[1] for s in subs],
                          [s[2] for s in subs])

    return {"mode": "accuracy", "side": side, "n_nodes": n, "n_tiles": len(subs),
            "macro_f1_full": f1_full, "macro_f1_patched": f1_patched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--accuracy-side", type=int, default=100)
    parser.add_argument("--accuracy-epochs", type=int, default=300)
    parser.add_argument("--patched-sides", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--tile-nodes", type=int, default=4000)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    results = []

    for side in args.sides:
        try:
            r = run_full(side, device, args)
            results.append(r)
            print(f"full   side={side:4d} n={r['n_nodes']:>8d} |E|={r['n_edges']:>9d} "
                  f"{r['sec_per_epoch']*1000:9.1f} ms/epoch  "
                  f"{r['peak_mem_mb']:8.1f} MB", flush=True)
        except (RuntimeError, MemoryError) as exc:
            results.append({"mode": "full", "side": side, "error": str(exc)[:200]})
            print(f"full   side={side:4d} FAILED: {str(exc)[:120]}", flush=True)

    for side in args.patched_sides:
        try:
            r = run_patched(side, device, args)
            results.append(r)
            print(f"patch  side={side:4d} n={r['n_nodes']:>8d} tiles={r['n_tiles']:>4d} "
                  f"{r['sec_per_epoch']*1000:9.1f} ms/epoch  "
                  f"{r['peak_mem_mb']:8.1f} MB", flush=True)
        except (RuntimeError, MemoryError) as exc:
            results.append({"mode": "patched", "side": side, "error": str(exc)[:200]})
            print(f"patch  side={side:4d} FAILED: {str(exc)[:120]}", flush=True)

    if args.accuracy_side:
        r = accuracy_full_vs_patched(args.accuracy_side, device, args)
        results.append(r)
        print(f"\naccuracy side={r['side']} ({r['n_nodes']} nodes, {r['n_tiles']} tiles): "
              f"full graph macro F1 {r['macro_f1_full']:.4f} | "
              f"tiled {r['macro_f1_patched']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"device": args.device, "results": results}, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
