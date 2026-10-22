'Is target conditioning specific to TSB, or does any learned mask have it?'

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, ".")
from nicheib.baselines import PlainGAT
from nicheib.gates import HardConcreteSampler

GENE_PROGRAMS = {
    "excitatory_neuron": ["SYT1", "SYN1", "SNAP25", "SLC17A7", "SLC17A6", "NRGN",
                          "CAMK2A", "GRIA1", "GRIA2", "GRIN1", "GRIN2B", "RBFOX3"],
    "oligodendrocyte": ["MBP", "PLP1", "MOG", "MAG", "MOBP", "CLDN11", "CNP",
                        "OLIG1", "OLIG2", "SOX10", "ERMN", "FA2H"],
    "astrocyte": ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "S100B",
                  "GJA1", "GLUL", "NDRG2", "CLU", "APOE"],
    "microglia": ["AIF1", "CSF1R", "CX3CR1", "TMEM119", "P2RY12", "ITGAM",
                  "CD68", "TREM2", "TYROBP", "C1QA", "C1QB", "C1QC"],
    "layer1_markers": ["RELN", "NDNF", "LAMP5", "LHX5"],
    "deep_layer_L5L6": ["PCP4", "BCL11B", "NTNG2", "TLE4", "FOXP2", "FEZF2",
                        "CTIP2", "CRYM", "HTR2C"],
}

# Targets TSB can actually predict on DLPFC; the paper's 0.172 figure is over these.
LEARNABLE = ["layer_label", "excitatory_neuron", "oligodendrocyte"]


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class _Gate(nn.Module):
    def __init__(self, dim, hidden=64, init_bias=3.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, h, edge_index):
        s, d = edge_index
        return self.net(torch.cat([h[s], h[d], h[s] * h[d]], -1)).squeeze(-1)


class _GATReg(nn.Module):
    """Gate-free GAT with a scalar head, so GraphMask can explain a regression target."""

    def __init__(self, in_dim, hidden=64, latent=32):
        super().__init__()
        from nicheib.tsb import GatedGraphConv

        self.encoder = nn.Linear(in_dim, hidden)
        self.conv1 = GatedGraphConv(hidden, hidden, heads=4)
        self.conv2 = GatedGraphConv(hidden, latent, heads=4)
        self.head = nn.Linear(latent, 1)

    def forward(self, x, edge_index):
        ones = x.new_ones(edge_index.shape[1])
        h = F.relu(self.encoder(x))
        h = F.relu(self.conv1(h, edge_index, ones))
        return {"prediction": self.head(self.conv2(h, edge_index, ones)).squeeze(-1)}


def train_backbone(data, target, task, n_classes, seed, epochs, lr=1e-3):
    """Frozen model GraphMask will explain: a gate-free GAT fit to this target."""
    seed_everything(seed)
    n = data.x.shape[0]
    labeled = (target >= 0) if task == "classification" else torch.isfinite(target)
    idx = labeled.nonzero(as_tuple=True)[0]
    perm = idx[torch.randperm(idx.shape[0])]
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[: int(0.7 * idx.shape[0])]] = True

    model = (PlainGAT(data.x.shape[1], 64, 32, n_classes, task_level="node")
             if task == "classification" else _GATReg(data.x.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        if task == "classification":
            loss = F.cross_entropy(out["logits"][train_mask], target[train_mask])
        else:
            loss = F.mse_loss(out["prediction"][train_mask], target[train_mask].float())
        loss.backward()
        opt.step()
    model.eval()
    return model, task


def graphmask(model, task, x, edge_index, epochs=300, lr=1e-3, tol=0.05, dual_lr=0.1,
              target_rate=0.2, rho=20.0):
    """Amortised per-layer Hard Concrete mask under a divergence constraint.

    Sparsity is anchored at `target_rate` rather than pushed to zero: an unanchored
    Lagrangian drives the mask to a constant (all-open or all-closed) on some targets,
    which makes a rank correlation between masks undefined and is not the operating
    point TSB is compared at.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        out = model(x, edge_index)
        ref = (F.softmax(out["logits"], -1) if task == "classification"
               else out["prediction"])
    dim = F.relu(model.encoder(x)).shape[1]
    g1, g2 = _Gate(dim), _Gate(dim)
    smp = HardConcreteSampler(2.0 / 3.0, -0.1, 1.1)
    lam = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(list(g1.parameters()) + list(g2.parameters()), lr=lr)
    opt_l = torch.optim.Adam([lam], lr=dual_lr)

    def fwd(sample):
        h = F.relu(model.encoder(x))
        l1 = g1(h, edge_index)
        m1 = smp.sample(l1) if sample else smp.deterministic(l1)
        h = F.relu(model.conv1(h, edge_index, m1))
        l2 = g2(h, edge_index)
        m2 = smp.sample(l2) if sample else smp.deterministic(l2)
        z = model.conv2(h, edge_index, m2)
        return model.head(z), l1, l2, m1, m2

    for _ in range(epochs):
        opt.zero_grad()
        opt_l.zero_grad()
        raw, l1, l2, _, _ = fwd(True)
        if task == "classification":
            div = F.kl_div(F.log_softmax(raw, -1), ref, reduction="batchmean")
        else:
            div = F.mse_loss(raw.squeeze(-1), ref)
        spars = 0.5 * (smp.expected_l0(l1).mean() + smp.expected_l0(l2).mean())
        budget = (spars - target_rate).pow(2)
        (rho * budget + F.softplus(lam) * (div - tol)).backward()
        opt.step()
        lam.grad = -lam.grad
        opt_l.step()

    with torch.no_grad():
        _, _, _, m1, m2 = fwd(False)
    for p in model.parameters():
        p.requires_grad_(True)
    return ((m1 + m2) / 2).cpu().numpy()


def safe_spearman(a, b):
    """Spearman that reports a degenerate mask instead of silently returning nan."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    rho, _ = stats.spearmanr(a, b)
    return float(rho)


def off_diag_mean(mats, idx):
    iu = np.triu_indices(len(idx), 1)
    vals = np.concatenate([m[np.ix_(idx, idx)][iu] for m in mats])
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def split_half_reliability(per_seed):
    """Reliability of two disjoint, equally sized seed-consensus masks."""
    n_seeds = per_seed.shape[0]
    half = n_seeds // 2
    if half < 1:
        return float("nan")
    return safe_spearman(
        per_seed[:half].mean(axis=0),
        per_seed[half:2 * half].mean(axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        default="data/dlpfc")
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import scanpy as sc
    from nicheib.data import load_dlpfc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_mats, perm_mats, order = [], [], None

    for section in args.sections:
        adata = sc.read_h5ad(f"{args.data_dir}/{section}.h5ad")
        progs = {}
        for name, genes in GENE_PROGRAMS.items():
            avail = [g for g in genes if g in adata.var_names]
            if len(avail) < 3:
                continue
            sc.tl.score_genes(adata, gene_list=avail, score_name="_s")
            v = torch.tensor(adata.obs["_s"].values.astype(np.float32))
            del adata.obs["_s"]
            progs[name] = (v - v.mean()) / (v.std() + 1e-8)

        data = load_dlpfc(section, data_dir=args.data_dir,
                          n_hvgs=1000, n_pcs=64, n_neighbors=6)
        n_classes = len(data.label_names)
        targets = ["layer_label"] + list(progs)
        order = order or targets
        print(f"\n=== {section}: {data.x.shape[0]} spots, "
              f"{data.edge_index.shape[1]} edges ===", flush=True)

        checkpoint_path = out_dir / f"checkpoint_{section}.npz"
        section_payload = {"target_names": np.array(targets),
                           "seeds": np.array(args.seeds)}
        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            section_payload.update({key: checkpoint[key]
                                    for key in checkpoint.files})
            print(f"  resumed checkpoint {checkpoint_path}", flush=True)
        for condition in ("real", "permuted"):
            masks = {}
            per_seed_masks = {}
            for t in targets:
                checkpoint_key = f"{condition}__{t}"
                if (checkpoint_key in section_payload
                        and section_payload[checkpoint_key].shape[0] == len(args.seeds)):
                    per_seed_masks[t] = section_payload[checkpoint_key]
                    print(f"  {condition:9s} {t:20s} resumed", flush=True)
                else:
                    per_seed = []
                    for seed in args.seeds:
                        rng = np.random.default_rng(1000 + seed)
                        if t == "layer_label":
                            y = data.y.clone()
                            if condition == "permuted":
                                lab = (data.y >= 0).nonzero(as_tuple=True)[0]
                                y[lab] = data.y[lab[torch.tensor(
                                    rng.permutation(len(lab)))]]
                            task, tgt, nc = "classification", y, n_classes
                        else:
                            v = progs[t]
                            if condition == "permuted":
                                v = v[torch.tensor(rng.permutation(len(v)))]
                            task, tgt, nc = "regression", v, 1
                        model, task = train_backbone(
                            data, tgt, task, nc, seed, args.epochs
                        )
                        per_seed.append(graphmask(
                            model, task, data.x, data.edge_index,
                            epochs=args.epochs,
                        ))
                    per_seed_masks[t] = np.stack(per_seed)
                    section_payload[checkpoint_key] = per_seed_masks[t]
                    np.savez_compressed(checkpoint_path, **section_payload)
                masks[t] = per_seed_masks[t].mean(axis=0)
                print(f"  {condition:9s} {t:20s} done ({len(args.seeds)} seeds, "
                      f"mean mask {masks[t].mean():.3f}, sd {masks[t].std():.3f})",
                      flush=True)

            n_t = len(targets)
            mat = np.eye(n_t)
            degenerate = [t for t in targets if np.std(masks[t]) == 0]
            if degenerate:
                print("    WARNING degenerate (constant) masks: %s" % degenerate,
                      flush=True)
            for i, j in itertools.combinations(range(n_t), 2):
                mat[i, j] = mat[j, i] = safe_spearman(masks[targets[i]],
                                                      masks[targets[j]])
            (real_mats if condition == "real" else perm_mats).append(mat)
            section_payload[f"{condition}_matrix"] = mat
            section_payload[f"{condition}_masks"] = np.stack(
                [per_seed_masks[t] for t in targets]
            )
            section_payload[f"{condition}_split_half"] = np.array(
                [split_half_reliability(per_seed_masks[t]) for t in targets]
            )

        np.savez_compressed(out_dir / f"gm_cond_{section}.npz",
                            **section_payload)

    learn_idx = [i for i, t in enumerate(order) if t in LEARNABLE]
    all_idx = list(range(len(order)))
    summary = {
        "n_sections": len(real_mats),
        "targets": order,
        "graphmask_real_offdiag_all": off_diag_mean(real_mats, all_idx),
        "graphmask_permuted_offdiag_all": off_diag_mean(perm_mats, all_idx),
        "graphmask_real_offdiag_learnable": off_diag_mean(real_mats, learn_idx),
        "graphmask_permuted_offdiag_learnable": off_diag_mean(perm_mats, learn_idx),
        "tsb_reference_learnable_real": 0.172,
        "tsb_reference_learnable_permuted": 0.633,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Target conditioning: GraphMask vs TSB ===")
    print("%-34s %10s %10s %10s" % ("", "real", "permuted", "contrast"))
    print("%-34s %10.3f %10.3f %10.3f" % (
        "GraphMask, learnable targets",
        summary["graphmask_real_offdiag_learnable"],
        summary["graphmask_permuted_offdiag_learnable"],
        summary["graphmask_permuted_offdiag_learnable"]
        - summary["graphmask_real_offdiag_learnable"]))
    print("%-34s %10.3f %10.3f %10.3f" % (
        "GraphMask, all targets",
        summary["graphmask_real_offdiag_all"],
        summary["graphmask_permuted_offdiag_all"],
        summary["graphmask_permuted_offdiag_all"]
        - summary["graphmask_real_offdiag_all"]))
    print("%-34s %10.3f %10.3f %10.3f" % (
        "TSB, learnable targets (reference)", 0.172, 0.633, 0.461))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
