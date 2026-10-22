'E1: Multi-seed gate stability, consensus maps, and held-out marker targets (DLPFC).'

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from nicheib.tsb import TargetedSpatialBottleneck
from nicheib.loss import tsb_loss

# Marker sets: identical to run_target_dependent.py so results stay comparable.
GENE_PROGRAMS = {
    "excitatory_neuron": [
        "SYT1", "SYN1", "SNAP25", "SLC17A7", "SLC17A6", "NRGN",
        "CAMK2A", "GRIA1", "GRIA2", "GRIN1", "GRIN2B", "RBFOX3",
    ],
    "oligodendrocyte": [
        "MBP", "PLP1", "MOG", "MAG", "MOBP", "CLDN11", "CNP",
        "OLIG1", "OLIG2", "SOX10", "ERMN", "FA2H",
    ],
    "astrocyte": [
        "GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "S100B",
        "GJA1", "GLUL", "NDRG2", "CLU", "APOE",
    ],
    "microglia": [
        "AIF1", "CSF1R", "CX3CR1", "TMEM119", "P2RY12", "ITGAM",
        "CD68", "TREM2", "TYROBP", "C1QA", "C1QB", "C1QC",
    ],
    "layer1_markers": ["RELN", "NDNF", "LAMP5", "LHX5"],
    "deep_layer_L5L6": [
        "PCP4", "BCL11B", "NTNG2", "TLE4", "FOXP2", "FEZF2",
        "CTIP2", "CRYM", "HTR2C",
    ],
}

# An extended panel used for the higher-powered version of the target-specificity and
# Mantel analyses: 7 targets give only 21 cross-target pairs, which is too few to test
# whether gate similarity tracks an external reference, and too few to characterise the
# specificity-versus-learnability relation. These span a deliberate range of expected
# learnability, from broad layer programmes to sparse interneuron subtypes.
EXTENDED_PROGRAMS = {
    **GENE_PROGRAMS,
    "layer2_3": ["CUX2", "CALB1", "LAMP5", "CARTPT", "RASGRF2"],
    "layer4": ["RORB", "NEFH", "SCN3B", "GRIK4"],
    "layer5": ["FEZF2", "BCL11B", "ETV1", "PCP4", "HTR2C"],
    "layer6": ["FOXP2", "TLE4", "SYT6", "NTNG2", "OPRK1"],
    "white_matter": ["MBP", "MOBP", "PLP1", "CNP", "MAL", "GFAP"],
    "opc": ["PDGFRA", "CSPG4", "SOX10", "OLIG1", "OLIG2"],
    "endothelial": ["CLDN5", "FLT1", "PECAM1", "VWF", "RGS5"],
    "inhibitory_neuron": ["GAD1", "GAD2", "SLC32A1", "SST", "PVALB", "VIP"],
    "synaptic_vesicle": ["SYP", "SYN1", "SNAP25", "VAMP2", "STX1A"],
    "mitochondrial": ["MT-CO1", "MT-CO2", "MT-ND1", "MT-ATP6", "COX4I1"],
    "ribosomal": ["RPL13", "RPS18", "RPL3", "RPS6", "RPLP1"],
    "stress_response": ["HSPA1A", "HSPB1", "FOS", "JUN", "EGR1"],
}

# Programs for which a held-out-marker target is trained (need >= 6 genes to split).
HELDOUT_PROGRAMS = ["excitatory_neuron", "oligodendrocyte", "astrocyte", "microglia"]

# Genes whose expression we persist for downstream Moran's I / ligand-receptor checks.
VALIDATION_GENES = sorted({
    g
    for genes in GENE_PROGRAMS.values()
    for g in genes
} | {
    # ligand-receptor partners from Appendix D that are not already marker genes
    "RTN4R", "GALC", "GPR37", "CD44", "RAGE", "AGER", "DAG1",
    "CX3CL1", "C1QBP", "TYROBP", "MFGE8", "GRIN2A", "HOMER1",
    "BDNF", "NTRK2", "NRXN1", "NLGN1", "SATB2", "CUX2", "RORB",
})


def split_markers(genes: list[str]) -> tuple[list[str], list[str]]:
    """Deterministically split a marker set into two disjoint halves."""
    ordered = sorted(genes)
    return ordered[0::2], ordered[1::2]


def score_gene_program(adata, gene_list: list[str], min_genes: int = 3):
    import scanpy as sc

    available = [g for g in gene_list if g in adata.var_names]
    if len(available) < min_genes:
        return None, available
    sc.tl.score_genes(adata, gene_list=available, score_name="_tmp_score")
    scores = adata.obs["_tmp_score"].values.copy().astype(np.float32)
    del adata.obs["_tmp_score"]
    return scores, available


def _train_mask(n: int, labeled_idx: torch.Tensor, frac: float = 0.7) -> torch.Tensor:
    perm = labeled_idx[torch.randperm(labeled_idx.shape[0])]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[perm[: int(frac * labeled_idx.shape[0])]] = True
    return mask


def coverage_penalty(expected_l0, edge_index, n_nodes, d_min):
    """Penalize nodes whose expected retained in-degree falls below d_min."""
    degree = torch.zeros(n_nodes, device=expected_l0.device)
    degree.scatter_add_(0, edge_index[1], expected_l0)
    return F.relu(d_min - degree).pow(2).mean()


def bottleneck_objective(model, out, prediction_loss, epoch, n_epochs, target_er,
                         variant, state, lambda_l0=5.0, rho=20.0,
                         dual_lr=0.1, lambda_cov=5.0, coverage_frac=0.75):
    """Return the training objective and update state shared by all target types."""
    warmup = max(1, int(0.2 * n_epochs))
    warmup_scale = min(1.0, (epoch + 1) / warmup)
    expected_l0 = out["expected_l0"]
    gap = expected_l0.mean() - target_er
    if variant == "submitted":
        total = prediction_loss + lambda_l0 * warmup_scale * gap.pow(2)
    else:
        total = prediction_loss + warmup_scale * (
            state["mu"] * gap + 0.5 * rho * gap.pow(2)
        )
        if variant == "final":
            n_nodes = state["n_nodes"]
            n_edges = expected_l0.numel()
            d_min = coverage_frac * target_er * (n_edges / n_nodes)
            total = total + lambda_cov * warmup_scale * coverage_penalty(
                expected_l0, state["edge_index"], n_nodes, d_min
            )
    state["gap"] = gap
    state["warmup"] = warmup
    state["dual_lr"] = dual_lr
    return total


def update_dual(model, epoch, variant, state):
    """Apply the projected dual update after the primal optimizer step."""
    if variant == "submitted":
        return
    torch.nn.utils.clip_grad_norm_(
        list(model.gate1.parameters()) + list(model.gate2.parameters()), 1.0
    )
    if epoch >= state["warmup"]:
        state["mu"] = float(np.clip(
            state["mu"] + state["dual_lr"] * float(state["gap"].detach()),
            0.0,
            50.0,
        ))


def train_regression(data, target, seed, n_epochs=300, lr=1e-3,
                     lambda_l0=5.0, target_er=0.2, variant="submitted"):
    """Train TSB on a continuous target; returns (edge_scores, r2, edge_rate)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = data.x.shape[0]
    labeled = torch.isfinite(target)
    train_mask = _train_mask(n, labeled.nonzero(as_tuple=True)[0])

    model = TargetedSpatialBottleneck(
        input_dim=data.x.shape[1], hidden_dim=64, latent_dim=32, output_dim=1,
        task_type="regression", use_xib=False, gate_init_bias=3.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    state = {"mu": 0.0, "edge_index": data.edge_index,
             "n_nodes": data.x.shape[0]}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        prediction_loss = F.mse_loss(
            out["prediction"].squeeze()[train_mask], target[train_mask]
        )
        total = bottleneck_objective(
            model, out, prediction_loss, epoch, n_epochs, target_er, variant,
            state, lambda_l0=lambda_l0,
        )
        total.backward()
        update_dual(model, epoch, variant, state)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out["prediction"].squeeze().cpu().numpy()
        y_np = target.cpu().numpy()
        test = (~train_mask & labeled).cpu().numpy()
        ss_res = np.sum((y_np[test] - pred[test]) ** 2)
        ss_tot = np.sum((y_np[test] - y_np[test].mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)
        return out["edge_prob"].cpu().numpy().astype(np.float32), r2, float(out["edge_rate"])


def train_classification(data, seed, y=None, n_epochs=300, lr=1e-3,
                         lambda_l0=5.0, target_er=0.2, variant="submitted"):
    """Train TSB on layer labels; returns (edge_scores, macro_f1, edge_rate)."""
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    y = data.y if y is None else y
    n = data.x.shape[0]
    n_classes = len(data.label_names) if hasattr(data, "label_names") else 7
    labeled = y >= 0
    train_mask = _train_mask(n, labeled.nonzero(as_tuple=True)[0])

    model = TargetedSpatialBottleneck(
        input_dim=data.x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    state = {"mu": 0.0, "edge_index": data.edge_index,
             "n_nodes": data.x.shape[0]}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        prediction_loss = F.cross_entropy(out["logits"][train_mask], y[train_mask])
        total = bottleneck_objective(
            model, out, prediction_loss, epoch, n_epochs, target_er, variant,
            state, lambda_l0=lambda_l0,
        )
        total.backward()
        update_dual(model, epoch, variant, state)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out["logits"].argmax(dim=-1).cpu().numpy()
        y_np = y.cpu().numpy()
        test = (labeled & ~train_mask).cpu().numpy()
        f1 = float(f1_score(y_np[test], pred[test], average="macro"))
        return out["edge_prob"].cpu().numpy().astype(np.float32), f1, float(out["edge_rate"])


def zscore(v: np.ndarray) -> torch.Tensor:
    t = torch.tensor(v, dtype=torch.float32)
    return (t - t.mean()) / (t.std() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="data/dlpfc")
    parser.add_argument("--out-dir", type=str,
                        default="results/consensus_conditioning")
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--n-seeds-real", type=int, default=10)
    parser.add_argument("--n-seeds-perm", type=int, default=5)
    parser.add_argument("--n-seeds-heldout", type=int, default=3)
    parser.add_argument("--extended-programs", action="store_true",
                        help="Use the 18-programme panel instead of the 6 in the paper")
    parser.add_argument("--target-er", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    # Paper names: fixed = TSB-fixed, dual = TSB-dual, cov = TSB-cov. The older
    # names "submitted" and "final" are kept as aliases so archived records still match.
    parser.add_argument("--variant", choices=["fixed", "dual", "cov", "submitted", "final"],
                        default="fixed")
    args = parser.parse_args()
    args.variant = {"fixed": "submitted", "cov": "final"}.get(args.variant, args.variant)

    import scanpy as sc
    from nicheib.data import load_dlpfc

    programs = EXTENDED_PROGRAMS if args.extended_programs else GENE_PROGRAMS

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = []

    for section in args.sections:
        print(f"\n{'=' * 60}\nSection {section}\n{'=' * 60}", flush=True)
        adata = sc.read_h5ad(f"{args.data_dir}/{section}.h5ad")

        # --- target construction -------------------------------------------------
        program_scores, program_genes = {}, {}
        for name, genes in programs.items():
            scores, used = score_gene_program(adata, genes)
            if scores is not None:
                program_scores[name] = scores
                program_genes[name] = used
                print(f"  program {name}: {len(used)}/{len(genes)} genes", flush=True)

        heldout_scores, heldout_eval_genes = {}, {}
        for name in HELDOUT_PROGRAMS:
            if name not in program_genes:
                continue
            train_half, eval_half = split_markers(program_genes[name])
            scores, used = score_gene_program(adata, train_half)
            if scores is not None and len(eval_half) >= 2:
                heldout_scores[name] = scores
                heldout_eval_genes[name] = eval_half
                print(f"  heldout {name}: train on {used}, evaluate on {eval_half}",
                      flush=True)

        # Persist normalised expression of the validation genes before HVG filtering.
        norm = adata.copy()
        sc.pp.normalize_total(norm, target_sum=1e4)
        sc.pp.log1p(norm)
        expr = {}
        for gene in VALIDATION_GENES:
            if gene in norm.var_names:
                col = norm[:, gene].X
                col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") \
                    else np.asarray(col).ravel()
                expr[f"expr__{gene}"] = col.astype(np.float32)
        del norm

        data = load_dlpfc(section, data_dir=args.data_dir,
                          n_hvgs=1000, n_pcs=64, n_neighbors=6)
        n_edges = data.edge_index.shape[1]
        print(f"  graph: {data.x.shape[0]} spots, {n_edges} edges", flush=True)

        checkpoint_path = outdir / f"checkpoint_{section}.npz"
        records_path = outdir / f"records_{section}.json"
        payload = {
            "edge_index": data.edge_index.cpu().numpy().astype(np.int32),
            "coords": data.coords.cpu().numpy().astype(np.float32),
            "y": data.y.cpu().numpy().astype(np.int32),
            **expr,
        }
        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            payload.update({key: checkpoint[key] for key in checkpoint.files})
            print(f"  resumed checkpoint {checkpoint_path}", flush=True)
        if records_path.exists():
            records.extend(json.loads(records_path.read_text()))

        def save_checkpoint():
            np.savez_compressed(checkpoint_path, **payload)
            records_path.write_text(json.dumps(
                [record for record in records if record["section"] == section],
                indent=2,
            ))

        target_names = ["layer_label"] + list(program_scores.keys())
        payload["target_names"] = np.array(target_names)
        payload["target_scores"] = np.stack(
            [np.zeros(data.x.shape[0], dtype=np.float32)]
            + [program_scores[t] for t in program_scores],
            axis=1,
        )

        # --- real targets, many seeds -------------------------------------------
        for target in target_names:
            key = f"real__{target}"
            if key in payload and payload[key].shape[0] == args.n_seeds_real:
                print(f"  real {target:>18s}: resumed", flush=True)
                continue
            maps = []
            for seed in range(args.n_seeds_real):
                if target == "layer_label":
                    es, val, er = train_classification(
                        data, seed, n_epochs=args.epochs, target_er=args.target_er,
                        variant=args.variant)
                    metric = "f1"
                else:
                    es, val, er = train_regression(
                        data, zscore(program_scores[target]), seed,
                        n_epochs=args.epochs, target_er=args.target_er,
                        variant=args.variant)
                    metric = "r2"
                maps.append(es)
                records.append({"section": section, "variant": args.variant,
                                "condition": "real", "target": target,
                                "seed": seed, "metric": metric, "value": val, "er": er})
            payload[f"real__{target}"] = np.stack(maps)
            save_checkpoint()
            vals = [r["value"] for r in records
                    if r["target"] == target and r["condition"] == "real"
                    and r["section"] == section]
            print(f"  real {target:>18s}: {np.mean(vals):+.3f} "
                  f"(n={len(vals)} seeds)", flush=True)

        # --- permuted-target control --------------------------------------------
        for target in target_names:
            key = f"perm__{target}"
            if key in payload and payload[key].shape[0] == args.n_seeds_perm:
                print(f"  perm {target:>18s}: resumed", flush=True)
                continue
            maps = []
            for seed in range(args.n_seeds_perm):
                rng = np.random.default_rng(1000 + seed)
                if target == "layer_label":
                    y_perm = data.y.clone()
                    lab = (data.y >= 0).nonzero(as_tuple=True)[0]
                    y_perm[lab] = data.y[lab[torch.tensor(rng.permutation(len(lab)))]]
                    es, val, er = train_classification(
                        data, seed, y=y_perm, n_epochs=args.epochs,
                        target_er=args.target_er, variant=args.variant)
                    metric = "f1"
                else:
                    shuffled = program_scores[target][rng.permutation(
                        len(program_scores[target]))]
                    es, val, er = train_regression(
                        data, zscore(shuffled), seed, n_epochs=args.epochs,
                        target_er=args.target_er, variant=args.variant)
                    metric = "r2"
                maps.append(es)
                records.append({"section": section, "variant": args.variant,
                                "condition": "permuted",
                                "target": target, "seed": seed, "metric": metric,
                                "value": val, "er": er})
            payload[f"perm__{target}"] = np.stack(maps)
            save_checkpoint()
            print(f"  perm {target:>18s}: done ({args.n_seeds_perm} seeds)", flush=True)

        # --- held-out marker targets ---------------------------------------------
        for name, scores in (heldout_scores.items()
                             if args.n_seeds_heldout > 0 else []):
            key = f"heldout__{name}"
            if key in payload and payload[key].shape[0] == args.n_seeds_heldout:
                print(f"  heldout {name:>15s}: resumed", flush=True)
                continue
            maps = []
            for seed in range(args.n_seeds_heldout):
                es, val, er = train_regression(
                    data, zscore(scores), seed, n_epochs=args.epochs,
                    target_er=args.target_er, variant=args.variant)
                maps.append(es)
                records.append({"section": section, "variant": args.variant,
                                "condition": "heldout", "target": name,
                                "seed": seed, "metric": "r2", "value": val, "er": er})
            payload[f"heldout__{name}"] = np.stack(maps)
            payload[f"heldout_eval_genes__{name}"] = np.array(heldout_eval_genes[name])
            save_checkpoint()
            print(f"  heldout {name:>15s}: done", flush=True)

        np.savez_compressed(outdir / f"stability_{section}.npz", **payload)
        with records_path.open("w") as f:
            json.dump([r for r in records if r["section"] == section], f, indent=2)
        print(f"  saved {outdir / f'stability_{section}.npz'}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
