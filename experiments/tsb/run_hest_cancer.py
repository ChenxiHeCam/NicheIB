'Run TSB target-dependent analysis on HEST cancer datasets.'
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, ".")
from nicheib.data import load_hest_subset
from nicheib.tsb import TargetedSpatialBottleneck
from nicheib.loss import tsb_loss

CANCER_PROGRAMS = {
    "tumor_epithelial": [
        "KRT8", "KRT18", "KRT19", "EPCAM", "CDH1", "MUC1",
        "ESR1", "PGR", "ERBB2", "GATA3", "FOXA1",
    ],
    "immune_infiltrate": [
        "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "PTPRC",
        "CD68", "CD163", "MS4A1", "CD19", "FOXP3", "GZMB",
    ],
    "stromal_fibroblast": [
        "VIM", "COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP",
        "DCN", "LUM", "PDGFRA", "THY1",
    ],
    "proliferation": [
        "MKI67", "TOP2A", "PCNA", "BIRC5", "AURKA", "CDK1",
        "CCNB1", "CCNA2", "TYMS", "RRM2",
    ],
}

KIDNEY_PROGRAMS = {
    "tumor_ccrcc": [
        "CA9", "PAX8", "PAX2", "CAIX", "VHL", "HIF1A",
        "VEGFA", "SLC2A1", "NDUFA4L2", "ENPP3",
    ],
    "immune_infiltrate": CANCER_PROGRAMS["immune_infiltrate"],
    "endothelial": [
        "PECAM1", "CDH5", "VWF", "ERG", "FLT1", "KDR",
        "EMCN", "PLVAP", "CLDN5",
    ],
    "stromal": CANCER_PROGRAMS["stromal_fibroblast"],
    "normal_tubule": [
        "AQP1", "AQP2", "SLC12A1", "SLC12A3", "UMOD",
        "SLC34A1", "LRP2",
    ],
}


def score_gene_program(adata, gene_list):
    import scanpy as sc
    available = [g for g in gene_list if g in adata.var_names]
    if len(available) < 3:
        return None
    sc.tl.score_genes(adata, gene_list=available, score_name="_tmp")
    scores = adata.obs["_tmp"].values.copy().astype(np.float32)
    del adata.obs["_tmp"]
    return scores


def train_and_analyze(data, target, target_type, n_epochs=300, lr=1e-3,
                      lambda_l0=5.0, target_er=0.2, seed=0, budget="penalty",
                      rho=20.0, dual_lr=0.1):
    """budget: 'penalty' reproduces the submission's fixed squared penalty;
    'dual' solves the budget as a constraint by projected dual ascent on the
    multiplier, which is what makes the budget bind on regression targets."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = data.x.shape[1]
    if target_type == "regression":
        out_dim = 1
    else:
        out_dim = int(target.max()) + 1

    model = TargetedSpatialBottleneck(
        input_dim=n_features,
        hidden_dim=64,
        latent_dim=32,
        output_dim=out_dim,
        task_type=target_type,
        use_xib=False,
        gate_init_bias=3.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = int(0.2 * n_epochs)

    n = data.x.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:int(0.7 * n)]] = True
    test_mask = ~train_mask

    model.train()
    mu = 0.0
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        ws = min(1.0, (epoch + 1) / warmup)
        if budget == "penalty":
            loss_dict = tsb_loss(
                out, target,
                task_type=target_type,
                valid_mask=train_mask,
                lambda_l0=lambda_l0 * ws,
                target_edge_rate=target_er,
                budget_mode="squared",
            )
            total = loss_dict["total"]
        else:
            # Augmented Lagrangian: the multiplier carries the constraint, the
            # quadratic term only stabilises the early iterations.
            loss_dict = tsb_loss(
                out, target,
                task_type=target_type,
                valid_mask=train_mask,
                lambda_l0=0.0,
                target_edge_rate=target_er,
                budget_mode="squared",
            )
            gap = out["expected_l0"].mean() - target_er
            total = loss_dict["task"] + ws * (mu * gap + 0.5 * rho * gap.pow(2))
        total.backward()
        if budget != "penalty":
            # A gate whose logit has run far negative has no gradient left, so it can
            # never reopen; clipping keeps the logits where the budget term can act.
            torch.nn.utils.clip_grad_norm_(
                list(model.gate1.parameters()) + list(model.gate2.parameters()), 1.0)
        optimizer.step()
        if budget != "penalty" and epoch >= warmup:
            mu = float(np.clip(mu + dual_lr * float(gap.detach()), 0.0, 50.0))

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

    edge_scores = out.get("edge_prob", torch.ones(data.edge_index.shape[1])).cpu().numpy()
    # Two conventions appear in this codebase; report both so they cannot be conflated.
    # `actual_er` counts edges whose deterministic gate is above a threshold; the budget
    # loss is written against the mean expected L0, which is what "achieved retention"
    # should mean when comparing against a requested budget.
    actual_er = float((edge_scores > 0.01).mean())
    achieved_el0 = float(out["expected_l0"].mean())

    if target_type == "regression":
        pred = out["prediction"]
        pred_vals = pred[test_mask].squeeze().cpu().numpy()
        true_vals = target[test_mask].cpu().numpy()
        ss_res = np.sum((pred_vals - true_vals) ** 2)
        ss_tot = np.sum((true_vals - true_vals.mean()) ** 2) + 1e-8
        metric_val = float(1.0 - ss_res / ss_tot)
        metric_name = "r2"
    else:
        from sklearn.metrics import f1_score
        pred = out["logits"]
        pred_labels = pred[test_mask].argmax(dim=1).cpu().numpy()
        true_labels = target[test_mask].cpu().numpy()
        metric_val = float(f1_score(true_labels, pred_labels, average="macro"))
        metric_name = "macro_f1"

    return {
        "edge_scores": edge_scores,
        "metric_name": metric_name,
        "metric_value": metric_val,
        "actual_er": actual_er,
        "achieved_el0": achieved_el0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", choices=["breast_idc", "kidney_sccrcc"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--target-er", type=float, default=0.2)
    parser.add_argument("--lambda-l0", type=float, default=5.0)
    parser.add_argument("--budget", choices=["penalty", "dual"],
                        default="penalty",
                        help="fixed squared penalty (as submitted) or dual ascent")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-sections", type=int, default=15)
    parser.add_argument("--section-idx", type=int, default=None,
                        help="Process only this section index (for SLURM arrays)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading %s from %s..." % (args.dataset, args.data_dir))
    sections = load_hest_subset(args.data_dir, n_hvgs=1000, n_pcs=64)
    sections = [s for s in sections if s.x.shape[0] >= 500]
    if len(sections) > args.max_sections:
        sections = sections[:args.max_sections]

    if args.section_idx is not None:
        if args.section_idx >= len(sections):
            print("Section index %d >= %d, skipping" % (args.section_idx, len(sections)))
            return
        sections = [sections[args.section_idx]]

    print("Using %d sections" % len(sections))
    programs = KIDNEY_PROGRAMS if args.dataset == "kidney_sccrcc" else CANCER_PROGRAMS

    import scanpy as sc

    all_results = {}
    all_edge_scores = {}

    for sec in sections:
        sid = sec.sample_id
        print("\n=== %s (%d spots) ===" % (sid, sec.x.shape[0]))

        adata_path = Path(args.data_dir) / "st" / ("%s.h5ad" % sid)
        if not adata_path.exists():
            adata_path = Path(args.data_dir) / ("%s.h5ad" % sid)
        adata = sc.read_h5ad(adata_path)

        targets = {}
        for prog_name, gene_list in programs.items():
            scores = score_gene_program(adata, gene_list)
            if scores is not None:
                n_found = len([g for g in gene_list if g in adata.var_names])
                targets[prog_name] = ("regression", torch.tensor(scores, dtype=torch.float32))
                print("  %s: %d/%d genes found" % (prog_name, n_found, len(gene_list)))

        sec_scores = {}
        sec_results = {}

        for seed in args.seeds:
            for tname, (ttype, tvals) in targets.items():
                print("  Seed %d, target %s..." % (seed, tname), end=" ", flush=True)
                res = train_and_analyze(
                    sec, tvals, ttype,
                    n_epochs=args.epochs, target_er=args.target_er, seed=seed,
                    lambda_l0=args.lambda_l0, budget=args.budget,
                )
                key = "%s_seed%d" % (tname, seed)
                sec_scores[key] = res["edge_scores"]
                sec_results[key] = {
                    "metric": res["metric_name"],
                    "value": res["metric_value"],
                    "actual_er": res["actual_er"],
                    "achieved_el0": res.get("achieved_el0"),
                    "budget_mode": args.budget,
                }
                print("%s=%.3f, thresholded_ER=%.3f, achieved_L0=%.3f"
                      % (res["metric_name"], res["metric_value"],
                         res["actual_er"], res["achieved_el0"]))

        all_results[sid] = sec_results
        all_edge_scores[sid] = sec_scores

        # Cross-target correlation
        target_names = sorted(targets.keys())
        n_t = len(target_names)
        corr = np.zeros((n_t, n_t))
        for i, t1 in enumerate(target_names):
            for j, t2 in enumerate(target_names):
                rhos = []
                for seed in args.seeds:
                    k1 = "%s_seed%d" % (t1, seed)
                    k2 = "%s_seed%d" % (t2, seed)
                    if k1 in sec_scores and k2 in sec_scores:
                        rho, _ = stats.spearmanr(sec_scores[k1], sec_scores[k2])
                        rhos.append(rho)
                corr[i, j] = np.mean(rhos) if rhos else 0

        all_results[sid]["cross_target_corr"] = {
            "target_names": target_names,
            "matrix": corr.tolist(),
        }

        print("\n  Cross-target correlation:")
        for i, t in enumerate(target_names):
            vals = ["%.2f" % v for v in corr[i]]
            print("  %-25s: %s" % (t, vals))

    suffix = "_sec%d" % args.section_idx if args.section_idx is not None else ""
    with open(out_dir / ("%s_results%s.json" % (args.dataset, suffix)), "w") as f:
        json.dump(all_results, f, indent=2)

    for sid, scores in all_edge_scores.items():
        np.savez(out_dir / ("edge_scores_%s.npz" % sid), **scores)

    print("\nResults saved to %s" % out_dir)


if __name__ == "__main__":
    main()
