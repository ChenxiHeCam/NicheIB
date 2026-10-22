'E5/E6: Making the external biological validation independent of the target.'

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CURATED_LR_PAIRS = {
    "oligodendrocyte": [("MAG", "RTN4R"), ("MBP", "GALC"),
                        ("PLP1", "GPR37"), ("MOG", "RTN4R")],
    "astrocyte": [("GFAP", "CD44"), ("S100B", "AGER"),
                  ("AQP4", "DAG1"), ("GJA1", "GJA1")],
    "microglia": [("CX3CL1", "CX3CR1"), ("C1QA", "C1QBP"),
                  ("TREM2", "TYROBP"), ("CD68", "MFGE8")],
    "excitatory_neuron": [("GRIN1", "GRIN2A"), ("SLC17A7", "HOMER1"),
                          ("BDNF", "NTRK2"), ("NRXN1", "NLGN1")],
}

# Marker panels used for the non-circular annotation-gate check.
MARKER_PANELS = {
    "oligodendrocyte": ["MBP", "PLP1", "MOG", "MAG"],
    "astrocyte": ["GFAP", "AQP4", "SLC1A2", "SLC1A3"],
    "excitatory_neuron": ["SLC17A7", "SATB2", "CUX2", "RORB", "FEZF2"],
}


def morans_i_weighted(expr, edge_index, weights, n_nodes):
    """Moran's I with per-edge weights — identical definition to the submission."""
    x = expr[:n_nodes].astype(np.float64)
    dev = x - x.mean()
    w_sum = weights.sum()
    denom = np.sum(dev ** 2)
    if w_sum == 0 or denom == 0:
        return float("nan")
    num = np.sum(weights * dev[edge_index[0]] * dev[edge_index[1]])
    return float((n_nodes / w_sum) * (num / denom))


def top_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    k = int(round(frac * len(scores)))
    mask = np.zeros(len(scores), dtype=np.float64)
    mask[np.argsort(scores)[-k:]] = 1.0
    return mask


def density_matched_mask(target_mask: np.ndarray, density: np.ndarray,
                         rng, n_bins: int = 10) -> np.ndarray:
    """Random edge set with the same per-decile counts as `target_mask`.

    Removes the confound that high-gate edges sit in regions where the marker
    programme is simply more highly expressed.
    """
    edges = np.arange(len(density))
    # Rank-based bins keep the strata balanced even when density is skewed.
    ranks = np.argsort(np.argsort(density))
    bins = np.minimum((ranks * n_bins) // len(density), n_bins - 1)
    out = np.zeros(len(density), dtype=np.float64)
    for b in range(n_bins):
        in_bin = edges[bins == b]
        need = int(target_mask[in_bin].sum())
        if need == 0 or len(in_bin) == 0:
            continue
        out[rng.choice(in_bin, size=min(need, len(in_bin)), replace=False)] = 1.0
    return out


def lr_edge_score(expr_lookup, edge_index, pairs):
    src, dst = edge_index[0], edge_index[1]
    scores = []
    for ligand, receptor in pairs:
        if ligand not in expr_lookup or receptor not in expr_lookup:
            continue
        le, re = expr_lookup[ligand], expr_lookup[receptor]
        scores.append(le[src] * re[dst] + le[dst] * re[src])
    return np.mean(scores, axis=0) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-frac", type=float, default=0.2)
    parser.add_argument("--n-random", type=int, default=50)
    args = parser.parse_args()

    files = sorted(Path(args.in_dir).glob("stability_*.npz"))
    if not files:
        raise SystemExit(f"no stability_*.npz under {args.in_dir}")

    heldout = defaultdict(list)          # (gate, eval program) -> enhancement ratios
    annotation = defaultdict(list)       # marker panel -> enhancement ratios
    lr_rows = []

    for path in files:
        z = np.load(path, allow_pickle=True)
        section = path.stem.replace("stability_", "")
        edge_index = z["edge_index"].astype(np.int64)
        n_nodes = z["coords"].shape[0]
        n_edges = edge_index.shape[1]
        uniform = np.ones(n_edges)
        rng = np.random.default_rng(abs(hash(section)) % (2 ** 31))
        expr = {k[len("expr__"):]: z[k] for k in z.files if k.startswith("expr__")}

        heldout_gates = {k[len("heldout__"):]: z[k].mean(axis=0)
                         for k in z.files if k.startswith("heldout__")}
        eval_genes = {k[len("heldout_eval_genes__"):]: [str(g) for g in z[k]]
                      for k in z.files if k.startswith("heldout_eval_genes__")}

        # ---- E5a: held-out markers, full gate x programme selectivity matrix ----
        # Reviewer review: the random-edge control does not test the abundance confound
        # that took the L-R number for the same cell type from 4.66x to 1.00x. We add a
        # density-matched control here too, matched on the evaluated programme's own
        # local expression so it is not tautological with the gate's target.
        for gate_name, gate in heldout_gates.items():
            mask = top_mask(gate, args.top_frac)
            rand_masks = [top_mask(rng.random(n_edges), args.top_frac)
                          for _ in range(args.n_random)]
            for eval_prog, genes in eval_genes.items():
                ratios, rand_ratios = [], []
                for g in genes:
                    if g not in expr or expr[g].std() == 0:
                        continue
                    i_full = morans_i_weighted(expr[g], edge_index, uniform, n_nodes)
                    if not np.isfinite(i_full) or abs(i_full) < 0.02:
                        continue  # unstable ratio when the baseline is ~0
                    ratios.append(morans_i_weighted(expr[g], edge_index, mask, n_nodes)
                                  / i_full)
                    rand_ratios.append(np.mean([
                        morans_i_weighted(expr[g], edge_index, rm, n_nodes) / i_full
                        for rm in rand_masks[:10]]))
                # Density-matched control: same decile distribution of local
                # expression of the genes being evaluated.
                dens_ratios = []
                present = [g for g in genes if g in expr]
                if present and ratios:
                    node_d = np.mean([expr[g] for g in present], axis=0)
                    edge_d = 0.5 * (node_d[edge_index[0]] + node_d[edge_index[1]])
                    dmasks = [density_matched_mask(mask, edge_d, rng)
                              for _ in range(10)]
                    for g in genes:
                        if g not in expr or expr[g].std() == 0:
                            continue
                        i_full = morans_i_weighted(expr[g], edge_index, uniform, n_nodes)
                        if not np.isfinite(i_full) or abs(i_full) < 0.02:
                            continue
                        dens_ratios.append(np.mean([
                            morans_i_weighted(expr[g], edge_index, dm, n_nodes) / i_full
                            for dm in dmasks]))
                if ratios:
                    heldout[(gate_name, eval_prog)].append(
                        (float(np.mean(ratios)), float(np.mean(rand_ratios)),
                         len(ratios),
                         float(np.mean(dens_ratios)) if dens_ratios else float("nan")))

        # ---- E5b: gate trained on the manual annotation only -------------------
        if "real__layer_label" in z.files:
            gate = z["real__layer_label"].mean(axis=0)
            mask = top_mask(gate, args.top_frac)
            rand_masks = [top_mask(rng.random(n_edges), args.top_frac)
                          for _ in range(10)]
            for panel, genes in MARKER_PANELS.items():
                ratios, rand_ratios = [], []
                for g in genes:
                    if g not in expr or expr[g].std() == 0:
                        continue
                    i_full = morans_i_weighted(expr[g], edge_index, uniform, n_nodes)
                    if not np.isfinite(i_full) or abs(i_full) < 0.02:
                        continue
                    ratios.append(morans_i_weighted(expr[g], edge_index, mask, n_nodes)
                                  / i_full)
                    rand_ratios.append(np.mean([
                        morans_i_weighted(expr[g], edge_index, rm, n_nodes) / i_full
                        for rm in rand_masks]))
                if ratios:
                    annotation[panel].append(
                        (float(np.mean(ratios)), float(np.mean(rand_ratios)), len(ratios)))

        # ---- E6: ligand-receptor enrichment with a density-matched control -----
        for cell_type, pairs in CURATED_LR_PAIRS.items():
            key = f"real__{cell_type}"
            if key not in z.files:
                continue
            lr = lr_edge_score(expr, edge_index, pairs)
            if lr is None:
                continue
            gate = z[key].mean(axis=0)
            markers = MARKER_PANELS.get(cell_type, [p[0] for p in pairs])
            present = [m for m in markers if m in expr]
            if not present:
                continue
            # The confound the paper names: local abundance of this cell type.
            node_density = np.mean([expr[m] for m in present], axis=0)
            edge_density = 0.5 * (node_density[edge_index[0]] +
                                  node_density[edge_index[1]])

            # Matching on the marker genes is near-tautological when those genes are
            # also the ligands, so we additionally match on a confound proxy that is
            # independent of the pathway: total expression summed over all stored
            # validation genes, which stands in for local library size / cellularity.
            all_expr = np.mean([v for v in expr.values()], axis=0)
            edge_library = 0.5 * (all_expr[edge_index[0]] + all_expr[edge_index[1]])

            top = top_mask(gate, args.top_frac)
            bottom = top_mask(-gate, args.top_frac)
            uniform_ctrl = [top_mask(rng.random(n_edges), args.top_frac)
                            for _ in range(args.n_random)]
            matched_ctrl = [density_matched_mask(top, edge_density, rng)
                            for _ in range(args.n_random)]
            library_ctrl = [density_matched_mask(top, edge_library, rng)
                            for _ in range(args.n_random)]

            def mean_lr(m):
                return float(lr[m > 0].mean())

            lr_rows.append({
                "section": section, "cell_type": cell_type,
                "lr_top": mean_lr(top),
                "lr_bottom": mean_lr(bottom),
                "lr_uniform_random": float(np.mean([mean_lr(m) for m in uniform_ctrl])),
                "lr_density_matched": float(np.mean([mean_lr(m) for m in matched_ctrl])),
                "lr_library_matched": float(np.mean([mean_lr(m) for m in library_ctrl])),
                "density_top": float(edge_density[top > 0].mean()),
                "density_matched": float(np.mean(
                    [edge_density[m > 0].mean() for m in matched_ctrl])),
                "density_uniform": float(np.mean(
                    [edge_density[m > 0].mean() for m in uniform_ctrl])),
            })
        print(f"  {section}: done", flush=True)

    def collapse(d):
        out = {}
        for k, vals in d.items():
            key = "|".join(k) if isinstance(k, tuple) else k
            entry = {"tsb_enhancement": float(np.mean([v[0] for v in vals])),
                     "random_enhancement": float(np.mean([v[1] for v in vals])),
                     "n_sections": len(vals),
                     "n_genes_mean": float(np.mean([v[2] for v in vals]))}
            dens = [v[3] for v in vals if len(v) > 3 and v[3] == v[3]]
            if dens:
                entry["density_matched_enhancement"] = float(np.mean(dens))
            out[key] = entry
        return out

    lr_summary = {}
    for ct in CURATED_LR_PAIRS:
        rows = [r for r in lr_rows if r["cell_type"] == ct]
        if not rows:
            continue
        lr_summary[ct] = {
            "n_sections": len(rows),
            "vs_bottom20": float(np.mean([r["lr_top"] / max(r["lr_bottom"], 1e-10)
                                          for r in rows])),
            "vs_uniform_random": float(np.mean(
                [r["lr_top"] / max(r["lr_uniform_random"], 1e-10) for r in rows])),
            "vs_density_matched": float(np.mean(
                [r["lr_top"] / max(r["lr_density_matched"], 1e-10) for r in rows])),
            "vs_library_matched": float(np.mean(
                [r["lr_top"] / max(r.get("lr_library_matched") or np.nan, 1e-10)
                 for r in rows])),
            "density_ratio_top_vs_uniform": float(np.mean(
                [r["density_top"] / max(r["density_uniform"], 1e-10) for r in rows])),
        }

    out = {"heldout_marker_morans": collapse(heldout),
           "annotation_gate_morans": collapse(annotation),
           "lr_enrichment": lr_summary,
           "lr_per_section": lr_rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n=== Held-out marker Moran's I enhancement (gate -> evaluated programme) ===")
    print("%-42s %10s %10s %14s" % ("gate | evaluated on", "TSB", "random",
                                    "density-match"))
    for k, v in sorted(out["heldout_marker_morans"].items()):
        dm = v.get("density_matched_enhancement")
        print("%-42s %10.2f %10.2f %14s" % (
            k, v["tsb_enhancement"], v["random_enhancement"],
            ("%.2f" % dm) if dm is not None else "-"))

    print("\n=== Annotation-trained gate (no expression in the target) ===")
    print("%-24s %10s %10s" % ("marker panel", "TSB", "random"))
    for k, v in sorted(out["annotation_gate_morans"].items()):
        print("%-24s %10.2f %10.2f" % (k, v["tsb_enhancement"], v["random_enhancement"]))

    print("\n=== L-R enrichment against progressively stricter controls ===")
    print("%-20s %12s %12s %14s %14s %10s" % (
        "cell type", "vs bottom20", "vs random", "vs marker-dens", "vs library",
        "density x"))
    for k, v in lr_summary.items():
        print("%-20s %12.2f %12.2f %14.2f %14.2f %10.2f" % (
            k, v["vs_bottom20"], v["vs_uniform_random"], v["vs_density_matched"],
            v["vs_library_matched"], v["density_ratio_top_vs_uniform"]))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
