'E2/E3: Consensus stability, target specificity, and an external similarity reference.'

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = stats.spearmanr(a, b)
    return float(rho)


def split_half_reliability(maps: np.ndarray, n_splits: int = 50,
                           rng=None) -> tuple[float, float]:
    """Correlate consensus maps built from two disjoint halves of the seeds."""
    rng = rng or np.random.default_rng(0)
    n_seeds = maps.shape[0]
    half = n_seeds // 2
    rhos = []
    for _ in range(n_splits):
        order = rng.permutation(n_seeds)
        a = maps[order[:half]].mean(axis=0)
        b = maps[order[half:2 * half]].mean(axis=0)
        rhos.append(spearman(a, b))
    return float(np.mean(rhos)), float(np.std(rhos))


def pairwise_seed_agreement(maps: np.ndarray) -> float:
    """Mean Spearman over all pairs of individual seeds (the published metric)."""
    rhos = [spearman(maps[i], maps[j])
            for i, j in itertools.combinations(range(maps.shape[0]), 2)]
    return float(np.mean(rhos))


def residualise(maps: np.ndarray, default: np.ndarray) -> np.ndarray:
    """Remove the permuted-target default from each seed's map by rank regression."""
    d = stats.rankdata(default)
    d = (d - d.mean()) / (d.std() + 1e-12)
    out = np.empty_like(maps, dtype=np.float64)
    for i, m in enumerate(maps):
        r = stats.rankdata(m)
        r = (r - r.mean()) / (r.std() + 1e-12)
        out[i] = r - float(r @ d / len(d)) * d
    return out


def mantel(a: np.ndarray, b: np.ndarray, n_perm: int = 9999, rng=None):
    """Mantel test between two symmetric similarity matrices (upper triangle)."""
    rng = rng or np.random.default_rng(0)
    n = a.shape[0]
    iu = np.triu_indices(n, 1)
    obs = spearman(a[iu], b[iu])
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(spearman(a[np.ix_(p, p)][iu], b[iu])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictable-threshold", type=float, default=0.5,
                        help="R^2 (or macro F1) above which a target counts as learnable")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("stability_*.npz"))
    if not files:
        raise SystemExit(f"no stability_*.npz under {in_dir}")
    print(f"Loading {len(files)} sections", flush=True)

    perf = {}
    for rec_file in in_dir.glob("records_*.json"):
        for r in json.loads(rec_file.read_text()):
            perf.setdefault((r["section"], r["condition"], r["target"]), []).append(
                (r["value"], r["er"]))

    per_section, gate_mats, perm_mats, prog_mats = [], [], [], []
    target_order = None

    for path in files:
        z = np.load(path, allow_pickle=True)
        section = path.stem.replace("stability_", "")
        targets = [str(t) for t in z["target_names"]]
        target_order = target_order or targets

        # The permuted-target maps of all targets collapse onto one pattern; pool
        # them into a single default map per section.
        default = np.concatenate(
            [z[f"perm__{t}"] for t in targets if f"perm__{t}" in z], axis=0
        ).mean(axis=0)

        section_row = {"section": section, "targets": {}}
        consensus, resid_consensus = {}, {}

        for t in targets:
            maps = z[f"real__{t}"].astype(np.float64)
            resid = residualise(maps, default)
            cons = maps.mean(axis=0)
            consensus[t] = cons
            resid_consensus[t] = resid.mean(axis=0)

            raw_rho = pairwise_seed_agreement(maps)
            sh_mean, sh_std = split_half_reliability(maps)
            res_raw = pairwise_seed_agreement(resid)
            res_sh, _ = split_half_reliability(resid)

            vals = [v for v, _ in perf.get((section, "real", t), [])]
            ers = [e for _, e in perf.get((section, "real", t), [])]
            section_row["targets"][t] = {
                "performance": float(np.mean(vals)) if vals else None,
                "edge_rate": float(np.mean(ers)) if ers else None,
                "seed_agreement_raw": raw_rho,
                "consensus_reliability": sh_mean,
                "consensus_reliability_sd": sh_std,
                "seed_agreement_residual": res_raw,
                "consensus_reliability_residual": res_sh,
                # How far the consensus sits from the permuted-target default.
                # Low values mean the gate learned nothing target-specific.
                "distance_from_default": 1.0 - spearman(cons, default),
                "corr_with_default": spearman(cons, default),
            }

        # Cross-target matrices from the 10-seed consensus (what Figure 2a shows).
        n_t = len(targets)
        gate_mat = np.eye(n_t)
        for i, j in itertools.combinations(range(n_t), 2):
            gate_mat[i, j] = gate_mat[j, i] = spearman(
                consensus[targets[i]], consensus[targets[j]])
        section_row["gate_matrix"] = gate_mat.tolist()

        resid_mat = np.eye(n_t)
        for i, j in itertools.combinations(range(n_t), 2):
            resid_mat[i, j] = resid_mat[j, i] = spearman(
                resid_consensus[targets[i]], resid_consensus[targets[j]])
        section_row["gate_matrix_residual"] = resid_mat.tolist()

        # Similarity of the targets themselves, for the external reference below.
        scores = z["target_scores"].astype(np.float64)
        prog_mat = np.eye(n_t)
        for i, j in itertools.combinations(range(n_t), 2):
            if targets[i] == "layer_label" or targets[j] == "layer_label":
                prog_mat[i, j] = prog_mat[j, i] = np.nan
            else:
                prog_mat[i, j] = prog_mat[j, i] = spearman(scores[:, i], scores[:, j])
        section_row["program_similarity"] = prog_mat.tolist()

        # The same matrix built from the permuted-target runs (Figure 2a').
        perm_consensus = {t: z[f"perm__{t}"].mean(axis=0) for t in targets
                          if f"perm__{t}" in z}
        perm_mat = np.eye(n_t)
        for i, j in itertools.combinations(range(n_t), 2):
            if targets[i] in perm_consensus and targets[j] in perm_consensus:
                perm_mat[i, j] = perm_mat[j, i] = spearman(
                    perm_consensus[targets[i]], perm_consensus[targets[j]])
        section_row["permuted_matrix"] = perm_mat.tolist()

        gate_mats.append(gate_mat)
        perm_mats.append(perm_mat)
        prog_mats.append(prog_mat)
        per_section.append(section_row)
        print(f"  {section}: done", flush=True)

    # ---- aggregate stability -------------------------------------------------
    summary = {}
    for t in target_order:
        rows = [s["targets"][t] for s in per_section if t in s["targets"]]
        summary[t] = {
            k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            for k in ("performance", "edge_rate", "seed_agreement_raw",
                      "consensus_reliability", "seed_agreement_residual",
                      "consensus_reliability_residual", "distance_from_default",
                      "corr_with_default")
        }
        summary[t]["consensus_reliability_sd"] = float(np.mean(
            [r["consensus_reliability_sd"] for r in rows]))
        summary[t]["n_sections"] = len(rows)

    # Does target specificity track target predictability?
    perf_vec = np.array([summary[t]["performance"] for t in target_order])
    spec_vec = np.array([summary[t]["distance_from_default"] for t in target_order])
    res_vec = np.array([summary[t]["consensus_reliability_residual"] for t in target_order])
    raw_vec = np.array([summary[t]["seed_agreement_raw"] for t in target_order])
    specificity = {
        "targets": target_order,
        "performance": perf_vec.tolist(),
        "distance_from_default": spec_vec.tolist(),
        "residual_consensus_reliability": res_vec.tolist(),
        "raw_seed_agreement": raw_vec.tolist(),
        "spearman_perf_vs_specificity": spearman(perf_vec, spec_vec),
        "spearman_perf_vs_residual_reliability": spearman(perf_vec, res_vec),
        "spearman_perf_vs_raw_agreement": spearman(perf_vec, raw_vec),
    }

    # ---- external similarity reference (leave-one-section-out) ---------------
    prog_stack = np.array(prog_mats)
    gate_stack = np.array(gate_mats)
    prog_idx = [i for i, t in enumerate(target_order) if t != "layer_label"]
    learnable = [i for i in prog_idx
                 if summary[target_order[i]]["performance"] >= args.predictable_threshold]

    def mantel_over_sections(idx_subset, label):
        if len(idx_subset) < 4:
            return {"label": label, "n_targets": len(idx_subset), "skipped": True}
        rs, ps = [], []
        for s in range(len(gate_stack)):
            others = [k for k in range(len(prog_stack)) if k != s]
            ref = np.nanmean(prog_stack[others], axis=0)
            g = gate_stack[s][np.ix_(idx_subset, idx_subset)]
            e = np.abs(ref[np.ix_(idx_subset, idx_subset)])
            r, p = mantel(g, e, n_perm=1999,
                          rng=np.random.default_rng(100 + s))
            rs.append(r)
            ps.append(p)
        # Fisher's method over the per-section permutation p-values.
        chi2 = -2 * np.sum(np.log(np.clip(ps, 1e-12, 1)))
        combined_p = float(stats.chi2.sf(chi2, 2 * len(ps)))
        return {"label": label,
                "targets": [target_order[i] for i in idx_subset],
                "n_targets": len(idx_subset),
                "mantel_r_mean": float(np.mean(rs)),
                "mantel_r_sd": float(np.std(rs)),
                "n_sections_positive": int(np.sum(np.array(rs) > 0)),
                "n_sections": len(rs),
                "per_section_p": [float(p) for p in ps],
                "fisher_combined_p": combined_p}

    mantel_results = [
        mantel_over_sections(prog_idx, "all gene programs"),
        mantel_over_sections(learnable, "learnable gene programs only"),
    ]

    # Off-diagonal mean of the cross-target matrix, restricted to targets the model
    # can actually predict. Targets with R^2 ~ 0 inflate this quantity because their
    # gates fall back on the permuted default, so the all-target average understates
    # how target-specific the learnable gates really are.
    perm_stack = np.array(perm_mats)

    def off_diag_mean(mat, idx):
        sub = mat[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        return float(np.nanmean(sub[iu]))

    all_idx = list(range(len(target_order)))
    contrast = {}
    for label, idx in (("all targets", all_idx),
                       ("learnable targets", sorted(set(learnable) | {
                           i for i, t in enumerate(target_order)
                           if t == "layer_label"})),
                       ("unlearnable targets", [i for i in prog_idx
                                                if i not in learnable])):
        if len(idx) < 2:
            continue
        contrast[label] = {
            "n_targets": len(idx),
            "targets": [target_order[i] for i in idx],
            "real_off_diag_mean": float(np.mean(
                [off_diag_mean(m, idx) for m in gate_stack])),
            "permuted_off_diag_mean": float(np.mean(
                [off_diag_mean(m, idx) for m in perm_stack])),
            "real_off_diag_sd_across_pairs": float(np.mean([
                np.nanstd(m[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)])
                for m in gate_stack])),
        }

    out = {
        "n_sections": len(per_section),
        "summary": summary,
        "specificity_vs_predictability": specificity,
        "mantel": mantel_results,
        "real_vs_permuted_contrast": contrast,
        "mean_permuted_matrix": np.nanmean(perm_stack, axis=0).tolist(),
        "mean_gate_matrix": np.nanmean(gate_stack, axis=0).tolist(),
        "mean_gate_matrix_residual": np.nanmean(
            np.array([s["gate_matrix_residual"] for s in per_section]), axis=0).tolist(),
        "mean_program_similarity": np.nanmean(prog_stack, axis=0).tolist(),
        "target_order": target_order,
        "per_section": per_section,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # ---- report --------------------------------------------------------------
    print("\n" + "=" * 96)
    print("%-20s %8s %9s %9s %9s %9s %9s" % (
        "target", "perf", "seed-raw", "consensus", "resid-raw", "resid-cons", "dist-def"))
    print("=" * 96)
    for t in target_order:
        s = summary[t]
        print("%-20s %8.3f %9.3f %9.3f %9.3f %9.3f %9.3f" % (
            t, s["performance"], s["seed_agreement_raw"], s["consensus_reliability"],
            s["seed_agreement_residual"], s["consensus_reliability_residual"],
            s["distance_from_default"]))
    print("\nSpearman across targets:")
    print("  performance vs distance-from-default        %.3f"
          % specificity["spearman_perf_vs_specificity"])
    print("  performance vs residual consensus reliability %.3f"
          % specificity["spearman_perf_vs_residual_reliability"])
    print("  performance vs RAW seed agreement            %.3f"
          % specificity["spearman_perf_vs_raw_agreement"])
    print("\nCross-target correlation, real vs permuted targets:")
    print("%-24s %8s %12s %12s %10s" % (
        "target subset", "n", "real off-d", "permuted", "pair sd"))
    for label, c in contrast.items():
        print("%-24s %8d %12.3f %12.3f %10.3f" % (
            label, c["n_targets"], c["real_off_diag_mean"],
            c["permuted_off_diag_mean"], c["real_off_diag_sd_across_pairs"]))

    print("\nMantel (gate similarity vs held-out program similarity):")
    for m in mantel_results:
        if m.get("skipped"):
            print(f"  {m['label']}: skipped ({m['n_targets']} targets)")
        else:
            print("  %-32s r = %.3f +- %.3f, %d/%d sections positive, "
                  "Fisher p = %.2e"
                  % (m["label"], m["mantel_r_mean"], m["mantel_r_sd"],
                     m["n_sections_positive"], m["n_sections"],
                     m["fisher_combined_p"]))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
