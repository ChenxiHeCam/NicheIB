'Rescue attempt for the ligand-receptor claim: a proper spatial null.'

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from heldout_marker_audit import (  # noqa: E402  (same directory)
    CURATED_LR_PAIRS,
    MARKER_PANELS,
    density_matched_mask,
    lr_edge_score,
    top_mask,
)


def stratified_permutation(values: np.ndarray, rng, n_strata: int = 20) -> np.ndarray:
    """Permute spot indices within expression strata.

    Keeps each spot's expression level in the same stratum as before, so the null
    preserves the marginal expression distribution and only destroys which spots are
    spatial neighbours of which.
    """
    n = len(values)
    order = np.argsort(values)
    perm = np.arange(n)
    bounds = np.linspace(0, n, n_strata + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        block = order[a:b]
        perm[block] = block[rng.permutation(len(block))]
    return perm


def communicating_calls(expr, edge_index, pairs, rng, n_perm=200, alpha=0.05):
    """Binary per-edge call: is this edge's L-R product above its spatial null?"""
    observed = lr_edge_score(expr, edge_index, pairs)
    if observed is None:
        return None, None
    n_nodes = max(int(edge_index.max()) + 1, 1)
    exceed = np.zeros(len(observed), dtype=np.int32)

    # Stratify on the mean expression of the pathway's genes so the permutation is
    # expression-preserving rather than uniform.
    genes = sorted({g for p in pairs for g in p if g in expr})
    if not genes:
        return None, None
    strat_on = np.mean([expr[g] for g in genes], axis=0)[:n_nodes]

    for _ in range(n_perm):
        perm = stratified_permutation(strat_on, rng)
        permuted_expr = {g: v[perm] for g, v in expr.items() if g in genes}
        null = lr_edge_score(permuted_expr, edge_index, pairs)
        exceed += (observed > null).astype(np.int32)

    p_values = 1.0 - exceed / n_perm
    return observed, (p_values <= alpha).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-frac", type=float, default=0.2)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--n-control", type=int, default=30)
    args = parser.parse_args()

    files = sorted(Path(args.in_dir).glob("stability_*.npz"))
    if not files:
        raise SystemExit(f"no stability_*.npz under {args.in_dir}")

    rows = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        section = path.stem.replace("stability_", "")
        edge_index = z["edge_index"].astype(np.int64)
        n_edges = edge_index.shape[1]
        expr = {k[len("expr__"):]: z[k] for k in z.files if k.startswith("expr__")}
        rng = np.random.default_rng(abs(hash(section)) % (2 ** 31))

        for cell_type, pairs in CURATED_LR_PAIRS.items():
            key = f"real__{cell_type}"
            if key not in z.files:
                continue
            observed, calls = communicating_calls(
                expr, edge_index, pairs, rng, n_perm=args.n_perm)
            if calls is None:
                continue

            gate = z[key].mean(axis=0)
            markers = [m for m in MARKER_PANELS.get(cell_type, [p[0] for p in pairs])
                       if m in expr]
            if not markers:
                continue
            node_density = np.mean([expr[m] for m in markers], axis=0)
            edge_density = 0.5 * (node_density[edge_index[0]] +
                                  node_density[edge_index[1]])

            top = top_mask(gate, args.top_frac)
            uniform = [top_mask(rng.random(n_edges), args.top_frac)
                       for _ in range(args.n_control)]
            matched = [density_matched_mask(top, edge_density, rng)
                       for _ in range(args.n_control)]

            frac = lambda m: float(calls[m > 0].mean())
            rows.append({
                "section": section, "cell_type": cell_type,
                "frac_communicating_all": float(calls.mean()),
                "frac_communicating_top": frac(top),
                "frac_communicating_uniform": float(np.mean([frac(m) for m in uniform])),
                "frac_communicating_matched": float(np.mean([frac(m) for m in matched])),
            })
            print(f"  {section} {cell_type}: top {rows[-1]['frac_communicating_top']:.3f} "
                  f"| uniform {rows[-1]['frac_communicating_uniform']:.3f} "
                  f"| density-matched {rows[-1]['frac_communicating_matched']:.3f} "
                  f"| overall {rows[-1]['frac_communicating_all']:.3f}", flush=True)

    summary = {}
    for ct in CURATED_LR_PAIRS:
        sel = [r for r in rows if r["cell_type"] == ct]
        if not sel:
            continue
        summary[ct] = {
            "n_sections": len(sel),
            "top": float(np.mean([r["frac_communicating_top"] for r in sel])),
            "uniform": float(np.mean([r["frac_communicating_uniform"] for r in sel])),
            "matched": float(np.mean([r["frac_communicating_matched"] for r in sel])),
            "enrichment_vs_uniform": float(np.mean(
                [r["frac_communicating_top"] / max(r["frac_communicating_uniform"], 1e-9)
                 for r in sel])),
            "enrichment_vs_matched": float(np.mean(
                [r["frac_communicating_top"] / max(r["frac_communicating_matched"], 1e-9)
                 for r in sel])),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"per_section": rows, "summary": summary}, indent=2))

    print("\n=== Fraction of edges called communicating against a spatial null ===")
    print("%-20s %10s %10s %12s %14s %16s" % (
        "cell type", "top 20%", "uniform", "matched", "vs uniform", "vs matched"))
    for ct, v in summary.items():
        print("%-20s %10.3f %10.3f %12.3f %14.2f %16.2f" % (
            ct, v["top"], v["uniform"], v["matched"],
            v["enrichment_vs_uniform"], v["enrichment_vs_matched"]))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
