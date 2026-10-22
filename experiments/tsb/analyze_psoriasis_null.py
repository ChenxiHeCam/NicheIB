'Permutation p-values for the n=23 psoriasis cohort (E11b).'

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["macro_f1", "balanced_accuracy", "accuracy"]


def _load(path: Path) -> dict | None:
    f = path / "psoriasis_results.json"
    if not f.is_file():
        return None
    try:
        with f.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def _fold_labels(perm: dict) -> dict[str, list[int]]:
    """Test-set labels per LOPO fold, under the labels that run actually used."""
    patients = perm.get("patients")
    labels = perm.get("permuted_labels") or perm.get("original_labels")
    if not patients or not labels:
        return {}
    out = defaultdict(list)
    for p, y in zip(patients, labels):
        out[f"lopo_{p}"].append(int(y))
    return dict(out)


def _statistics(run: dict, fold_labels: dict[str, list[int]]) -> dict[str, float]:
    """Mean of each metric over (seed x fold), per method.

    Two fold sets: every fold, and only folds whose held-out patient contributes
    both classes. The second matters because one patient (PsO_6) has two lesional
    sections and no non-lesional one, so balanced accuracy on that fold is not a
    two-class quantity.
    """
    both = {k for k, v in fold_labels.items() if len(set(v)) > 1}
    acc: dict[str, list[float]] = defaultdict(list)
    for r in run.get("records", []):
        method = r.get("method")
        split = r.get("split_id")
        for m in METRICS:
            v = r.get(m)
            if v is None:
                continue
            acc[f"{method}|{m}|all"].append(float(v))
            if not both or split in both:
                acc[f"{method}|{m}|two_class_folds"].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    real_run = _load(args.base / "real")
    if real_run is None:
        raise SystemExit(f"no real run under {args.base}/real")

    perms: dict[str, list[dict]] = {"within_patient": [], "global": []}
    label_template: dict = {}
    for d in sorted(args.base.iterdir()):
        if not d.is_dir() or d.name == "real":
            continue
        run = _load(d)
        if run is None:
            print(f"  skip (no results): {d.name}")
            continue
        scope = (run.get("permutation") or {}).get("scope")
        if scope not in perms:
            print(f"  skip (unknown scope {scope}): {d.name}")
            continue
        perms[scope].append(run)
        if not label_template:
            label_template = run["permutation"]

    # The real run has no permutation block, so borrow the patient vector and the
    # original labels from any permutation run — they describe the same cohort.
    real_folds = _fold_labels({"patients": label_template.get("patients"),
                               "permuted_labels": label_template.get("original_labels")})
    real_stats = _statistics(real_run, real_folds)

    report: dict = {"n_permutations": {k: len(v) for k, v in perms.items()},
                    "observed": real_stats, "tests": {}}

    for scope, runs in perms.items():
        if not runs:
            continue
        # How many of the drawn permutations are distinct label vectors: with
        # Monte-Carlo sampling duplicates are legitimate, but the reader should
        # know how much of the permutation space was actually visited.
        seen = {tuple(r["permutation"]["permuted_labels"]) for r in runs}
        null_stats = [_statistics(r, _fold_labels(r["permutation"])) for r in runs]

        for key, obs in real_stats.items():
            vals = np.asarray([s[key] for s in null_stats if key in s], dtype=float)
            if vals.size == 0:
                continue
            n_ge = int((vals >= obs - 1e-12).sum())
            report["tests"].setdefault(key, {})[scope] = {
                "observed": obs,
                "null_mean": float(vals.mean()),
                "null_sd": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
                "null_q95": float(np.quantile(vals, 0.95)),
                "null_max": float(vals.max()),
                "n_perm": int(vals.size),
                "n_distinct_permutations": len(seen),
                "n_ge_observed": n_ge,
                "p_value": (1.0 + n_ge) / (1.0 + vals.size),
            }

    print(f"\nreal run: {args.base/'real'}")
    print(f"permutations: {report['n_permutations']}\n")
    header = f"{'statistic':44s} {'obs':>7s} {'null mean±sd':>16s} {'q95':>7s} {'p':>8s}  B"
    for key in sorted(report["tests"]):
        print(f"\n--- {key} ---")
        print(header)
        for scope in ["within_patient", "global"]:
            t = report["tests"][key].get(scope)
            if t is None:
                continue
            print(f"{scope:44s} {t['observed']:7.4f} "
                  f"{t['null_mean']:7.4f}±{t['null_sd']:<8.4f} {t['null_q95']:7.4f} "
                  f"{t['p_value']:8.4f}  {t['n_perm']} ({t['n_distinct_permutations']} distinct)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
