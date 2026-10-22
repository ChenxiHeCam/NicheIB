'Aggregate per-section GraphMask conditioning and reliability results.'

import argparse
import json
from pathlib import Path

import numpy as np


LEARNABLE = {"layer_label", "excitatory_neuron", "oligodendrocyte"}


def off_diag_mean(matrix, indices):
    subset = matrix[np.ix_(indices, indices)]
    upper = np.triu_indices(len(indices), 1)
    return float(np.nanmean(subset[upper]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = sorted(Path(args.in_dir).glob("gm_cond_*.npz"))
    if not files:
        raise SystemExit(f"No gm_cond_*.npz under {args.in_dir}")

    rows = []
    target_names = None
    for path in files:
        data = np.load(path, allow_pickle=True)
        names = [str(value) for value in data["target_names"]]
        if target_names is None:
            target_names = names
        elif names != target_names:
            raise ValueError(f"Target order mismatch in {path}")
        learnable = [i for i, name in enumerate(names) if name in LEARNABLE]
        all_targets = list(range(len(names)))
        rows.append({
            "section": path.stem.replace("gm_cond_", ""),
            "real_all": off_diag_mean(data["real_matrix"], all_targets),
            "permuted_all": off_diag_mean(data["permuted_matrix"], all_targets),
            "real_learnable": off_diag_mean(data["real_matrix"], learnable),
            "permuted_learnable": off_diag_mean(
                data["permuted_matrix"], learnable
            ),
            "real_split_half": data["real_split_half"].tolist(),
            "permuted_split_half": data["permuted_split_half"].tolist(),
        })

    summary = {
        "n_sections": len(rows),
        "targets": target_names,
        "real_offdiag_all": float(np.mean([row["real_all"] for row in rows])),
        "permuted_offdiag_all": float(
            np.mean([row["permuted_all"] for row in rows])
        ),
        "real_offdiag_learnable": float(
            np.mean([row["real_learnable"] for row in rows])
        ),
        "permuted_offdiag_learnable": float(
            np.mean([row["permuted_learnable"] for row in rows])
        ),
        "real_split_half_by_target": {
            name: float(np.nanmean(
                [row["real_split_half"][i] for row in rows]
            ))
            for i, name in enumerate(target_names)
        },
        "permuted_split_half_by_target": {
            name: float(np.nanmean(
                [row["permuted_split_half"][i] for row in rows]
            ))
            for i, name in enumerate(target_names)
        },
        "real_split_half_n_finite_by_target": {
            name: int(np.isfinite(
                [row["real_split_half"][i] for row in rows]
            ).sum())
            for i, name in enumerate(target_names)
        },
        "permuted_split_half_n_finite_by_target": {
            name: int(np.isfinite(
                [row["permuted_split_half"][i] for row in rows]
            ).sum())
            for i, name in enumerate(target_names)
        },
        "per_section": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "per_section"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
