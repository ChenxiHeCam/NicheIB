"""Derive small archived evidence files from large per-run outputs.

Usage:
    python derive_small_evidence.py --checkpoint checkpoint_151673.npz \
        --dual-checkpoint checkpoint_dual_151673.npz --lambda-dir lambda_sweep_runs

Checkpoints are written by the consensus-conditioning script for the TSB-cov
(--variant final) and TSB-dual variants;
the lambda directory holds one records.jsonl per DLPFC section from the
fixed-penalty lambda_L0 sweep. Outputs go to evidence/ and are read by
build_assets.py. No values are recomputed except seed averages of edge scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TARGETS = ["layer_label", "excitatory_neuron", "oligodendrocyte", "astrocyte",
           "microglia", "layer1_markers", "deep_layer_L5L6"]


def consensus(z, suffix):
    """Seed-averaged real maps and the permuted default for one variant."""
    maps = {f"real__{t}{suffix}": z[f"real__{t}"].mean(axis=0).astype(np.float32) for t in TARGETS}
    maps[f"permuted_default{suffix}"] = np.concatenate(
        [z[f"perm__{t}"] for t in TARGETS]).mean(axis=0).astype(np.float32)
    return maps


def edge_maps(checkpoint, dual_checkpoint):
    z = np.load(checkpoint, allow_pickle=False)
    d = np.load(dual_checkpoint, allow_pickle=False)
    assert (z["edge_index"] == d["edge_index"]).all() and (z["coords"] == d["coords"]).all()
    names = list(z["target_names"])
    out = ROOT / "evidence/edge_maps"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "consensus_151673.npz",
        coords=z["coords"], layer=z["y"], edge_index=z["edge_index"],
        oligodendrocyte_score=z["target_scores"][:, names.index("oligodendrocyte")],
        n_real_seeds=np.int32(z["real__layer_label"].shape[0]),
        n_permuted_maps=np.int32(sum(z[f"perm__{t}"].shape[0] for t in TARGETS)),
        **consensus(z, ""), **consensus(d, "_dual"))


def gate_magnitudes(checkpoint, dual_checkpoint):
    """Per-seed deterministic gate values of the top 20% edges, for each target."""
    summary = {}
    for name, path in (("cov", checkpoint), ("dual", dual_checkpoint)):
        z = np.load(path, allow_pickle=False)
        summary[name] = {}
        for t in TARGETS:
            maps = z[f"real__{t}"]
            k = int(round(.2 * maps.shape[1]))
            top = np.sort(maps, axis=1)[:, -k:]
            summary[name][t] = dict(mean_top20=float(top.mean()),
                                    frac_zero=float((maps <= 1e-6).mean()),
                                    n_seeds=int(maps.shape[0]))
    out = ROOT / "evidence/edge_maps/gate_magnitudes_151673.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")


def lambda_sweep(directory):
    records = [json.loads(line) for path in sorted(Path(directory).glob("*/records.jsonl"))
               for line in path.read_text().splitlines() if line.strip()]
    assert len({r["sample_id"] for r in records}) == 12
    out = ROOT / "evidence/lambda_sweep_dlpfc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(records) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dual-checkpoint", required=True)
    parser.add_argument("--lambda-dir", required=True)
    args = parser.parse_args()
    edge_maps(args.checkpoint, args.dual_checkpoint)
    gate_magnitudes(args.checkpoint, args.dual_checkpoint)
    lambda_sweep(args.lambda_dir)
