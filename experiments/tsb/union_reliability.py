"Split-half consensus reliability over the union of ACTIVE edges (review's point)."
import glob, itertools, json, os
import numpy as np
from scipy.stats import spearmanr

R = "results/consensus_conditioning"
ACTIVE = 0.01
N_SPLITS = 20
rng = np.random.default_rng(0)

TARGETS = ["layer_label", "oligodendrocyte", "excitatory_neuron", "layer1_markers",
           "microglia", "astrocyte", "deep_layer_L5L6"]

def degree_score(edge_index, mask, n_nodes):
    """Edge score from endpoint degrees of the subgraph induced by `mask`."""
    deg = np.zeros(n_nodes)
    u, v = edge_index[0][mask], edge_index[1][mask]
    np.add.at(deg, u, 1.0)
    np.add.at(deg, v, 1.0)
    return deg[edge_index[0]] * deg[edge_index[1]]

out = {t: {"all": [], "union": [], "degree_null": [], "n_union": []} for t in TARGETS}
for path in sorted(glob.glob(os.path.join(R, "stability_*.npz"))):
    d = np.load(path, allow_pickle=True)
    ei = d["edge_index"]
    n_nodes = int(max(ei.max() + 1, d["coords"].shape[0]))
    for t in TARGETS:
        key = f"real__{t}"
        if key not in d:
            continue
        G = np.asarray(d[key], dtype=float)           # (n_seeds, n_edges)
        n_seeds = G.shape[0]
        if n_seeds < 4:
            continue
        half = n_seeds // 2
        for _ in range(N_SPLITS):
            idx = rng.permutation(n_seeds)
            A = G[idx[:half]].mean(0)
            B = G[idx[half:2 * half]].mean(0)
            r_all = spearmanr(A, B).statistic
            union = (A > ACTIVE) | (B > ACTIVE)
            if union.sum() < 50:
                continue
            r_union = spearmanr(A[union], B[union]).statistic
            dscore = degree_score(ei, A > ACTIVE, n_nodes)
            r_deg = spearmanr(dscore[union], B[union]).statistic
            for k, v in (("all", r_all), ("union", r_union), ("degree_null", r_deg)):
                if np.isfinite(v):
                    out[t][k].append(float(v))
            out[t]["n_union"].append(float(union.mean()))

print(f"{'target':20s} {'all-edge':>9s} {'union':>9s} {'degree-only':>12s} {'active frac':>12s}")
summary = {}
for t in TARGETS:
    if not out[t]["union"]:
        continue
    m = {k: float(np.mean(v)) for k, v in out[t].items() if v}
    summary[t] = m
    print(f"{t:20s} {m['all']:9.3f} {m['union']:9.3f} {m['degree_null']:12.3f} {m['n_union']:12.3f}")

with open(os.path.join(R, "union_reliability.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved union_reliability.json")
