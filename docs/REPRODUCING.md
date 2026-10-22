# Reproduction protocols

Run commands from the repository root. The README gives the three main entry points. Start with one section before scheduling a full run, and use explicit input/output directories.

## Regenerating the paper's tables and figures

`python camera_ready/build_assets.py` reads the archived records in `camera_ready/evidence`, regenerates every table and figure in the paper, and writes SHA-256 checksums of the evidence files to `camera_ready/generated/verification.json`. Paths inside archived metadata were normalized for privacy; numerical values are unchanged, so the checksums identify the released files rather than the original working copies.

## Variants

- **TSB-fixed** (`--variant fixed`): squared budget penalty with coefficient lambda_0 (5 unless swept); an ablation.
- **TSB-dual** (`--variant dual`): augmented-Lagrangian budget with projected dual ascent, no coverage term; the mode for contrasting targets.
- **TSB-cov** (`--variant cov`): augmented-Lagrangian budget plus the expected-degree coverage penalty; the mode for exporting subgraphs.

`dlpfc_matched_mask_refit.py` trains TSB-fixed and TSB-cov and exports exactly matched edge counts for every selector to fresh GCNs. `consensus_conditioning.py` defaults to TSB-fixed; pass `--variant dual` or `--variant cov` explicitly. Archived records use the older labels `submitted` (TSB-fixed) and `final` (TSB-cov). `dual_refit.py` adds TSB-dual to the DLPFC and mouse refit protocols, reusing the archived k, splits and seeds. Records from different variants should not be pooled.

## What the refit protocol measures

The matched-mask refit retrains a fresh GCN on a selected edge set within the same tissue section. It measures whether the edges are useful to another model; it is neither the gating model's own accuracy nor transfer to unseen donors. Seed repetitions measure algorithmic variation; the four serial sections per DLPFC donor are not independent biological replicates.

GraphMask is an in-house adaptation and explains different frozen models in the refit protocol (TSB-cov) and the conditioning protocol (a target-specific GAT).

## Testing

`python -m pytest tests -q` exercises core components and a tiny TSB-cov training run. It does not establish budget convergence or reproduce a full experiment. Dependency bounds are not a frozen training environment.

## Manuscript source

The separate source package compiles with `latexmk -pdf main.tex`; generated tables and figures are included, so no training is required.
