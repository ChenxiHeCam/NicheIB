# NicheIB

Code and small result records for **NicheIB: Targeted Spatial Bottlenecks for Predictive Microenvironment Discovery**, Chenxi He, NeurIPS 2026 (poster).

[Paper and discussion](https://openreview.net/forum?id=LaZNrmtjRL)

## What this release supports

NicheIB learns target-conditioned spatial edge scores using Hard Concrete gates. The evaluation distinguishes the gated predictor's own accuracy from the utility of exported edges for a fresh GCN. It also examines target conditioning relative to permuted targets and the trade-off between node coverage and specificity.

The evidence does **not** establish globally minimal biological niches, causal cell interactions, universal predictive superiority, or validated clinical use. Matched biological controls (run on TSB-fixed) support target-specific structure for one of four gene programs. See `docs/RESULTS_AND_LIMITATIONS.md`.

## Installation

Use a separate environment (Python >=3.10; Python 3.11 is a reasonable starting point). Install the appropriate PyTorch build for your hardware first if necessary, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

The dependency bounds in `pyproject.toml` are not a frozen training environment. Numerical results may vary across PyTorch, PyG, NumPy, Scanpy, BLAS, and hardware versions. The included JSON records are the archived evaluation outputs; local smoke tests do not rerun the full biological study.

## Reproduce the published summaries without training

```bash
python -m pip install numpy scipy matplotlib
python camera_ready/build_assets.py
```

This checks run identities, recomputes the matched-mask refit tables, Holm-adjusted comparisons, and the lambda sweep, regenerates all paper figures, and writes `camera_ready/generated/verification.json` with checksums of every evidence file. It needs no raw expression matrices and performs no network calls. Files under `camera_ready/evidence` are small result records (JSON) and seed-averaged edge scores for one DLPFC section (`edge_maps/consensus_151673.npz`), not raw patient data or model checkpoints. `camera_ready/derive_small_evidence.py` documents how the last two were derived from larger run outputs.

## Run the main experiments

Run commands from the repository root. Obtain the original datasets first; they are not bundled. Details are in `docs/DATA.md` and `docs/REPRODUCING.md`.

```bash
# DLPFC matched-mask refit; every selector exports the same edge count.
python experiments/tsb/dlpfc_matched_mask_refit.py \
  --data-dir data/dlpfc --sections 151673 --seeds 0 1 2 \
  --budgets 0.1 0.2 0.3 --epochs 300 --out-dir results/unified

# Consensus conditioning; use --variant cov for TSB-cov and --variant dual for TSB-dual.
python experiments/tsb/consensus_conditioning.py \
  --data-dir data/dlpfc --sections 151673 --variant cov \
  --n-seeds-real 10 --n-seeds-perm 5 --n-seeds-heldout 0 \
  --epochs 300 --out-dir results/consensus_cov

# Mouse-brain matched-mask refit.
python experiments/tsb/mouse_matched_mask_refit.py \
  --data-dir data/squidpy --out results/mouse/records.json \
  --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 300 --budget 0.2
```

The paper uses two operating modes: TSB-dual (augmented-Lagrangian budget, `--variant dual`) for contrasting targets and TSB-cov (budget plus coverage penalty, `--variant cov`) for exporting subgraphs; TSB-fixed (`--variant fixed`) is the fixed-penalty ablation. Archived records label TSB-cov as `final` and TSB-fixed as `submitted`; both old names remain accepted. The refit scripts train TSB-fixed and TSB-cov, and `experiments/tsb/dual_refit.py` adds TSB-dual under the same protocol. `train_tsb.py` and the other `run_*.py` runners use the fixed-penalty objective and reproduce supplementary analyses only.

## Layout

```text
nicheib/                 Model, gates, baselines, loaders, metrics
experiments/tsb/         Final entry points and supporting/historical analyses
tests/                  Core tests and a tiny TSB-cov training smoke test
scripts/                Data acquisition/conversion helpers
camera_ready/evidence/  Archived small numerical summaries
camera_ready/generated/ Regenerated tables, figures, verification output
docs/                   Data, protocols, interpretation, release notes
```

No raw datasets, private account files, cluster job logs, Overleaf credentials, reviewer text, or training checkpoints are included. Compute-intensive training belongs on an appropriately provisioned workstation or scheduler, not an HPC login node. Start with one section and profile resources before scaling.

## Citation

```bibtex
@inproceedings{he2026nicheib,
  title={NicheIB: Targeted Spatial Bottlenecks for Predictive Microenvironment Discovery},
  author={He, Chenxi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=LaZNrmtjRL}
}
```

## License and contact

Original code is MIT licensed, copyright 2026 Chenxi He. Third-party dependencies and source datasets retain their own licenses and terms. This code license does not grant rights to redistribute source tissue datasets. Contact: ch2067@cam.ac.uk.
