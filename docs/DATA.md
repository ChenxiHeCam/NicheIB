# Data preparation

Raw expression matrices, patient-level records, model checkpoints, and private storage locations are not included. Obtain each dataset from its original provider under that provider's terms. The manuscript bibliography identifies the original studies. No dataset redistribution rights are granted by the code license.

## DLPFC

The loader expects one AnnData file per section in a directory supplied with `--data-dir`, named `<section>.h5ad`. The study uses sections 151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, and 151676. These are 12 sections from three donors, not 12 independent donors.

Files must contain the expression matrix, spatial coordinates in `obsm['spatial']`, and cortical-layer annotations in an accepted observation column (`ground_truth`, `sce.layer_guess`, or `layer_guess`). Inspect `nicheib/data.py` for exact filtering and preprocessing. Place prepared files under `data/dlpfc/`, or pass another directory explicitly. The package does not provide an automated, validated reconstruction of these annotated files from the provider's raw downloads.

## Mouse and other spatial platforms

Mouse-brain and platform runners use Squidpy datasets or cached AnnData files. Supply a writable cache directory, for example `data/squidpy/`, and inspect the relevant loader and runner before downloading. Dataset functions and annotation schemas can change across dependency versions. Data acquisition may require network access; summary regeneration does not.

## Additional analyses

Psoriasis uses GSE202011; the helper `scripts/preprocess_psoriasis.py` converts previously downloaded inputs. HEST regression inputs are downloaded from the gated Hugging Face dataset `MahmoodLab/hest` with `experiments/tsb/download_hest_v2.py` (breast IDC) and `experiments/tsb/download_hest_ccrcc.py` (ccRCC); both take the output directory as their first argument. These inputs are not bundled, and no claim is made that the small archived summary files can recreate raw tissue data. Participant identities are not included in this release.
