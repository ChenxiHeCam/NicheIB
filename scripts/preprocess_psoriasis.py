"""Preprocess GSE202011 psoriasis Visium data into per-sample h5ad files.

Reads Space Ranger H5 matrices + spatial position CSVs from GEO downloads,
combines into AnnData objects with spatial coordinates, and saves as h5ad.

Expected input layout (after extracting GEO tarballs):
  data_dir/raw/GSMxxxxxxx_ST_XX_Y_filtered_feature_bc_matrix.h5
  data_dir/images/images/ST_XX_Y_spatial/tissue_positions_list.csv.gz

Output:
  data_dir/h5ad/ST_XX_Y.h5ad   (one per sample)
  data_dir/h5ad/metadata.json   (sample -> patient, condition, PASI)
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import scanpy as sc


# --- Sample metadata ---
# Parsed from GEO sample descriptions and the publication (Castillo et al. 2023).
# Patient IDs in the data files use internal numbering (13-22, HM/HF for healthy).
# PASI scores are mapped from the paper's Table S1.
SAMPLE_META = {
    # PsO patients
    "ST_13": {"patient": "PsO_1", "disease": "PsO", "pasi": 11.2},
    "ST_14": {"patient": "PsO_2", "disease": "PsO", "pasi": 1.8},
    "ST_15": {"patient": "PsA_2", "disease": "PsA", "pasi": 5.0},
    "ST_16": {"patient": "PsO_3", "disease": "PsO", "pasi": 6.0},
    "ST_17": {"patient": "PsO_4", "disease": "PsO", "pasi": 32.0},
    "ST_18": {"patient": "PsA_3", "disease": "PsA", "pasi": 13.2},
    "ST_20": {"patient": "PsA_1", "disease": "PsA", "pasi": 28.8},
    "ST_21": {"patient": "PsO_5", "disease": "PsO", "pasi": 18.0},
    "ST_22": {"patient": "PsA_4", "disease": "PsA", "pasi": 8.4},
    # Lesional-only patients
    "ST_L_RP": {"patient": "PsO_6", "disease": "PsO", "pasi": 10.1},
    # Healthy controls
    "ST_HM_1": {"patient": "Healthy_M1", "disease": "Healthy", "pasi": 0.0},
    "ST_HF_1": {"patient": "Healthy_F1", "disease": "Healthy", "pasi": 0.0},
    "ST_HF_2": {"patient": "Healthy_F2", "disease": "Healthy", "pasi": 0.0},
}


def _parse_sample_name(filename: str) -> str:
    """Extract sample name from H5 filename (e.g. GSM6086615_ST_13_L -> ST_13_L)."""
    m = re.match(r"GSM\d+_(ST_.+)_filtered_feature_bc_matrix\.h5", filename)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse sample name from: {filename}")


def _get_patient_key(sample_name: str) -> str:
    """Map sample name to patient key in SAMPLE_META."""
    # Try exact prefix matches in decreasing specificity
    if sample_name.startswith("ST_L_RP"):
        return "ST_L_RP"
    if sample_name.startswith("ST_HM_1"):
        return "ST_HM_1"
    if sample_name.startswith("ST_HF_1") and "1T" in sample_name:
        return "ST_HF_1"
    if sample_name.startswith("ST_HF_2"):
        return "ST_HF_2"
    if sample_name.startswith("ST_HF_1"):
        return "ST_HF_1"
    # Disease patients: ST_XX_L or ST_XX_NL
    m = re.match(r"(ST_\d+)", sample_name)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot determine patient for: {sample_name}")


def _get_condition(sample_name: str) -> str:
    """Determine lesional/non-lesional/healthy from sample name."""
    if "_HM_" in sample_name or "_HF_" in sample_name:
        return "healthy"
    if "_NL" in sample_name:
        return "non_lesional"
    return "lesional"


def _find_spatial_dir(images_root: Path, sample_name: str) -> Path | None:
    """Find the spatial directory matching a sample name."""
    spatial_dir = images_root / f"{sample_name}_spatial"
    if spatial_dir.is_dir():
        return spatial_dir
    # Try without GSM prefix variations
    for d in images_root.iterdir():
        if d.is_dir() and sample_name in d.name:
            return d
    return None


def _read_positions(spatial_dir: Path) -> pd.DataFrame:
    """Read tissue_positions_list.csv(.gz) from spatial directory."""
    for name in ["tissue_positions_list.csv.gz", "tissue_positions_list.csv",
                 "tissue_positions.csv.gz", "tissue_positions.csv"]:
        path = spatial_dir / name
        if path.exists():
            df = pd.read_csv(path, header=None)
            # Standard format: barcode, in_tissue, row, col, pxl_row, pxl_col
            if df.shape[1] >= 6:
                df.columns = ["barcode", "in_tissue", "array_row", "array_col",
                              "pxl_row_in_fullres", "pxl_col_in_fullres"][:df.shape[1]]
            return df
    raise FileNotFoundError(f"No positions file in {spatial_dir}")


def preprocess_sample(h5_path: Path, spatial_dir: Path, out_dir: Path) -> dict:
    """Process one sample: read H5 + spatial positions, save as h5ad."""
    sample_name = _parse_sample_name(h5_path.name)
    condition = _get_condition(sample_name)
    patient_key = _get_patient_key(sample_name)
    meta = SAMPLE_META.get(patient_key, {"patient": patient_key, "disease": "unknown", "pasi": -1})

    # Read expression
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    # Read spatial positions
    positions = _read_positions(spatial_dir)
    positions = positions.set_index("barcode")

    # Align barcodes
    common = adata.obs_names.intersection(positions.index)
    if len(common) == 0:
        raise ValueError(f"No barcode overlap for {sample_name}")
    adata = adata[common].copy()
    positions = positions.loc[common]

    # Filter to tissue spots
    if "in_tissue" in positions.columns:
        in_tissue = positions["in_tissue"].astype(bool)
        adata = adata[in_tissue].copy()
        positions = positions[in_tissue]

    # Set spatial coordinates
    adata.obsm["spatial"] = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)

    # Store metadata in .obs (for downstream use)
    adata.obs["sample_id"] = sample_name
    adata.obs["patient"] = meta["patient"]
    adata.obs["disease"] = meta["disease"]
    adata.obs["condition"] = condition
    adata.obs["pasi"] = meta["pasi"]

    # Save
    out_path = out_dir / f"{sample_name}.h5ad"
    adata.write_h5ad(out_path)
    print(f"  {sample_name}: {adata.n_obs} spots, {adata.n_vars} genes, {condition}")

    return {
        "sample_id": sample_name,
        "patient": meta["patient"],
        "disease": meta["disease"],
        "condition": condition,
        "pasi": meta["pasi"],
        "n_spots": adata.n_obs,
        "n_genes": adata.n_vars,
        "h5ad_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Root dir containing raw/ and images/ subdirectories")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output dir for h5ad files (default: data_dir/h5ad)")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    # Handle nested images/images/ structure from tar extraction
    images_dir = args.data_dir / "images" / "images"
    if not images_dir.is_dir():
        images_dir = args.data_dir / "images"
    out_dir = args.output_dir or args.data_dir / "h5ad"
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(raw_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files in {raw_dir}")
    print(f"Spatial dirs in {images_dir}")

    records = []
    errors = []
    for h5_path in h5_files:
        sample_name = _parse_sample_name(h5_path.name)
        spatial_dir = _find_spatial_dir(images_dir, sample_name)
        if spatial_dir is None:
            print(f"  SKIP {sample_name}: no spatial dir found")
            errors.append(sample_name)
            continue
        try:
            record = preprocess_sample(h5_path, spatial_dir, out_dir)
            records.append(record)
        except Exception as e:
            print(f"  ERROR {sample_name}: {e}")
            errors.append(sample_name)

    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"samples": records, "errors": errors}, f, indent=2)
    print(f"\nProcessed {len(records)}/{len(h5_files)} samples")
    print(f"Metadata saved to {meta_path}")

    # Summary
    conditions = {}
    for r in records:
        conditions.setdefault(r["condition"], []).append(r["sample_id"])
    for cond, samples in sorted(conditions.items()):
        print(f"  {cond}: {len(samples)} samples")


if __name__ == "__main__":
    main()
