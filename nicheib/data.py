"""Data loading and spatial graph construction for NicheIB."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

try:
    import scanpy as sc
except Exception:
    sc = None

try:
    from torch_geometric.data import Data
except ModuleNotFoundError:
    Data = None

IPMN_PDAC_LABELS = {"LG": 0, "HG": 1, "PDAC": 2}
IPMN_PDAC_LABEL_NAMES = ["LG", "HG", "PDAC"]
IPMN_PDAC_ADVANCED_LABELS = {"LG": 0, "HG": 1, "PDAC": 1}
IPMN_PDAC_ADVANCED_LABEL_NAMES = ["LG", "advanced"]


def _require_pyg() -> None:
    if Data is None:
        raise ModuleNotFoundError("torch_geometric is required for spatial graph construction")


def _require_scanpy() -> None:
    if sc is None:
        raise ModuleNotFoundError("scanpy is required for spatial transcriptomics data loading")


def build_spatial_graph(
    adata: sc.AnnData,
    n_neighbors: int = 6,
    n_hvgs: int = 3000,
    n_pcs: int = 200,
    spatial_key: str = "spatial",
) -> Data:
    """Construct a PyG Data object from an AnnData spatial transcriptomics sample.

    Pipeline: HVG selection -> normalize -> log1p -> scale -> PCA -> kNN spatial graph.

    Args:
        adata: AnnData with .obsm[spatial_key] containing (x, y) coordinates.
               Must contain raw counts for HVG selection (seurat_v3 flavor).
        n_neighbors: k for spatial kNN graph (6 for Visium hexagonal grid)
        n_hvgs: number of highly variable genes to select
        n_pcs: number of PCA components for node features
        spatial_key: key in adata.obsm for spatial coordinates
    """
    _require_pyg()
    _require_scanpy()
    if spatial_key not in adata.obsm:
        raise KeyError(f"AnnData is missing obsm['{spatial_key}']")
    if adata.n_obs <= 1:
        raise ValueError("At least two spots are required to build a spatial graph")

    n_hvgs = min(n_hvgs, adata.n_vars)
    if n_hvgs <= 0:
        raise ValueError("At least one gene is required to build node features")

    # Work on a gene-subset copy only (not full adata copy — saves memory)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor="seurat_v3", inplace=True)
    except Exception:
        x_raw = adata.X
        if hasattr(x_raw, "multiply"):
            mean = np.asarray(x_raw.mean(axis=0)).ravel()
            mean_sq = np.asarray(x_raw.multiply(x_raw).mean(axis=0)).ravel()
        else:
            x_dense = np.asarray(x_raw)
            mean = x_dense.mean(axis=0)
            mean_sq = np.square(x_dense).mean(axis=0)
        variance = np.maximum(mean_sq - np.square(mean), 0.0)
        keep = np.zeros(adata.n_vars, dtype=bool)
        n_keep = min(n_hvgs, adata.n_vars)
        keep[np.argsort(variance)[-n_keep:]] = True
        adata.var["highly_variable"] = keep
    adata_sub = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    sc.pp.scale(adata_sub, max_value=10)
    n_comps = min(n_pcs, max(1, adata_sub.n_vars - 1), max(1, adata_sub.n_obs - 1))
    if n_comps == 1:
        x_array = np.asarray(adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X, dtype=np.float32)
        x_array = x_array[:, :1]
    else:
        sc.tl.pca(adata_sub, n_comps=n_comps)
        x_array = adata_sub.obsm["X_pca"].astype(np.float32)

    x = torch.tensor(x_array, dtype=torch.float32)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    # Spatial kNN graph — vectorized construction
    k = min(n_neighbors, adata.n_obs - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Drop self-edges (column 0 is the point itself)
    n = len(coords)
    src = np.repeat(np.arange(n), k)
    dst = indices[:, 1:].ravel()
    edge_dist = distances[:, 1:].ravel()

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Edge weights: inverse distance, normalized to [0, 1]
    edge_weight = 1.0 / (edge_dist + 1e-6)
    edge_weight = edge_weight / edge_weight.max()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    data.coords = torch.tensor(coords, dtype=torch.float32)
    return data


def _knn_edges(features: np.ndarray, k: int, metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build directed kNN edge arrays from a feature matrix."""
    k_actual = min(k, len(features) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=metric)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    n = len(features)
    src = np.repeat(np.arange(n), k_actual)
    dst = indices[:, 1:].ravel()
    dist = distances[:, 1:].ravel()
    return src, dst, dist


def build_multirelational_graph(
    adata,
    n_neighbors: int = 6,
    n_hvgs: int = 3000,
    n_pcs: int = 200,
    spatial_key: str = "spatial",
    expr_neighbors: int = 0,
    morph_neighbors: int = 0,
    morph_key: str | None = None,
) -> "Data":
    """Build a spatial graph with optional expression and morphology edge types.

    Edge types:
        0 — spatial kNN (always present)
        1 — expression cosine kNN (when expr_neighbors > 0)
        2 — morphology embedding kNN (when morph_neighbors > 0 and morph_key exists)

    Returns a Data object with an additional `edge_type` LongTensor.
    """
    _require_pyg()
    _require_scanpy()

    # Start with the standard spatial graph
    data = build_spatial_graph(adata, n_neighbors=n_neighbors, n_hvgs=n_hvgs,
                               n_pcs=n_pcs, spatial_key=spatial_key)
    n_spatial = data.edge_index.shape[1]
    edge_type = torch.zeros(n_spatial, dtype=torch.long)

    all_src = [data.edge_index[0]]
    all_dst = [data.edge_index[1]]
    all_weight = [data.edge_attr]
    all_type = [edge_type]

    x_np = data.x.numpy()

    # Expression cosine kNN edges (type 1)
    if expr_neighbors > 0 and x_np.shape[0] > 1:
        from sklearn.preprocessing import normalize
        x_normed = normalize(x_np, axis=1)
        src, dst, dist = _knn_edges(x_normed, expr_neighbors, metric="cosine")
        weight = 1.0 / (dist + 1e-6)
        weight = weight / (weight.max() + 1e-12)
        all_src.append(torch.tensor(src, dtype=torch.long))
        all_dst.append(torch.tensor(dst, dtype=torch.long))
        all_weight.append(torch.tensor(weight, dtype=torch.float32))
        all_type.append(torch.ones(len(src), dtype=torch.long))

    # Morphology embedding kNN edges (type 2)
    if morph_neighbors > 0 and morph_key is not None and morph_key in adata.obsm:
        morph_feat = np.asarray(adata.obsm[morph_key], dtype=np.float64)
        src, dst, dist = _knn_edges(morph_feat, morph_neighbors, metric="euclidean")
        weight = 1.0 / (dist + 1e-6)
        weight = weight / (weight.max() + 1e-12)
        all_src.append(torch.tensor(src, dtype=torch.long))
        all_dst.append(torch.tensor(dst, dtype=torch.long))
        all_weight.append(torch.tensor(weight, dtype=torch.float32))
        all_type.append(torch.full((len(src),), 2, dtype=torch.long))

    data.edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
    data.edge_attr = torch.cat(all_weight)
    data.edge_type = torch.cat(all_type)
    data.n_edge_types = 1 + (expr_neighbors > 0) + (morph_neighbors > 0 and morph_key is not None and morph_key in adata.obsm)
    return data


def ensure_not_large_local_path(path: str, allow_local: bool = False) -> None:
    """Guard against accidental large dataset writes to the laptop.

    Full TSB datasets should live on compute-server scratch. Local paths are allowed only
    for tiny smoke fixtures when `allow_local=True`.
    """
    resolved = Path(path).expanduser().resolve()
    if str(resolved).startswith("/Users/") and not allow_local:
        raise ValueError(
            f"Refusing local data path '{resolved}'. Use compute-server scratch for large TSB data "
            "or pass allow_local=True for a tiny smoke fixture."
        )


def _find_label_column(obs_columns, candidates):
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in obs_columns:
            return c
    return None


def _matrix_variance(x) -> np.ndarray:
    if hasattr(x, "multiply"):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        x_dense = np.asarray(x)
        mean = x_dense.mean(axis=0)
        mean_sq = np.square(x_dense).mean(axis=0)
    return np.maximum(mean_sq - np.square(mean), 0.0)


def _resolve_ipmn_h5ad_dir(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    candidates = [root, root / "processed_h5ad", root / "ipmn_pdac" / "processed_h5ad"]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.h5ad")):
            return candidate
    return candidates[-1]


def _normalize_ipmn_grade(value: str) -> str:
    text = str(value).strip().upper()
    aliases = {
        "LOW": "LG",
        "LOW_GRADE": "LG",
        "LOW-GRADE": "LG",
        "LOW GRADE": "LG",
        "HIGH": "HG",
        "HIGH_GRADE": "HG",
        "HIGH-GRADE": "HG",
        "HIGH GRADE": "HG",
    }
    return aliases.get(text, text)


def _select_shared_genes(adatas: list[sc.AnnData], n_hvgs: int) -> list[str]:
    common = set(adatas[0].var_names)
    for adata in adatas[1:]:
        common &= set(adata.var_names)
    if not common:
        raise ValueError("IPMN/PDAC sections have no shared genes")

    common_genes = np.array(sorted(common), dtype=object)
    variance_sum = np.zeros(common_genes.shape[0], dtype=np.float64)
    for adata in adatas:
        indexer = adata.var_names.get_indexer(common_genes)
        subset = adata[:, indexer]
        variance_sum += _matrix_variance(subset.X)

    n_keep = min(n_hvgs, common_genes.shape[0])
    keep_idx = np.argsort(variance_sum)[-n_keep:]
    return sorted(str(gene) for gene in common_genes[keep_idx])


def load_dlpfc(
    sample_id: str,
    data_dir: str = "data/dlpfc",
    n_neighbors: int = 6,
    n_hvgs: int = 3000,
    n_pcs: int = 200,
    spatial_key: str = "spatial",
) -> Data:
    """Load a DLPFC Visium section with ground-truth layer annotations."""
    path = os.path.join(data_dir, f"{sample_id}.h5ad")
    _require_scanpy()
    adata = sc.read_h5ad(path)
    data = build_spatial_graph(
        adata,
        n_neighbors=n_neighbors,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        spatial_key=spatial_key,
    )

    gt_col = _find_label_column(
        adata.obs.columns, ["ground_truth", "sce.layer_guess", "layer_guess"]
    )
    if gt_col is not None:
        gt = adata.obs[gt_col]
        valid = gt.notna()
        if not valid.all():
            data.valid_mask = torch.tensor(valid.values, dtype=torch.bool)
        labels = gt[valid].astype("category")
        codes = np.full(len(gt), -1, dtype=np.int64)
        codes[valid.values] = labels.cat.codes.values
        data.y = torch.tensor(codes, dtype=torch.long)
        data.label_names = list(labels.cat.categories)
    else:
        data.valid_mask = torch.ones(data.x.shape[0], dtype=torch.bool)

    return data


def load_hest_subset(
    data_dir: str = "data/hest",
    organ: str | None = None,
    oncotree_code: str | None = None,
    n_neighbors: int = 6,
    n_hvgs: int = 3000,
    n_pcs: int = 200,
    spatial_key: str = "spatial",
    label_key: str | None = None,
) -> list[Data]:
    """Load HEST-1k spatial transcriptomics sections as graph data.

    Each h5ad file in data_dir/st/ becomes one graph. If label_key is provided,
    graph-level or node-level labels are extracted from adata.obs.
    """
    _require_scanpy()
    _require_pyg()
    st_dir = Path(data_dir) / "st"
    if not st_dir.exists():
        st_dir = Path(data_dir)

    paths = sorted(st_dir.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No h5ad files found under {st_dir}")

    sections = []
    for path in paths:
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
        if spatial_key not in adata.obsm:
            print(f"  {path.stem}: missing obsm['{spatial_key}'], skipping")
            continue

        data = build_spatial_graph(
            adata,
            n_neighbors=n_neighbors,
            n_hvgs=min(n_hvgs, adata.n_vars),
            n_pcs=n_pcs,
            spatial_key=spatial_key,
        )
        data.sample_id = path.stem
        data.sample_name = path.stem

        # Extract organ/oncotree metadata if available
        for meta_key in ["organ", "oncotree_code", "species", "technology"]:
            if meta_key in adata.obs.columns:
                setattr(data, meta_key, str(adata.obs[meta_key].iloc[0]))
            elif meta_key in adata.uns:
                setattr(data, meta_key, str(adata.uns[meta_key]))

        # Extract labels if requested
        if label_key and label_key in adata.obs.columns:
            labels = adata.obs[label_key].astype("category")
            data.y = torch.tensor(labels.cat.codes.values, dtype=torch.long)
            data.label_names = list(labels.cat.categories)

        sections.append(data)
        print(f"  {path.stem}: {data.x.shape[0]} spots, {data.x.shape[1]} features")

    return sections


def load_breast_cancer(data_dir: str = "data/breast") -> list[Data]:
    """Load 10x Genomics breast cancer Visium sections.

    These sections are useful for smoke tests and exploratory biology. They do
    not provide direct outcome labels, so pseudo-labels should not be used as
    primary validation for TSB.
    """
    _require_scanpy()
    sections = []
    for name in ["breast_A1", "breast_A2"]:
        h5ad_path = os.path.join(data_dir, f"{name}.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"  {name}: not found at {h5ad_path}, skipping")
            continue
        adata = sc.read_h5ad(h5ad_path)
        data = build_spatial_graph(adata)
        data.sample_name = name

        # Optional exploratory labels only. Do not treat these as external validation.
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
        sc.tl.leiden(adata, resolution=0.8)
        data.y = torch.tensor(adata.obs["leiden"].astype(int).values, dtype=torch.long)
        data.valid_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        data.is_proxy_target = True

        sections.append(data)
        print(f"  {name}: {data.x.shape[0]} spots, {data.x.shape[1]} features")

    return sections


def load_sthelar(
    data_dir: str = "data/sthelar",
    section_ids: list[str] | None = None,
    n_neighbors: int = 6,
    n_hvgs: int = 3000,
    n_pcs: int = 200,
    spatial_key: str = "spatial",
    cell_type_key: str = "cell_type",
) -> list[Data]:
    """Load STHELAR Xenium sections with cell-type annotations.

    Each h5ad file becomes one graph with node-level cell-type labels.
    """
    _require_scanpy()
    _require_pyg()
    root = Path(data_dir)
    paths = sorted(root.glob("*.h5ad"))
    if not paths:
        # Try nested directory structures
        for sub in ["sections", "st", "data"]:
            sub_dir = root / sub
            if sub_dir.exists():
                paths = sorted(sub_dir.glob("*.h5ad"))
                if paths:
                    break
    if not paths:
        raise FileNotFoundError(f"No h5ad files found under {root}")

    if section_ids is not None:
        id_set = set(section_ids)
        paths = [p for p in paths if p.stem in id_set]
        if not paths:
            raise FileNotFoundError(f"None of the requested section_ids found: {section_ids}")

    sections = []
    for path in paths:
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
        if spatial_key not in adata.obsm:
            print(f"  {path.stem}: missing obsm['{spatial_key}'], skipping")
            continue

        data = build_spatial_graph(
            adata,
            n_neighbors=n_neighbors,
            n_hvgs=min(n_hvgs, adata.n_vars),
            n_pcs=n_pcs,
            spatial_key=spatial_key,
        )
        data.sample_id = path.stem
        data.sample_name = path.stem

        # Extract cell-type labels
        ct_col = _find_label_column(
            adata.obs.columns,
            [cell_type_key, "celltype", "cell_label", "CellType", "cluster"],
        )
        if ct_col is not None:
            labels = adata.obs[ct_col]
            valid = labels.notna()
            if not valid.all():
                data.valid_mask = torch.tensor(valid.values, dtype=torch.bool)
            labels_cat = labels[valid].astype("category")
            codes = np.full(len(labels), -1, dtype=np.int64)
            codes[valid.values] = labels_cat.cat.codes.values
            data.y = torch.tensor(codes, dtype=torch.long)
            data.label_names = list(labels_cat.cat.categories)
        else:
            print(f"  {path.stem}: no cell-type column found, loading without labels")

        sections.append(data)
        print(f"  {path.stem}: {data.x.shape[0]} cells, {data.x.shape[1]} features")

    return sections


def load_mouse_brain(
    data_dir: str | None = None,
    n_hvgs: int = 1000,
    n_pcs: int = 64,
    n_neighbors: int = 6,
) -> Data:
    """Load squidpy mouse brain coronal section with 15 annotated brain regions.

    Downloads via ``squidpy.datasets.visium_hne_adata()`` on first call.
    The label column is ``cluster`` (15 brain region annotations).

    Returns a PyG Data object compatible with the DLPFC pipeline:
        data.x         — (n_spots, n_pcs) float32
        data.edge_index — (2, n_edges) long
        data.edge_attr  — (n_edges,) float32 edge weights
        data.y          — (n_spots,) long  (0..14)
        data.coords     — (n_spots, 2) float32
        data.valid_mask  — (n_spots,) bool
        data.label_names — list[str] of cluster names
    """
    _require_pyg()
    _require_scanpy()

    # Prefer a pre-saved copy: concurrent jobs calling the squidpy downloader race on
    # its cache file and corrupt it, which is a real hazard under a SLURM job array.
    local = Path(data_dir) / "visium_hne.h5ad" if data_dir else None
    if local is not None and local.exists():
        adata = sc.read_h5ad(local)
    else:
        import squidpy as sq

        adata = sq.datasets.visium_hne_adata()

    # Build spatial graph (HVG -> normalize -> PCA -> kNN)
    data = build_spatial_graph(
        adata,
        n_neighbors=n_neighbors,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        spatial_key="spatial",
    )

    # Extract cluster labels (15 brain regions)
    label_col = "cluster"
    if label_col not in adata.obs.columns:
        raise KeyError(f"Expected label column '{label_col}' in adata.obs")

    labels = adata.obs[label_col]
    valid = labels.notna()
    if not valid.all():
        data.valid_mask = torch.tensor(valid.values, dtype=torch.bool)
    else:
        data.valid_mask = torch.ones(data.x.shape[0], dtype=torch.bool)

    labels_cat = labels[valid].astype("category")
    codes = np.full(len(labels), -1, dtype=np.int64)
    codes[valid.values] = labels_cat.cat.codes.values
    data.y = torch.tensor(codes, dtype=torch.long)
    data.label_names = list(labels_cat.cat.categories)

    return data


def load_ipmn_pdac(
    data_dir: str = "data/ipmn_pdac",
    n_neighbors: int = 6,
    n_hvgs: int = 3000,
    n_pcs: int = 200,
    spatial_key: str = "spatial",
    label_mode: str = "grade",
) -> list[Data]:
    """Load GSE233293 IPMN/PDAC sections as graph-level classification samples."""
    _require_scanpy()
    h5ad_dir = _resolve_ipmn_h5ad_dir(data_dir)
    paths = sorted(h5ad_dir.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No processed IPMN/PDAC h5ad files found under {h5ad_dir}")

    adatas = []
    for path in paths:
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
        if spatial_key not in adata.obsm:
            raise KeyError(f"{path} is missing obsm['{spatial_key}']")
        adatas.append(adata)

    if label_mode == "grade":
        label_map = IPMN_PDAC_LABELS
        label_names = IPMN_PDAC_LABEL_NAMES
    elif label_mode == "advanced":
        label_map = IPMN_PDAC_ADVANCED_LABELS
        label_names = IPMN_PDAC_ADVANCED_LABEL_NAMES
    else:
        raise ValueError(f"Unknown IPMN/PDAC label mode: {label_mode}")

    shared_genes = _select_shared_genes(adatas, n_hvgs)
    sections = []
    for path, adata in zip(paths, adatas):
        grade_col = _find_label_column(adata.obs.columns, ["grade", "condition", "diagnosis"])
        if grade_col is None:
            grade = _normalize_ipmn_grade(path.stem.split("_")[0])
        else:
            grade = _normalize_ipmn_grade(adata.obs[grade_col].iloc[0])
        if grade not in label_map:
            raise ValueError(f"Unknown IPMN/PDAC grade '{grade}' in {path}")

        sample_id = str(adata.obs["sample_id"].iloc[0]) if "sample_id" in adata.obs else path.stem
        adata_section = adata[:, shared_genes].copy()
        data = build_spatial_graph(
            adata_section,
            n_neighbors=n_neighbors,
            n_hvgs=len(shared_genes),
            n_pcs=n_pcs,
            spatial_key=spatial_key,
        )
        data.y = torch.tensor([label_map[grade]], dtype=torch.long)
        data.sample_id = sample_id
        data.sample_name = sample_id
        data.grade = grade
        data.label_mode = label_mode
        data.label_names = label_names
        data.n_spots = int(adata.n_obs)
        sections.append(data)

    return sections


# ---------------------------------------------------------------------------
# GSE202011 Psoriasis Visium (graph-level classification)
# ---------------------------------------------------------------------------

PSORIASIS_LABELS = {"non_lesional": 0, "lesional": 1}
PSORIASIS_LABEL_NAMES = ["non_lesional", "lesional"]
PSORIASIS_3CLASS_LABELS = {"healthy": 0, "non_lesional": 1, "lesional": 2}
PSORIASIS_3CLASS_LABEL_NAMES = ["healthy", "non_lesional", "lesional"]


def load_psoriasis(
    data_dir: str = "data/psoriasis",
    n_neighbors: int = 6,
    n_hvgs: int = 1000,
    n_pcs: int = 64,
    spatial_key: str = "spatial",
    label_mode: str = "binary",
) -> list[Data]:
    """Load GSE202011 psoriasis Visium sections as graph-level classification samples.

    Args:
        data_dir: Root directory containing h5ad/ subdirectory with preprocessed files.
        n_neighbors: k for spatial kNN graph.
        n_hvgs: number of HVGs for node features.
        n_pcs: number of PCA components.
        spatial_key: key in adata.obsm for spatial coordinates.
        label_mode: "binary" (lesional vs non-lesional, excludes healthy)
                    or "3class" (healthy vs non-lesional vs lesional).

    Returns:
        List of PyG Data objects with graph-level labels.
    """
    _require_scanpy()
    h5ad_dir = Path(data_dir) / "h5ad"
    if not h5ad_dir.is_dir():
        h5ad_dir = Path(data_dir)
    paths = sorted(h5ad_dir.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No h5ad files found under {h5ad_dir}")

    if label_mode == "binary":
        label_map = PSORIASIS_LABELS
        label_names = PSORIASIS_LABEL_NAMES
    elif label_mode == "3class":
        label_map = PSORIASIS_3CLASS_LABELS
        label_names = PSORIASIS_3CLASS_LABEL_NAMES
    else:
        raise ValueError(f"Unknown psoriasis label mode: {label_mode}")

    adatas = []
    conditions = []
    sample_paths = []
    for path in paths:
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
        if spatial_key not in adata.obsm:
            raise KeyError(f"{path} is missing obsm['{spatial_key}']")
        condition = str(adata.obs["condition"].iloc[0]) if "condition" in adata.obs else "unknown"
        if condition not in label_map:
            if label_mode == "binary":
                continue  # skip healthy in binary mode
            raise ValueError(f"Unknown condition '{condition}' in {path}")
        adatas.append(adata)
        conditions.append(condition)
        sample_paths.append(path)

    shared_genes = _select_shared_genes(adatas, n_hvgs)
    sections = []
    for path, adata, condition in zip(sample_paths, adatas, conditions):
        sample_id = str(adata.obs["sample_id"].iloc[0]) if "sample_id" in adata.obs else path.stem
        patient = str(adata.obs["patient"].iloc[0]) if "patient" in adata.obs else "unknown"
        pasi = float(adata.obs["pasi"].iloc[0]) if "pasi" in adata.obs else -1.0

        adata_section = adata[:, shared_genes].copy()
        data = build_spatial_graph(
            adata_section,
            n_neighbors=n_neighbors,
            n_hvgs=len(shared_genes),
            n_pcs=n_pcs,
            spatial_key=spatial_key,
        )
        data.y = torch.tensor([label_map[condition]], dtype=torch.long)
        data.sample_id = sample_id
        data.sample_name = sample_id
        data.patient = patient
        data.condition = condition
        data.pasi = pasi
        data.label_mode = label_mode
        data.label_names = label_names
        data.n_spots = int(adata.n_obs)
        sections.append(data)

    return sections
