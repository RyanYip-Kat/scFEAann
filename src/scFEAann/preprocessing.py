"""Convert AnnData expression data into the tensors used by scFEA."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .expression import resolve_expression


PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODULE_GENE_FILE = "module_gene_m168.csv"
DEFAULT_STOICHIOMETRY_MATRIX = "cmMat_c70_m168.csv"
DEFAULT_COMPOUND_NAME_FILE = "cName_c70_m168.csv"


@dataclass
class PreparedInputs:
    """AnnData-independent inputs consumed by the scFEA core."""

    X: torch.Tensor
    gene_expression: torch.Tensor
    module_masks: torch.Tensor
    gene_scale: torch.Tensor
    module_scale: torch.Tensor
    stoichiometry: torch.Tensor
    gene_names: list[str]
    cell_names: list[str]
    module_names: list[str]
    compound_names: list[str | int]
    expression_source: str
    model_metadata: dict[str, Any]


def _model_path(data_dir: str | Path | None, filename: str) -> Path:
    base = PACKAGE_DATA_DIR if data_dir is None else Path(data_dir)
    path = Path(filename)
    return path if path.is_absolute() else base / path


def _read_expression(
    adata: ad.AnnData,
    *,
    layer: str | None,
    use_raw: bool,
) -> tuple[pd.DataFrame, str]:
    matrix, var_names, source = resolve_expression(
        adata, layer=layer, use_raw=use_raw
    )
    if sparse.issparse(matrix):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense = np.asarray(dense, dtype=np.float64)
    if not np.isfinite(dense).all():
        raise ValueError(
            f"expression source '{source}' must contain only finite values"
        )
    return (
        pd.DataFrame(
            dense,
            index=pd.Index(adata.obs_names),
            columns=pd.Index(var_names),
        ),
        source,
    )


def _apply_magic(expression: pd.DataFrame) -> pd.DataFrame:
    try:
        import magic
    except ImportError as error:
        raise ImportError(
            "sc_imputation=True requires the optional 'magic-impute' package. "
            "Install it with `uv sync --extra magic`."
        ) from error

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformed = magic.MAGIC().fit_transform(expression)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    return pd.DataFrame(
        np.asarray(transformed, dtype=np.float64),
        index=expression.index,
        columns=expression.columns,
    )


def _read_compound_names(path: Path, n_compounds: int) -> list[str | int]:
    if n_compounds == 0:
        return []
    if not path.exists():
        raise ValueError(f"compound name file does not exist: {path}")
    compound_table = pd.read_csv(path, sep=",", header=0)
    names: list[str | int] = list(compound_table.columns)
    if len(names) != n_compounds:
        raise ValueError(
            f"compound name file has {len(names)} names but the stoichiometry "
            f"matrix has {n_compounds} rows"
        )
    return names


def prepare_scfea_inputs(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    sc_imputation: bool = False,
    data_dir: str | Path | None = None,
    module_gene_file: str | Path | None = None,
    stoichiometry_matrix: str | Path | None = None,
    compound_name_file: str | Path | None = None,
) -> PreparedInputs:
    """Prepare scFEA tensors while preserving the legacy preprocessing order.

    Log transformation and gene-scale normalization are computed on the full
    selected expression matrix before restricting to module genes.
    """
    module_path = _model_path(
        data_dir, module_gene_file or DEFAULT_MODULE_GENE_FILE
    )
    stoichiometry_path = _model_path(
        data_dir, stoichiometry_matrix or DEFAULT_STOICHIOMETRY_MATRIX
    )
    compound_path = _model_path(
        data_dir, compound_name_file or DEFAULT_COMPOUND_NAME_FILE
    )
    for label, path in (
        ("module gene", module_path),
        ("stoichiometry", stoichiometry_path),
    ):
        if not path.exists():
            raise ValueError(f"{label} file does not exist: {path}")

    expression, source = _read_expression(
        adata, layer=layer, use_raw=use_raw
    )
    if sc_imputation:
        expression = _apply_magic(expression)
    if expression.to_numpy().max() > 50:
        expression = (expression + 1).apply(np.log2)

    gene_scale = expression.sum(axis=1)
    gene_scale = gene_scale / gene_scale.mean()

    module_genes = pd.read_csv(module_path, sep=",", index_col=0)
    if module_genes.empty:
        raise ValueError(f"module gene file is empty: {module_path}")
    module_lengths = module_genes.notna().sum(axis=1).to_numpy(dtype=np.int64)
    if np.any(module_lengths == 0):
        empty = module_genes.index[module_lengths == 0].tolist()
        raise ValueError(f"module gene file contains empty modules: {empty}")

    module_gene_set = set(
        module_genes.stack().dropna().astype(str).tolist()
    )
    overlap = sorted(set(expression.columns.astype(str)) & module_gene_set)
    if not overlap:
        raise ValueError(
            "expression genes and module genes have no overlap; check gene "
            "identifiers and the selected expression source"
        )
    expression = expression.loc[:, overlap]
    cell_names = expression.index.astype(str).tolist()

    module_blocks: list[pd.DataFrame] = []
    module_masks: list[list[bool]] = []
    module_scale_columns: dict[str, np.ndarray] = {}
    for module_name, row in module_genes.iterrows():
        genes = set(row.dropna().astype(str))
        module_masks.append([gene in genes for gene in overlap])
        masked = expression.copy()
        masked.loc[:, [gene for gene in overlap if gene not in genes]] = 0
        block = masked.T
        block["Module_Gene"] = [
            f"{module_index:02d}_{gene}"
            for module_index, gene in enumerate(overlap)
        ]
        module_blocks.append(block)
        module_scale_columns[str(module_name)] = masked.sum(axis=1).to_numpy()

    module_frame = pd.concat(module_blocks, axis=0)
    X = torch.tensor(
        module_frame.loc[:, cell_names].to_numpy(dtype=np.float32).T,
        dtype=torch.float32,
    )
    gene_scale_tensor = torch.tensor(
        gene_scale.to_numpy(dtype=np.float32), dtype=torch.float32
    )

    module_scale = np.column_stack(
        [module_scale_columns[str(name)] for name in module_genes.index]
    )
    module_scale = module_scale / module_lengths[np.newaxis, :]
    module_scale_tensor = torch.tensor(
        module_scale, dtype=torch.float32
    )

    stoichiometry_frame = pd.read_csv(
        stoichiometry_path, sep=",", header=None
    )
    stoichiometry = stoichiometry_frame.to_numpy(dtype=np.float32)
    if stoichiometry.ndim != 2:
        raise ValueError(f"stoichiometry matrix must be two-dimensional: {stoichiometry_path}")
    if stoichiometry.shape[1] != module_genes.shape[0]:
        raise ValueError(
            f"stoichiometry matrix has {stoichiometry.shape[1]} module columns "
            f"but the module gene file has {module_genes.shape[0]} rows"
        )
    compound_names = _read_compound_names(
        compound_path, stoichiometry.shape[0]
    )

    return PreparedInputs(
        X=X,
        gene_expression=torch.tensor(
            expression.to_numpy(dtype=np.float32), dtype=torch.float32
        ),
        module_masks=torch.tensor(module_masks, dtype=torch.bool),
        gene_scale=gene_scale_tensor,
        module_scale=module_scale_tensor,
        stoichiometry=torch.tensor(stoichiometry, dtype=torch.float32),
        gene_names=overlap,
        cell_names=cell_names,
        module_names=[str(name) for name in module_genes.index],
        compound_names=compound_names,
        expression_source=source,
        model_metadata={
            "data_dir": str(PACKAGE_DATA_DIR if data_dir is None else Path(data_dir)),
            "module_gene_file": str(module_path),
            "stoichiometry_matrix": str(stoichiometry_path),
            "compound_name_file": str(compound_path),
        },
    )
