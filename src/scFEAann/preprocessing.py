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

    X: torch.Tensor | None
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


def _read_expression_matrix(
    adata: ad.AnnData,
    *,
    layer: str | None,
    use_raw: bool,
) -> tuple[Any, pd.Index, str]:
    matrix, var_names, source = resolve_expression(
        adata, layer=layer, use_raw=use_raw
    )
    if sparse.issparse(matrix):
        if not np.isfinite(matrix.data).all():
            raise ValueError(
                f"expression source '{source}' must contain only finite values"
            )
    else:
        dense = np.asarray(matrix)
        if not np.isfinite(dense).all():
            raise ValueError(
                f"expression source '{source}' must contain only finite values"
            )
    return matrix, var_names, source


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


def _materialize_module_x(
    gene_expression: torch.Tensor,
    module_masks: torch.Tensor,
    module_indices: list[int],
) -> torch.Tensor:
    n_cells, n_genes = gene_expression.shape
    X = torch.empty(
        (n_cells, len(module_indices) * n_genes),
        dtype=torch.float32,
    )
    for output_index, module_index in enumerate(module_indices):
        start = output_index * n_genes
        stop = start + n_genes
        torch.mul(
            gene_expression,
            module_masks[module_index],
            out=X[:, start:stop],
        )
    return X


def materialize_module_blocks(
    prepared: PreparedInputs,
    module_names: list[str],
) -> PreparedInputs:
    """Build an aligned module subset from deferred prepared inputs."""
    if not module_names:
        raise ValueError("at least one module is required")
    if len(set(module_names)) != len(module_names):
        raise ValueError("module_names must be unique")
    module_position = {
        name: index for index, name in enumerate(prepared.module_names)
    }
    missing = [name for name in module_names if name not in module_position]
    if missing:
        raise ValueError(f"modules are absent from the prepared model: {missing}")
    indices = [module_position[name] for name in module_names]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return PreparedInputs(
        X=_materialize_module_x(
            prepared.gene_expression,
            prepared.module_masks,
            indices,
        ),
        gene_expression=prepared.gene_expression,
        module_masks=prepared.module_masks.index_select(0, index_tensor),
        gene_scale=prepared.gene_scale,
        module_scale=prepared.module_scale.index_select(1, index_tensor),
        stoichiometry=prepared.stoichiometry.index_select(1, index_tensor),
        gene_names=list(prepared.gene_names),
        cell_names=list(prepared.cell_names),
        module_names=list(module_names),
        compound_names=list(prepared.compound_names),
        expression_source=prepared.expression_source,
        model_metadata=dict(prepared.model_metadata),
    )


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
    build_module_blocks: bool = True,
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

    source_matrix, var_names, source = _read_expression_matrix(
        adata, layer=layer, use_raw=use_raw
    )
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
    expression_gene_names = var_names.astype(str).tolist()
    overlap = sorted(set(expression_gene_names) & module_gene_set)
    if not overlap:
        raise ValueError(
            "expression genes and module genes have no overlap; check gene "
            "identifiers and the selected expression source"
        )
    gene_positions = {
        gene: index for index, gene in enumerate(expression_gene_names)
    }
    overlap_indices = [gene_positions[gene] for gene in overlap]
    cell_names = adata.obs_names.astype(str).tolist()

    if sc_imputation:
        if sparse.issparse(source_matrix):
            expression_values = source_matrix.toarray()
        else:
            expression_values = np.asarray(source_matrix)
        expression = pd.DataFrame(
            np.asarray(expression_values, dtype=np.float64),
            index=pd.Index(adata.obs_names),
            columns=pd.Index(expression_gene_names),
        )
        expression = _apply_magic(expression)
        if expression.to_numpy().max() > 50:
            expression = (expression + 1).apply(np.log2)
        gene_scale_values = expression.sum(axis=1).to_numpy(dtype=np.float64)
        overlap_values = expression.loc[:, overlap].to_numpy(dtype=np.float64)
    elif sparse.issparse(source_matrix):
        working = source_matrix.astype(np.float64, copy=True)
        if working.data.size and working.data.max() > 50:
            working.data = np.log2(working.data + 1)
        gene_scale_values = np.asarray(working.sum(axis=1)).ravel()
        overlap_values = working[:, overlap_indices].toarray()
        overlap_values = np.asarray(overlap_values, dtype=np.float64)
    else:
        expression_values = np.asarray(source_matrix, dtype=np.float64)
        if expression_values.shape[0] and expression_values.size and expression_values.max() > 50:
            expression_values = np.log2(expression_values + 1)
        gene_scale_values = expression_values.sum(axis=1)
        overlap_values = expression_values[:, overlap_indices]

    gene_scale = gene_scale_values / gene_scale_values.mean()

    n_modules = module_genes.shape[0]
    n_cells = overlap_values.shape[0]
    module_masks = np.empty((n_modules, len(overlap)), dtype=bool)
    module_scale = np.empty((n_cells, n_modules), dtype=np.float64)
    for module_index, (_, row) in enumerate(module_genes.iterrows()):
        genes = set(row.dropna().astype(str))
        mask = np.asarray([gene in genes for gene in overlap], dtype=bool)
        module_masks[module_index] = mask
        module_scale[:, module_index] = overlap_values[:, mask].sum(axis=1)

    module_masks_tensor = torch.tensor(module_masks, dtype=torch.bool)
    gene_scale_tensor = torch.tensor(
        np.asarray(gene_scale, dtype=np.float32), dtype=torch.float32
    )
    gene_expression_tensor = torch.tensor(
        np.asarray(overlap_values, dtype=np.float32), dtype=torch.float32
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
    if build_module_blocks:
        X = _materialize_module_x(
            gene_expression_tensor,
            module_masks_tensor,
            list(range(module_genes.shape[0])),
        )
    else:
        X = None

    return PreparedInputs(
        X=X,
        gene_expression=gene_expression_tensor,
        module_masks=module_masks_tensor,
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
