"""Read and write scFEA results on AnnData objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


METADATA_KEY = "scFEAann"
FLUX_KEY = "scFEA_flux"
BALANCE_KEY = "scFEA_balance"
GRADIENT_KEY = "scFEA_grad"
SUPERMODULE_FLUX_KEY = "scFEA_supermodule_flux"
SUPERMODULE_BALANCE_KEY = "scFEA_supermodule_balance"
SUPERMODULE_FLUX_SUM_KEY = "scFEA_supermodule_flux_sum"
SUPERMODULE_FLUX_MEAN_KEY = "scFEA_supermodule_flux_mean"


def _root_metadata(adata: ad.AnnData) -> dict[str, Any]:
    if METADATA_KEY not in adata.uns:
        adata.uns[METADATA_KEY] = {}
    metadata = adata.uns[METADATA_KEY]
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"adata.uns['{METADATA_KEY}'] exists but is not a mapping"
        )
    return metadata


def validate_group_key(adata: ad.AnnData, group_key: str | None) -> None:
    """Validate an optional observation grouping column without using it in fit."""
    if group_key is None:
        return
    if not isinstance(group_key, str) or not group_key:
        raise ValueError("group_key must be a non-empty obs column name")
    if group_key not in adata.obs.columns:
        raise ValueError(
            f"group_key '{group_key}' is not present in adata.obs"
        )


def write_flux_results(
    adata: ad.AnnData,
    flux: np.ndarray,
    balance: np.ndarray,
    *,
    module_names: list[str],
    compound_names: list[str | int],
    group_key: str | None,
    expression_source: str,
    gene_names: list[str],
    model_metadata: dict[str, Any],
    parameters: dict[str, Any],
    loss_history: dict[str, np.ndarray],
    elapsed_seconds: float,
) -> None:
    """Write flux, balance, and traceable metadata into an AnnData object."""
    adata.obsm[FLUX_KEY] = np.asarray(flux, dtype=np.float32)
    adata.obsm[BALANCE_KEY] = np.asarray(balance, dtype=np.float32)
    metadata = _root_metadata(adata)
    metadata["group_key"] = group_key
    metadata["expression_source"] = expression_source
    metadata["gene_names"] = list(gene_names)
    metadata["model"] = dict(model_metadata)
    metadata["parameters"] = dict(parameters)
    metadata["flux"] = {
        "obsm_key": FLUX_KEY,
        "group_key": group_key,
        "expression_source": expression_source,
        "gene_names": list(gene_names),
        "model": dict(model_metadata),
        "parameters": dict(parameters),
        "module_names": list(module_names),
        "loss_history": {
            key: np.asarray(value, dtype=np.float64)
            for key, value in loss_history.items()
        },
        "elapsed_seconds": float(elapsed_seconds),
    }
    metadata["balance"] = {
        "obsm_key": BALANCE_KEY,
        "compound_names": [str(name) for name in compound_names],
    }


def write_gradient_results(
    adata: ad.AnnData,
    gradient: np.ndarray,
    *,
    module_names: list[str],
    gene_names: list[str],
    group_key: str | None,
    expression_source: str,
    model_metadata: dict[str, Any],
    parameters: dict[str, Any],
    loss_history: dict[str, np.ndarray],
    elapsed_seconds: float,
) -> None:
    """Write a C-order flattened gene-by-module gradient and its axis metadata."""
    gradient = np.asarray(gradient, dtype=np.float32)
    if gradient.ndim != 3:
        raise ValueError(
            "gradient must have shape (n_obs, n_genes, n_modules)"
        )
    flat_gradient = gradient.reshape(gradient.shape[0], -1)
    adata.obsm[GRADIENT_KEY] = flat_gradient
    metadata = _root_metadata(adata)
    metadata["group_key"] = group_key
    metadata["expression_source"] = expression_source
    metadata["model"] = dict(model_metadata)
    metadata["parameters"] = dict(parameters)
    metadata["gradient"] = {
        "obsm_key": GRADIENT_KEY,
        "group_key": group_key,
        "expression_source": expression_source,
        "model": dict(model_metadata),
        "parameters": dict(parameters),
        "gene_names": list(gene_names),
        "module_names": list(module_names),
        "shape": np.asarray(gradient.shape, dtype=np.int64),
        "flatten_order": "C",
        "axis_order": ["cell", "gene", "module"],
        "loss_history": {
            key: np.asarray(value, dtype=np.float64)
            for key, value in loss_history.items()
        },
        "elapsed_seconds": float(elapsed_seconds),
    }


def get_scfeaann_gradient(
    adata: ad.AnnData, cell: str | int | list[str] | None = None
) -> pd.DataFrame | np.ndarray:
    """Restore a stored flattened gradient to gene-by-module structure.

    Parameters
    ----------
    adata
        AnnData containing ``obsm["scFEA_grad"]`` and its ``uns`` metadata.
    cell
        Cell name, integer position, or list of cell names. If omitted, all
        cells are returned. A single cell returns a gene-by-module DataFrame;
        multiple cells return an array shaped
        ``(n_selected_cells, n_genes, n_modules)``.
    """
    if GRADIENT_KEY not in adata.obsm:
        raise ValueError(
            f"adata.obsm['{GRADIENT_KEY}'] is missing; run "
            "calculate_scfeaann_gradient first"
        )
    metadata = adata.uns.get(METADATA_KEY, {})
    gradient_metadata = metadata.get("gradient")
    if not gradient_metadata:
        raise ValueError("gradient metadata is missing from adata.uns")
    gene_names = list(gradient_metadata["gene_names"])
    module_names = list(gradient_metadata["module_names"])
    shape = tuple(int(value) for value in gradient_metadata["shape"])
    flat = np.asarray(adata.obsm[GRADIENT_KEY])
    restored = flat.reshape(shape)

    if cell is None:
        return restored
    if isinstance(cell, int) and not isinstance(cell, bool):
        if cell < 0 or cell >= restored.shape[0]:
            raise ValueError(f"cell index {cell} is out of range")
        return pd.DataFrame(
            restored[cell], index=gene_names, columns=module_names
        )
    if isinstance(cell, str):
        matches = np.flatnonzero(adata.obs_names == cell)
        if len(matches) != 1:
            raise ValueError(f"cell name '{cell}' is not present exactly once")
        return pd.DataFrame(
            restored[int(matches[0])],
            index=gene_names,
            columns=module_names,
        )
    if isinstance(cell, list):
        if not all(isinstance(item, str) for item in cell):
            raise ValueError("cell lists must contain observation names")
        indexer = adata.obs_names.get_indexer(cell)
        if np.any(indexer < 0):
            missing = [
                name for name, position in zip(cell, indexer, strict=True)
                if position < 0
            ]
            raise ValueError(f"cell names are not present: {missing}")
        return restored[indexer]
    raise ValueError(
        "cell must be None, an integer position, a cell name, or a list of names"
    )


def write_supermodule_results(
    adata: ad.AnnData,
    *,
    supermodule_flux: np.ndarray,
    supermodule_balance: np.ndarray,
    flux_sum: np.ndarray,
    flux_mean: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
    module_names: list[str],
    compound_names: list[str],
    class_results: dict[str, dict[str, Any]],
    total_requested: bool,
    total_completed: bool,
) -> None:
    """Write independently trained supermodule model outputs to AnnData."""
    n_obs = adata.n_obs
    n_classes = len(class_ids)
    n_modules = len(module_names)
    n_compounds = len(compound_names)
    if n_classes != len(class_names) or n_classes != 22:
        raise ValueError("exactly 22 class IDs and class names are required")
    if len(set(class_ids)) != n_classes:
        raise ValueError("supermodule class IDs must be unique")
    expected_shapes = {
        SUPERMODULE_FLUX_KEY: (n_obs, n_modules),
        SUPERMODULE_BALANCE_KEY: (n_obs, n_classes * n_compounds),
        SUPERMODULE_FLUX_SUM_KEY: (n_obs, n_classes),
        SUPERMODULE_FLUX_MEAN_KEY: (n_obs, n_classes),
    }
    arrays = {
        SUPERMODULE_FLUX_KEY: supermodule_flux,
        SUPERMODULE_BALANCE_KEY: supermodule_balance,
        SUPERMODULE_FLUX_SUM_KEY: flux_sum,
        SUPERMODULE_FLUX_MEAN_KEY: flux_mean,
    }
    for key, expected in expected_shapes.items():
        value = np.asarray(arrays[key])
        if value.shape != expected:
            raise ValueError(
                f"{key} has shape {value.shape}, expected {expected}"
            )
        adata.obsm[key] = np.asarray(value, dtype=np.float32)

    metadata = _root_metadata(adata)
    metadata["supermodules"] = {
        "class_ids": np.asarray(class_ids, dtype=np.int64),
        "class_names": list(class_names),
        "module_names": list(module_names),
        "compound_names": [str(name) for name in compound_names],
        "n_classes": n_classes,
        "n_modules": n_modules,
        "n_compounds": n_compounds,
        "balance_axis_order": ["supermodule_class", "compound"],
        "balance_flatten_order": "C",
        "class_results": class_results,
        "total_model": {
            "requested": bool(total_requested),
            "completed": bool(total_completed),
            "flux_key": FLUX_KEY,
            "balance_key": BALANCE_KEY,
        },
    }


def _supermodule_metadata(adata: ad.AnnData) -> dict[str, Any]:
    metadata = adata.uns.get(METADATA_KEY, {})
    supermodule_metadata = (
        metadata.get("supermodules", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    if not supermodule_metadata:
        raise ValueError("supermodule metadata is missing from adata.uns")
    return supermodule_metadata


def _supermodule_class_index(
    metadata: dict[str, Any],
    class_id: int | str,
) -> int:
    class_ids = [int(value) for value in metadata["class_ids"]]
    try:
        normalized = int(class_id)
    except (TypeError, ValueError) as error:
        raise ValueError("class_id must be a Supermodule integer ID") from error
    if normalized not in class_ids:
        raise ValueError(f"supermodule class {normalized} is not available")
    return class_ids.index(normalized)


def get_scfeaann_supermodule_flux(
    adata: ad.AnnData,
    class_id: int | str | None = None,
) -> pd.DataFrame:
    """Return all class-specific module flux or one class as a DataFrame."""
    if SUPERMODULE_FLUX_KEY not in adata.obsm:
        raise ValueError(f"adata.obsm['{SUPERMODULE_FLUX_KEY}'] is missing")
    metadata = _supermodule_metadata(adata)
    flux = pd.DataFrame(
        np.asarray(adata.obsm[SUPERMODULE_FLUX_KEY]),
        index=adata.obs_names,
        columns=[str(name) for name in metadata["module_names"]],
    )
    if class_id is None:
        return flux
    class_index = _supermodule_class_index(metadata, class_id)
    key = str(int(metadata["class_ids"][class_index]))
    result = metadata["class_results"][key]
    if result.get("status") != "success":
        raise ValueError(f"supermodule class {key} did not complete successfully")
    modules = [str(name) for name in result.get("module_names", [])]
    return flux.loc[:, modules]


def get_scfeaann_supermodule_balance(
    adata: ad.AnnData,
    class_id: int | str | None = None,
) -> np.ndarray | pd.DataFrame:
    """Restore class-specific balance to class-by-compound structure."""
    if SUPERMODULE_BALANCE_KEY not in adata.obsm:
        raise ValueError(f"adata.obsm['{SUPERMODULE_BALANCE_KEY}'] is missing")
    metadata = _supermodule_metadata(adata)
    balance = np.asarray(adata.obsm[SUPERMODULE_BALANCE_KEY]).reshape(
        adata.n_obs,
        int(metadata["n_classes"]),
        int(metadata["n_compounds"]),
    )
    if class_id is None:
        return balance
    class_index = _supermodule_class_index(metadata, class_id)
    return pd.DataFrame(
        balance[:, class_index, :],
        index=adata.obs_names,
        columns=[str(name) for name in metadata["compound_names"]],
    )
