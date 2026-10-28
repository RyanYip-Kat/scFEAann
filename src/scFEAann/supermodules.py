"""Human supermodule definitions and model-slice helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch

from .modules import get_module_annotation
from .preprocessing import (
    PACKAGE_DATA_DIR,
    PreparedInputs,
    materialize_module_blocks,
)


DEFAULT_SUPERMODULE_FILE = "supermetabolic_modules.csv"


def _resolve_path(
    data_dir: str | Path | None,
    filename: str | Path,
) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    base = PACKAGE_DATA_DIR if data_dir is None else Path(data_dir)
    return base / path


def load_supermodules(
    *,
    data_dir: str | Path | None = None,
    supermodule_file: str | Path | None = None,
) -> pd.DataFrame:
    """Load the 22 human supermodule definitions, indexed by supermodule ID."""
    path = _resolve_path(
        data_dir, supermodule_file or DEFAULT_SUPERMODULE_FILE
    )
    if not path.exists():
        raise ValueError(f"supermodule file does not exist: {path}")
    table = pd.read_csv(path)
    required = {
        "Supermodule ID",
        "Supermodule class",
        "Number of modules",
        "Number of genes",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"supermodule file is missing columns: {sorted(missing)}")
    table["Supermodule ID"] = table["Supermodule ID"].astype(int)
    if not table["Supermodule ID"].is_unique:
        raise ValueError("supermodule IDs must be unique")
    return table.set_index("Supermodule ID").sort_index()


def _normalize_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_supermodule_classes(
    classes: int | str | list[int | str] | tuple[int | str, ...] | None,
    *,
    table: pd.DataFrame | None = None,
) -> list[int]:
    """Resolve IDs or class names to a sorted, unique list of supermodule IDs."""
    table = load_supermodules() if table is None else table
    if classes is None:
        return table.index.astype(int).tolist()
    values = (
        list(classes)
        if isinstance(classes, (list, tuple))
        else [classes]
    )
    available_ids = set(table.index.astype(int))
    normalized = {
        _normalize_class_name(str(name)): int(supermodule_id)
        for supermodule_id, name in table["Supermodule class"].items()
    }
    resolved: set[int] = set()
    for value in values:
        if isinstance(value, int) or (
            isinstance(value, str) and value.strip().isdigit()
        ):
            supermodule_id = int(value)
            if supermodule_id not in available_ids:
                raise ValueError(
                    f"unknown supermodule ID {supermodule_id}; available IDs: "
                    f"{sorted(available_ids)}"
                )
            resolved.add(supermodule_id)
            continue
        if not isinstance(value, str):
            raise TypeError("supermodule selectors must be IDs or class names")
        key = _normalize_class_name(value)
        if key not in normalized:
            raise ValueError(
                f"unknown supermodule class '{value}'; available classes: "
                f"{table['Supermodule class'].tolist()}"
            )
        resolved.add(normalized[key])
    return sorted(resolved)


def class_module_names(
    supermodule_id: int,
    *,
    module_annotation: pd.DataFrame | None = None,
) -> list[str]:
    """Return module IDs assigned to one supermodule in model order."""
    annotation = (
        get_module_annotation()
        if module_annotation is None
        else module_annotation
    )
    if supermodule_id not in set(annotation["supermodule_id"]):
        raise ValueError(f"unknown supermodule ID {supermodule_id}")
    return annotation.index[
        annotation["supermodule_id"] == supermodule_id
    ].astype(str).tolist()


def subset_prepared_inputs(
    prepared: PreparedInputs,
    module_names: list[str],
) -> PreparedInputs:
    """Select an aligned module subset from one prepared AnnData tensor."""
    if prepared.X is None:
        return materialize_module_blocks(prepared, module_names)
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
    indices = torch.tensor(
        [module_position[name] for name in module_names],
        dtype=torch.long,
    )
    n_modules = len(prepared.module_names)
    n_genes = len(prepared.gene_names)
    if prepared.X.shape[1] != n_modules * n_genes:
        raise ValueError("prepared X width does not match module and gene counts")
    expression_by_module = prepared.X.reshape(
        prepared.X.shape[0], n_modules, n_genes
    )
    selected_x = expression_by_module.index_select(1, indices).reshape(
        prepared.X.shape[0], len(module_names) * n_genes
    )
    return PreparedInputs(
        X=selected_x,
        gene_expression=prepared.gene_expression,
        module_masks=prepared.module_masks.index_select(0, indices),
        gene_scale=prepared.gene_scale,
        module_scale=prepared.module_scale.index_select(1, indices),
        stoichiometry=prepared.stoichiometry.index_select(1, indices),
        gene_names=list(prepared.gene_names),
        cell_names=list(prepared.cell_names),
        module_names=list(module_names),
        compound_names=list(prepared.compound_names),
        expression_source=prepared.expression_source,
        model_metadata=dict(prepared.model_metadata),
    )
