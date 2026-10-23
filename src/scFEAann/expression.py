"""Select and validate an expression matrix from an AnnData object."""

from __future__ import annotations

from typing import Any

import anndata as ad
import pandas as pd


def resolve_expression(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    use_raw: bool = False,
) -> tuple[Any, pd.Index, str]:
    """Return the requested expression matrix, gene names, and source label.

    Parameters
    ----------
    adata
        AnnData whose observations are cells and variables are genes.
    layer
        Optional key in ``adata.layers``. Cannot be combined with ``use_raw``.
    use_raw
        If true, use ``adata.raw.X`` and ``adata.raw.var_names``.

    Returns
    -------
    tuple
        ``(matrix, var_names, source)`` where matrix has shape
        ``(n_obs, n_vars)``. The returned matrix is a view or reference to the
        selected AnnData storage and is not modified.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if layer is not None and use_raw:
        raise ValueError("layer and use_raw=True are mutually exclusive")
    if not pd.Index(adata.obs_names).is_unique:
        raise ValueError("adata.obs_names must be unique")

    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True requires adata.raw")
        matrix = adata.raw.X
        var_names = pd.Index(adata.raw.var_names)
        source = "raw"
    elif layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"layer '{layer}' is not present in adata.layers")
        matrix = adata.layers[layer]
        var_names = pd.Index(adata.var_names)
        source = f"layer:{layer}"
    else:
        matrix = adata.X
        var_names = pd.Index(adata.var_names)
        source = "X"

    if matrix is None:
        raise ValueError(f"expression source '{source}' is empty")
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError(f"expression source '{source}' must be two-dimensional")
    if matrix.shape != (adata.n_obs, len(var_names)):
        raise ValueError(
            f"expression source '{source}' has shape {matrix.shape}, expected "
            f"{(adata.n_obs, len(var_names))}"
        )
    if not var_names.is_unique:
        raise ValueError(f"var_names for expression source '{source}' must be unique")

    return matrix, var_names, source
