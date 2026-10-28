"""Public AnnData APIs for scFEA flux and gradient analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from .core.training import (
    DEFAULT_SEED,
    LAMB_BA,
    LAMB_CELL,
    LAMB_MOD,
    LAMB_NG,
    LEARN_RATE,
    compute_scfea_gradient,
    fit_scfea,
    predict_scfea,
    resolve_device,
)
from .preprocessing import prepare_scfea_inputs
from .modules import get_module_annotation
from .results import (
    validate_group_key,
    write_flux_results,
    write_gradient_results,
    write_supermodule_results,
)
from .supermodules import (
    class_module_names,
    load_supermodules,
    resolve_supermodule_classes,
    subset_prepared_inputs,
)


def _validate_prediction_batch_size(value: int) -> None:
    if value <= 0:
        raise ValueError("prediction_batch_size must be greater than zero")


def _common_parameters(
    *,
    sc_imputation: bool,
    epochs: int,
    device: str,
    prediction_batch_size: int | None = None,
    module_gene_file: str | Path | None,
    stoichiometry_matrix: str | Path | None,
    compound_name_file: str | Path | None,
) -> dict[str, Any]:
    return {
        "sc_imputation": bool(sc_imputation),
        "epochs": int(epochs),
        "device": device,
        "random_seed": DEFAULT_SEED,
        "learning_rate": LEARN_RATE,
        "lambdas": {
            "balance": LAMB_BA,
            "nonnegative": LAMB_NG,
            "cell": LAMB_CELL,
            "module": LAMB_MOD,
        },
        "module_gene_file": (
            None if module_gene_file is None else str(module_gene_file)
        ),
        "stoichiometry_matrix": (
            None if stoichiometry_matrix is None else str(stoichiometry_matrix)
        ),
        "compound_name_file": (
            None if compound_name_file is None else str(compound_name_file)
        ),
        **(
            {}
            if prediction_batch_size is None
            else {"prediction_batch_size": int(prediction_batch_size)}
        ),
    }


def _fit_predict_prepared(
    prepared,
    *,
    epochs: int,
    device: str,
    verbose: bool,
    lamb_module: float = LAMB_MOD,
    safe_module_correlation: bool = False,
    prediction_batch_size: int = 1024,
):
    n_comps = prepared.stoichiometry.shape[0]
    fitted = fit_scfea(
        prepared.X,
        prepared.gene_scale,
        prepared.module_scale,
        prepared.stoichiometry,
        n_modules=len(prepared.module_names),
        n_genes=len(prepared.gene_names),
        n_comps=n_comps,
        epochs=epochs,
        device=device,
        lamb_module=lamb_module,
        safe_module_correlation=safe_module_correlation,
        verbose=verbose,
    )
    flux, balance = predict_scfea(
        fitted.net,
        prepared.X,
        prepared.stoichiometry,
        n_modules=len(prepared.module_names),
        n_genes=len(prepared.gene_names),
        n_comps=n_comps,
        device=device,
        prediction_batch_size=prediction_batch_size,
    )
    return fitted, flux, balance


def run_scfeaann(
    adata: ad.AnnData,
    *,
    group_key: str | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    sc_imputation: bool = False,
    epochs: int = 100,
    device: str = "auto",
    data_dir: str | Path | None = None,
    module_gene_file: str | Path | None = None,
    stoichiometry_matrix: str | Path | None = None,
    compound_name_file: str | Path | None = None,
    copy: bool = False,
    verbose: bool = True,
    prediction_batch_size: int = 1024,
) -> ad.AnnData:
    """Estimate single-cell metabolic flux and compound balance from AnnData.

    Parameters
    ----------
    adata
        Single-cell AnnData with observations as cells and variables as genes.
    group_key
        Optional column in ``adata.obs``. It is validated, preserved, and
        recorded in ``uns["scFEAann"]``. It is not used to split training.
    layer
        Optional key in ``adata.layers``. By default ``adata.X`` is used.
        Cannot be combined with ``use_raw=True``.
    use_raw
        Use ``adata.raw.X`` and ``adata.raw.var_names`` when true.
    sc_imputation
        Apply MAGIC to the complete selected expression matrix before the
        original log and scale processing.
    epochs
        Number of training epochs. The original default is 100.
    device
        Torch device string, ``"auto"``, ``"cpu"``, or an explicit device such
        as ``"cuda:0"``.
    data_dir
        Directory containing model CSV files. Defaults to packaged scFEA data.
    module_gene_file
        Module-to-gene CSV filename. Defaults to human M168.
    stoichiometry_matrix
        Compound-to-module CSV filename. Defaults to human C70/M168.
    compound_name_file
        Compound-name CSV filename. Defaults to human C70/M168.
    copy
        If false, update and return ``adata``. If true, update and return a
        copy, leaving the input unchanged.
    verbose
        Show the epoch progress bar.
    prediction_batch_size
        Number of cells evaluated per inference chunk. The default is 1024.
        This changes neither the network, loss, optimizer, training batch, nor
        random seed. It is recorded in ``uns["scFEAann"]["parameters"]``.

    Returns
    -------
    anndata.AnnData
        The input or copied object with ``obsm["scFEA_flux"]``,
        ``obsm["scFEA_balance"]``, and ``uns["scFEAann"]``.

    Notes
    -----
    All cells are trained in one global batch, matching the supplied scFEA
    implementation. The group column is metadata only.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    _validate_prediction_batch_size(prediction_batch_size)
    validate_group_key(adata, group_key)
    target = adata.copy() if copy else adata

    prepared = prepare_scfea_inputs(
        target,
        layer=layer,
        use_raw=use_raw,
        sc_imputation=sc_imputation,
        data_dir=data_dir,
        module_gene_file=module_gene_file,
        stoichiometry_matrix=stoichiometry_matrix,
        compound_name_file=compound_name_file,
    )
    fitted, flux, balance = _fit_predict_prepared(
        prepared,
        epochs=epochs,
        device=device,
        verbose=verbose,
        prediction_batch_size=prediction_batch_size,
    )
    write_flux_results(
        target,
        flux,
        balance,
        module_names=prepared.module_names,
        compound_names=prepared.compound_names,
        group_key=group_key,
        expression_source=prepared.expression_source,
        gene_names=prepared.gene_names,
        model_metadata=prepared.model_metadata,
        parameters=_common_parameters(
            sc_imputation=sc_imputation,
            epochs=epochs,
            device=str(resolve_device(device)),
            prediction_batch_size=prediction_batch_size,
            module_gene_file=module_gene_file,
            stoichiometry_matrix=stoichiometry_matrix,
            compound_name_file=compound_name_file,
        ),
        loss_history=fitted.loss_history,
        elapsed_seconds=fitted.elapsed_seconds,
    )
    return target


def run_scfeaann_supermodules(
    adata: ad.AnnData,
    *,
    group_key: str | None = None,
    classes: int | str | list[int | str] | tuple[int | str, ...] | None = None,
    include_total: bool = True,
    layer: str | None = None,
    use_raw: bool = False,
    sc_imputation: bool = False,
    epochs: int = 100,
    device: str = "auto",
    data_dir: str | Path | None = None,
    module_gene_file: str | Path | None = None,
    stoichiometry_matrix: str | Path | None = None,
    compound_name_file: str | Path | None = None,
    supermodule_file: str | Path | None = None,
    copy: bool = False,
    verbose: bool = True,
    continue_on_error: bool = True,
    prediction_batch_size: int = 1024,
) -> ad.AnnData:
    """Train independent scFEA models for selected human supermodules.

    The global 168-module model is optionally trained first. Each requested
    supermodule class then gets its own FLUX model using its assigned module
    rows and stoichiometry columns while retaining all compound rows.

    Parameters
    ----------
    adata
        Single-cell AnnData with observations as cells and variables as genes.
    classes
        Supermodule IDs, class names, or a sequence of either. ``None`` selects
        all 22 classes.
    include_total
        Train the global 168-module model before class-specific models. When
        false, no full module tensor is constructed.
    prediction_batch_size
        Number of cells evaluated per inference chunk. The default is 1024. It
        does not change training behavior and is recorded in each requested
        class result.
    Other parameters
        Match :func:`run_scfeaann`.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    _validate_prediction_batch_size(prediction_batch_size)
    validate_group_key(adata, group_key)
    supermodule_table = load_supermodules(
        data_dir=data_dir,
        supermodule_file=supermodule_file,
    )
    class_ids = resolve_supermodule_classes(
        classes, table=supermodule_table
    )
    target = adata.copy() if copy else adata
    prepared = prepare_scfea_inputs(
        target,
        layer=layer,
        use_raw=use_raw,
        sc_imputation=sc_imputation,
        data_dir=data_dir,
        module_gene_file=module_gene_file,
        stoichiometry_matrix=stoichiometry_matrix,
        compound_name_file=compound_name_file,
        build_module_blocks=include_total,
    )
    model_file = Path(
        prepared.model_metadata["module_gene_file"]
    ).name
    if model_file != "module_gene_m168.csv":
        raise ValueError(
            "supermodule-specific modeling currently requires the human "
            "module_gene_m168.csv model"
        )
    annotation = get_module_annotation()
    if not set(prepared.module_names).issubset(annotation.index):
        raise ValueError("prepared modules are not covered by human annotation")

    total_requested = bool(include_total)
    total_completed = False
    if total_requested:
        fitted, flux, balance = _fit_predict_prepared(
            prepared,
            epochs=epochs,
            device=str(resolve_device(device)),
            verbose=verbose,
            prediction_batch_size=prediction_batch_size,
        )
        write_flux_results(
            target,
            flux,
            balance,
            module_names=prepared.module_names,
            compound_names=prepared.compound_names,
            group_key=group_key,
            expression_source=prepared.expression_source,
            gene_names=prepared.gene_names,
            model_metadata=prepared.model_metadata,
            parameters=_common_parameters(
                sc_imputation=sc_imputation,
                epochs=epochs,
                device=str(resolve_device(device)),
                prediction_batch_size=prediction_batch_size,
                module_gene_file=module_gene_file,
                stoichiometry_matrix=stoichiometry_matrix,
                compound_name_file=compound_name_file,
            ),
            loss_history=fitted.loss_history,
            elapsed_seconds=fitted.elapsed_seconds,
        )
        total_completed = True
        prepared.X = None
        del fitted, flux, balance

    all_class_ids = supermodule_table.index.astype(int).tolist()
    all_class_names = [
        str(supermodule_table.loc[class_id, "Supermodule class"])
        for class_id in all_class_ids
    ]
    class_slot = {
        class_id: index for index, class_id in enumerate(all_class_ids)
    }
    module_position = {
        name: index for index, name in enumerate(prepared.module_names)
    }
    n_compounds = len(prepared.compound_names)
    supermodule_flux = np.full(
        (target.n_obs, len(prepared.module_names)),
        np.nan,
        dtype=np.float32,
    )
    supermodule_balance = np.full(
        (target.n_obs, len(all_class_ids) * n_compounds),
        np.nan,
        dtype=np.float32,
    )
    flux_sum = np.full(
        (target.n_obs, len(all_class_ids)), np.nan, dtype=np.float32
    )
    flux_mean = np.full(
        (target.n_obs, len(all_class_ids)), np.nan, dtype=np.float32
    )
    class_results: dict[str, dict[str, Any]] = {}
    successful = 0

    for class_id in all_class_ids:
        module_names = class_module_names(
            class_id, module_annotation=annotation
        )
        result: dict[str, Any] = {
            "supermodule_id": int(class_id),
            "supermodule_class": str(
                supermodule_table.loc[class_id, "Supermodule class"]
            ),
            "module_names": module_names,
            "module_indices": [
                module_position[name]
                for name in module_names
                if name in module_position
            ],
            "n_modules": len(module_names),
            "n_genes": len(prepared.gene_names),
            "n_compounds": n_compounds,
            "status": "not_requested",
            "error": None,
            "loss_history": {},
            "elapsed_seconds": None,
            "parameters": _common_parameters(
                sc_imputation=sc_imputation,
                epochs=epochs,
                device=str(resolve_device(device)),
                prediction_batch_size=prediction_batch_size,
                module_gene_file=module_gene_file,
                stoichiometry_matrix=stoichiometry_matrix,
                compound_name_file=compound_name_file,
            ),
        }
        class_results[str(class_id)] = result
        if class_id not in class_ids:
            continue
        try:
            subset = subset_prepared_inputs(prepared, module_names)
            fitted, class_flux, class_balance = _fit_predict_prepared(
                subset,
                epochs=epochs,
                device=str(resolve_device(device)),
                verbose=verbose,
                lamb_module=0.0 if len(subset.module_names) < 2 else LAMB_MOD,
                safe_module_correlation=True,
                prediction_batch_size=prediction_batch_size,
            )
            if not np.isfinite(class_flux).all() or not np.isfinite(
                class_balance
            ).all():
                raise RuntimeError(
                    "supermodule model produced non-finite flux or balance"
                )
            positions = [module_position[name] for name in module_names]
            supermodule_flux[:, positions] = class_flux
            slot = class_slot[class_id]
            start = slot * n_compounds
            stop = start + n_compounds
            supermodule_balance[:, start:stop] = class_balance
            flux_sum[:, slot] = class_flux.sum(axis=1)
            flux_mean[:, slot] = class_flux.mean(axis=1)
            result.update(
                {
                    "status": "success",
                    "loss_history": fitted.loss_history,
                    "elapsed_seconds": float(fitted.elapsed_seconds),
                }
            )
            result["parameters"]["lambdas"]["module"] = (
                0.0 if len(subset.module_names) < 2 else LAMB_MOD
            )
            result["module_loss_disabled"] = len(subset.module_names) < 2
            undefined_correlation_cells = int(
                (
                    subset.module_scale.var(axis=1, unbiased=False)
                    == 0
                ).sum().item()
            )
            result["module_correlation_fallback"] = (
                "skip_undefined" if undefined_correlation_cells else None
            )
            result["undefined_module_correlation_cells"] = (
                undefined_correlation_cells
            )
            successful += 1
        except Exception as error:
            result.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if not continue_on_error:
                raise

    if successful == 0:
        raise RuntimeError("no supermodule-specific model completed successfully")
    write_supermodule_results(
        target,
        supermodule_flux=supermodule_flux,
        supermodule_balance=supermodule_balance,
        flux_sum=flux_sum,
        flux_mean=flux_mean,
        class_ids=all_class_ids,
        class_names=all_class_names,
        module_names=prepared.module_names,
        compound_names=[str(name) for name in prepared.compound_names],
        class_results=class_results,
        total_requested=total_requested,
        total_completed=total_completed,
    )
    return target


def calculate_scfeaann_gradient(
    adata: ad.AnnData,
    *,
    group_key: str | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    sc_imputation: bool = False,
    epochs: int = 100,
    device: str = "auto",
    data_dir: str | Path | None = None,
    module_gene_file: str | Path | None = None,
    stoichiometry_matrix: str | Path | None = None,
    compound_name_file: str | Path | None = None,
    copy: bool = False,
    verbose: bool = True,
) -> ad.AnnData:
    """Calculate scFEA gene-to-module gradients for every cell.

    This is an independent API corresponding to the original
    ``scFEA_grad.py`` workflow. It performs its own training and does not
    require a prior ``run_scfeaann`` call.

    Parameters
    ----------
    adata
        Single-cell AnnData with observations as cells and variables as genes.
    group_key
        Optional ``obs`` column that is validated and preserved. It is not used
        to split training.
    layer
        Optional expression layer. The default source is ``adata.X``.
    use_raw
        Use ``adata.raw.X`` when true. Cannot be combined with ``layer``.
    sc_imputation
        Apply the original MAGIC imputation path before log and scaling.
    epochs
        Number of training epochs before per-cell gradient refitting.
    device
        Torch device string, ``"auto"``, ``"cpu"``, or an explicit device.
    data_dir
        Directory containing model CSV files.
    module_gene_file
        Optional module-to-gene CSV filename.
    stoichiometry_matrix
        Optional compound-to-module CSV filename.
    compound_name_file
        Optional compound-name CSV filename.
    copy
        If false, update and return ``adata``. If true, update and return a
        copy.
    verbose
        Show training and per-cell progress bars.

    Returns
    -------
    anndata.AnnData
        Object with ``obsm["scFEA_grad"]`` shaped
        ``(n_obs, n_gradient_genes * n_modules)`` and metadata under
        ``uns["scFEAann"]["gradient"]``. Use
        :func:`get_scfeaann_gradient` to restore gene-by-module structure.

    Notes
    -----
    Gradient entries for module genes with zero expression follow the original
    scFEA normalization and can therefore be NaN. Array flattening is C-order
    over ``(cells, genes, modules)``.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    validate_group_key(adata, group_key)
    target = adata.copy() if copy else adata

    prepared = prepare_scfea_inputs(
        target,
        layer=layer,
        use_raw=use_raw,
        sc_imputation=sc_imputation,
        data_dir=data_dir,
        module_gene_file=module_gene_file,
        stoichiometry_matrix=stoichiometry_matrix,
        compound_name_file=compound_name_file,
    )
    fitted = fit_scfea(
        prepared.X,
        prepared.gene_scale,
        prepared.module_scale,
        prepared.stoichiometry,
        n_modules=len(prepared.module_names),
        n_genes=len(prepared.gene_names),
        n_comps=prepared.stoichiometry.shape[0],
        epochs=epochs,
        device=device,
        verbose=verbose,
    )
    gradient = compute_scfea_gradient(
        fitted.net,
        fitted.optimizer,
        prepared.X,
        prepared.gene_scale,
        prepared.module_scale,
        prepared.gene_expression,
        prepared.module_masks,
        prepared.stoichiometry,
        n_modules=len(prepared.module_names),
        n_genes=len(prepared.gene_names),
        n_comps=prepared.stoichiometry.shape[0],
        device=device,
        verbose=verbose,
    )
    write_gradient_results(
        target,
        gradient,
        module_names=prepared.module_names,
        gene_names=prepared.gene_names,
        group_key=group_key,
        expression_source=prepared.expression_source,
        model_metadata=prepared.model_metadata,
        parameters=_common_parameters(
            sc_imputation=sc_imputation,
            epochs=epochs,
            device=str(resolve_device(device)),
            module_gene_file=module_gene_file,
            stoichiometry_matrix=stoichiometry_matrix,
            compound_name_file=compound_name_file,
        ),
        loss_history=fitted.loss_history,
        elapsed_seconds=fitted.elapsed_seconds,
    )
    return target
