# Supermodule-Specific scFEA Models

## Goal

Add a first-class API that trains one original scFEA model for each selected
human supermodule class, while also retaining the existing global 168-module
model. All outputs will be stored on AnnData and persist through H5AD.

## Important semantics

- A per-class model is a separately trained submodel, not a slice of the global
  result.
- All cells participate in every class model.
- The global model remains the existing 168-module `run_scfeaann` workflow.
- Per-class models use the same network structure, loss terms, learning rate,
  seed, optimizer, and preprocessing as scFEA.
- Version one supports flux, balance, loss history, and node-level summaries.
  It does not calculate per-class gradients.

## API

```python
run_scfeaann_supermodules(
    adata,
    *,
    group_key=None,
    classes=None,
    include_total=True,
    layer=None,
    use_raw=False,
    sc_imputation=False,
    epochs=100,
    device="auto",
    data_dir=None,
    supermodule_file=None,
    copy=False,
    verbose=True,
    continue_on_error=True,
)
```

`classes=None` selects all 22 supermodules. A class selector can be a
Supermodule ID or a class name.

## Model construction

For each class:

1. Find its module IDs from `Human_M168_information.symbols.csv`.
2. Select the matching module-gene rows.
3. Select the same columns from the 70-by-168 stoichiometry matrix.
4. Keep all 70 compounds on the balance axis.
5. Reuse global gene-scale values and select only class module-scale values.
6. Train a new FLUX model with `n_modules` equal to the class module count.
7. Predict flux and balance for every cell.

## AnnData output

Global results remain:

```python
obsm["scFEA_flux"]       # n_obs x 168
obsm["scFEA_balance"]    # n_obs x 70
```

Per-class module flux is assembled back into original module order:

```python
obsm["scFEA_supermodule_flux"]  # n_obs x 168
```

Per-class balance is stored in fixed 22-class slots:

```python
obsm["scFEA_supermodule_balance"]  # n_obs x (22 * 70)
```

Flattening is C-order with class as the outer axis and compound as the inner
axis. Unselected or failed class slots contain `NaN`.

Network-ready node features:

```python
obsm["scFEA_supermodule_flux_sum"]   # n_obs x 22
obsm["scFEA_supermodule_flux_mean"]  # n_obs x 22
```

Metadata under `uns["scFEAann"]["supermodules"]` records:

- Supermodule IDs and class names.
- Global module and compound axes.
- Class-to-module indices.
- Balance flattening order.
- Per-class status, error, elapsed time, loss history, and model parameters.
- Whether the total run was requested and whether it completed.

## Error handling

- Invalid class selectors raise `ValueError`.
- Non-human or incompatible models raise `ValueError`.
- With `continue_on_error=True`, one failed class does not abort the remaining
  classes; its status is recorded and its output slots remain `NaN`.
- With `continue_on_error=False`, the first class error is raised.
- At least one successful class is required.

## Verification

- Unit tests verify class-to-module selection and tensor slicing.
- Synthetic integration tests train selected classes and the global model for
  one epoch.
- Tests verify all obsm shapes, NaN handling, metadata, copy semantics, and
  H5AD round-trip.
- A smoke run uses a small test-data subset before any expensive full run.
