# scFEAann AnnData API Design

## Goal

Package the existing scFEA algorithms as `scFEAann`, a Python package whose
public API accepts AnnData and writes results back to the same AnnData object.
The mathematical network, losses, optimization steps, and gradient definition
must remain those of the supplied scFEA source.

## Non-goals

- Do not reinterpret scFEA as a group-wise model.
- Do not change the network architecture, loss terms, learning rate, random
  seed, or default training epochs.
- Do not replace AnnData's sparse matrix with a new public storage format.
- Do not make `calculate_scfeaann_gradient` run automatically after flux
  inference.

## Public API

```python
run_scfeaann(
    adata,
    *,
    group_key=None,
    layer=None,
    use_raw=False,
    sc_imputation=False,
    epochs=100,
    device="auto",
    data_dir=None,
    module_gene_file=None,
    stoichiometry_matrix=None,
    compound_name_file=None,
    copy=False,
    verbose=True,
)

calculate_scfeaann_gradient(
    adata,
    *,
    group_key=None,
    layer=None,
    use_raw=False,
    sc_imputation=False,
    epochs=100,
    device="auto",
    data_dir=None,
    module_gene_file=None,
    stoichiometry_matrix=None,
    compound_name_file=None,
    copy=False,
    verbose=True,
)

get_scfeaann_gradient(adata, cell=None)
```

`run_scfeaann` writes flux and balance. `calculate_scfeaann_gradient` performs
its own training and writes the gene-to-module gradient. Both default to
in-place modification and return the modified AnnData. With `copy=True`, the
input object is unchanged.

## Expression source

- The default source is `adata.X`.
- `layer="name"` selects `adata.layers["name"]`.
- `use_raw=True` selects `adata.raw.X` and `adata.raw.var_names`.
- `layer` and `use_raw=True` are mutually exclusive.
- Every source must be shaped `n_obs x n_vars`.
- `obs_names` and `var_names` must be unique.
- The adapter must not subset or otherwise modify the input AnnData.

## Group handling

`group_key` is optional. If supplied, it must name an `adata.obs` column. Its
values are preserved unchanged. The group is metadata only: all observed cells
are used in one training batch. The selected key is recorded in
`adata.uns["scFEAann"]`.

## Numerical compatibility

The adapter reproduces the supplied scFEA pipeline:

1. Read an expression matrix in cell-by-gene orientation.
2. Optionally apply MAGIC to the complete expression source.
3. Apply `log2(x + 1)` when the maximum value exceeds 50.
4. Compute the gene-scale target from all source genes.
5. Select overlap genes and sort them lexicographically.
6. Construct one masked gene block per metabolic module.
7. Compute the module-scale target using the original module gene counts.
8. Train all cells as one batch with seed 16 and learning rate 0.008.

Device placement may use tensors on the selected torch device, but it must not
change the numerical formulas.

## Result storage

Flux:

```python
adata.obsm["scFEA_flux"]     # (n_obs, n_modules), float32
```

Balance:

```python
adata.obsm["scFEA_balance"]  # (n_obs, n_compounds), float32
```

Gradient:

```python
adata.obsm["scFEA_grad"]     # (n_obs, n_genes * n_modules), float32
```

The gradient uses C-order flattening of `(n_cells, n_genes, n_modules)`:
gene is the outer varying axis and module is the inner varying axis.

All axis labels and run metadata live under:

```python
adata.uns["scFEAann"]
```

The metadata includes expression source, group key, model filenames, overlap
genes, module names, compound names, gradient genes, original gradient shape,
flattening convention, public parameters, and loss history. Starting one API
must not erase results produced by the other.

`get_scfeaann_gradient(adata, cell=None)` returns:

- a `gene x module` DataFrame for one cell;
- an array shaped `n_selected_cells x n_genes x n_modules` for several cells
  or all cells.

AnnData `write_h5ad()` must preserve all results.

## Internal architecture

```text
src/scFEAann/
├── __init__.py
├── api.py
├── expression.py
├── preprocessing.py
├── results.py
├── core/
│   ├── dataset.py
│   ├── flux.py
│   ├── loss.py
│   └── training.py
└── data/
```

`core` and `preprocessing` do not depend on AnnData. AnnData access is limited
to `expression.py`, `results.py`, and the public API.

## Errors

Raise `ValueError` with actionable messages for:

- mutually exclusive `layer` and `use_raw`;
- absent raw data, layer, or group column;
- non-2D matrices;
- duplicate observation or variable names;
- missing required model files;
- non-finite selected expression values;
- no overlap genes;
- mismatched stoichiometry dimensions;
- invalid gradient reconstruction requests.

Raise `ImportError` with installation guidance when MAGIC is requested but
`magic-impute` is unavailable.

## Verification

- Unit tests cover source selection and every validation error.
- Adapter tests compare prepared tensors with a CSV reference path.
- End-to-end tests exercise both public APIs with a very small AnnData.
- Tests verify copy semantics, group preservation, metadata, and `.h5ad`
  round-trip.
- A parity test compares flux, balance, and gradients produced from equivalent
  CSV and AnnData data under CPU with a fixed seed.
- The package is installed and tested inside a `.venv` created by `uv venv`.
