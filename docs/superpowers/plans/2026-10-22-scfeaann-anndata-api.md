# scFEAann AnnData API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scFEAann`, a uv-managed package that runs the unchanged scFEA
flux and gradient algorithms directly on AnnData objects.

**Architecture:** Copy the scFEA core math into an AnnData-independent core
package, reproduce the original preprocessing as pure functions, and keep
matrix-source selection plus result storage in thin adapter modules.

**Tech Stack:** Python 3.12, uv, anndata, NumPy, pandas, SciPy, torch, pytest,
and optional magic-impute.

**Spec:** `docs/superpowers/specs/2026-10-22-scfeaann-anndata-api-design.md`

## Global Constraints

- The public package and import name is `scFEAann`.
- The default expression source is `adata.X`.
- `layer` and `use_raw=True` are mutually exclusive.
- All cells participate in one training batch; groups are metadata only.
- `copy=False` updates and returns the input AnnData; `copy=True` leaves it
  unchanged.
- Flux is stored at `obsm["scFEA_flux"]`.
- Balance is stored at `obsm["scFEA_balance"]`.
- Gradient is stored at `obsm["scFEA_grad"]` using C-order flattening of
  `(cells, genes, modules)`.
- Metadata is stored at `uns["scFEAann"]`.
- Network architecture, loss formulas, optimizer, learning rate 0.008, seed 16,
  and default epochs 100 remain unchanged.

---

### Task 1: Package scaffold and expression-source adapter

**Files:**
- Create: `pyproject.toml`
- Create: `src/scFEAann/__init__.py`
- Create: `src/scFEAann/expression.py`
- Test: `tests/test_expression.py`

**Interfaces:**
- Produces: `resolve_expression(adata, layer=None, use_raw=False) -> tuple[matrix, pd.Index, str]`

- [ ] **Step 1: Write failing source-selection tests**

```python
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scFEAann.expression import resolve_expression


def make_adata():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    adata.layers["counts"] = X * 10
    adata.raw = adata
    return adata


def test_defaults_to_x():
    adata = make_adata()
    matrix, genes, source = resolve_expression(adata)
    assert np.array_equal(np.asarray(matrix), np.asarray(adata.X))
    assert genes.tolist() == ["g1", "g2"]
    assert source == "X"


def test_selects_layer():
    adata = make_adata()
    matrix, _, source = resolve_expression(adata, layer="counts")
    assert np.array_equal(np.asarray(matrix), np.asarray(adata.layers["counts"]))
    assert source == "layer:counts"


def test_selects_raw():
    adata = make_adata()
    matrix, _, source = resolve_expression(adata, use_raw=True)
    assert np.array_equal(np.asarray(matrix), np.asarray(adata.raw.X))
    assert source == "raw"


def test_rejects_layer_and_raw_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_expression(make_adata(), layer="counts", use_raw=True)
```

- [ ] **Step 2: Run tests and verify expected import failure**

Run: `uv run pytest tests/test_expression.py -q`

Expected: collection fails because `scFEAann` does not exist.

- [ ] **Step 3: Add scaffold and minimal adapter**

Create `pyproject.toml` with a `src` layout, `scFEAann` package discovery,
runtime dependencies, optional `magic` and `test` groups, and package data for
`data/*.csv`. Implement source selection and validation for absent layer/raw,
matrix dimensionality, and duplicate names.

- [ ] **Step 4: Run source-selection tests**

Run: `uv run pytest tests/test_expression.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/scFEAann tests/test_expression.py
git commit -m "feat: add scFEAann expression source adapter"
```

### Task 2: Preprocessing parity and model assets

**Files:**
- Create: `src/scFEAann/preprocessing.py`
- Copy: `src/scFEAann/data/*.csv`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `resolve_expression(...)`
- Produces: `prepare_scfea_inputs(adata, *, layer, use_raw, sc_imputation, data_dir, module_gene_file, stoichiometry_matrix, compound_name_file) -> PreparedInputs`
- Produces: `PreparedInputs` fields `X`, `gene_scale`, `module_scale`, `gene_names`, `cell_names`, `module_names`, `compound_names`, `stoichiometry`, `gene_source`, `model_metadata`.

- [ ] **Step 1: Write failing adapter parity test**

```python
import anndata as ad
import pandas as pd
import numpy as np

from scFEAann.preprocessing import prepare_scfea_inputs


def test_prepared_genes_are_legacy_sorted_overlap():
    expr = pd.read_csv("input/Melissa_full.csv", index_col=0)
    adata = ad.AnnData(
        expr.T.astype(float),
        obs=pd.DataFrame(index=expr.columns.astype(str)),
        var=pd.DataFrame(index=expr.index.astype(str)),
    )
    prepared = prepare_scfea_inputs(adata)
    module = pd.read_csv("data/module_gene_m168.csv", index_col=0)
    expected = sorted(set(expr.index) & set(module.stack().dropna()))
    assert prepared.gene_names == expected
    assert prepared.X.shape == (
        adata.n_obs,
        module.shape[0] * len(expected),
    )
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_preprocessing.py -q`

Expected: import failure for `prepare_scfea_inputs`.

- [ ] **Step 3: Implement exact preprocessing and bundle model files**

Preserve the complete-matrix MAGIC/log/gene-scale sequence before overlap
selection. Build module-masked input blocks in module-file order. Count
non-missing module genes exactly as the original code does. Copy all candidate
human, mouse, iron, and glutaminolysis model CSV files into package data.

- [ ] **Step 4: Run preprocessing tests**

Run: `uv run pytest tests/test_preprocessing.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/scFEAann/preprocessing.py src/scFEAann/data tests/test_preprocessing.py
git commit -m "feat: add AnnData preprocessing adapter"
```

### Task 3: Core network, loss, and training

**Files:**
- Create: `src/scFEAann/core/__init__.py`
- Create: `src/scFEAann/core/dataset.py`
- Create: `src/scFEAann/core/flux.py`
- Create: `src/scFEAann/core/loss.py`
- Create: `src/scFEAann/core/training.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `FLUX`, `MyDataset`, `scfea_loss`, `fit_scfea`, `predict_scfea`,
  `compute_scfea_gradient`.
- `fit_scfea(...) -> FitResult(net, losses, elapsed_seconds)`.

- [ ] **Step 1: Write failing formula and shape tests**

```python
import torch

from scFEAann.core.flux import FLUX
from scFEAann.core.loss import scfea_loss


def test_flux_shapes_and_balance_formula():
    cm = torch.tensor([[1.0, -1.0]])
    net = FLUX(None, n_modules=2, f_in=3, f_out=1)
    x = torch.ones(4, 6)
    m, c = net(x, 2, 3, 1, cm)
    assert m.shape == (4, 2)
    assert c.shape == (4, 1)
    assert torch.equal(c[:, 0], m[:, 0] - m[:, 1])


def test_loss_has_five_components():
    m = torch.ones(2, 2, requires_grad=True)
    c = torch.zeros(2, 1)
    parts = scfea_loss(
        m,
        c,
        gene_scale=torch.ones(2),
        module_scale=torch.ones(2, 2),
    )
    assert len(parts) == 5
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_core.py -q`

Expected: import failure for `scFEAann.core`.

- [ ] **Step 3: Port the core implementation**

Copy the supplied `FLUX`, `MyDataset`, `myLoss`, and training steps into the
new modules. Only device placement and file-output separation may change.
Return loss arrays and elapsed time instead of writing loss/time files.

- [ ] **Step 4: Run core tests**

Run: `uv run pytest tests/test_core.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/scFEAann/core tests/test_core.py
git commit -m "feat: port scFEA core into scFEAann"
```

### Task 4: Flux and balance public API

**Files:**
- Create: `src/scFEAann/results.py`
- Create: `src/scFEAann/api.py`
- Modify: `src/scFEAann/__init__.py`
- Test: `tests/test_flux_api.py`

**Interfaces:**
- Produces: `run_scfeaann(adata, ...) -> AnnData`
- Produces: `write_flux_results(adata, flux, balance, context) -> None`
- Produces: `validate_group_key(adata, group_key) -> None`

- [ ] **Step 1: Write failing API behavior tests**

```python
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scFEAann import run_scfeaann


def tiny_adata():
    genes = ["ADPGK", "G6PC", "ALDOA"]
    return AnnData(
        np.arange(15, dtype=float).reshape(5, 3),
        obs=pd.DataFrame({"cell_type": ["a", "a", "b", "b", "b"]}, index=[f"c{i}" for i in range(5)]),
        var=pd.DataFrame(index=genes),
    )


def test_flux_api_writes_results_and_preserves_group():
    adata = tiny_adata()
    returned = run_scfeaann(
        adata, group_key="cell_type", epochs=1, verbose=False
    )
    assert returned is adata
    assert adata.obsm["scFEA_flux"].shape == (5, 168)
    assert adata.obsm["scFEA_balance"].shape == (5, 70)
    assert adata.obs["cell_type"].tolist() == ["a", "a", "b", "b", "b"]
    assert adata.uns["scFEAann"]["group_key"] == "cell_type"


def test_copy_true_does_not_modify_input():
    adata = tiny_adata()
    returned = run_scfeaann(adata, epochs=1, copy=True, verbose=False)
    assert "scFEA_flux" not in adata.obsm
    assert "scFEA_flux" in returned.obsm


def test_missing_group_column_raises():
    with pytest.raises(ValueError, match="group_key"):
        run_scfeaann(tiny_adata(), group_key="missing", epochs=1)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_flux_api.py -q`

Expected: feature functions are missing.

- [ ] **Step 3: Implement public flux API and result storage**

Use module and compound axis labels from the model files. Store parameters,
loss history, and source metadata under `uns["scFEAann"]` without deleting
existing gradient metadata.

- [ ] **Step 4: Run flux API tests**

Run: `uv run pytest tests/test_flux_api.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/scFEAann/api.py src/scFEAann/results.py src/scFEAann/__init__.py tests/test_flux_api.py
git commit -m "feat: add AnnData flux API"
```

### Task 5: Independent gradient API and retrieval helper

**Files:**
- Modify: `src/scFEAann/api.py`
- Modify: `src/scFEAann/results.py`
- Modify: `src/scFEAann/__init__.py`
- Test: `tests/test_gradient_api.py`

**Interfaces:**
- Produces: `calculate_scfeaann_gradient(adata, ...) -> AnnData`
- Produces: `get_scfeaann_gradient(adata, cell=None) -> pd.DataFrame | np.ndarray`

- [ ] **Step 1: Write failing gradient tests**

```python
import numpy as np

from scFEAann import calculate_scfeaann_gradient, get_scfeaann_gradient
from tests.helpers import tiny_adata


def test_gradient_is_stored_flat_and_can_be_restored():
    adata = tiny_adata()
    calculate_scfeaann_gradient(adata, epochs=1, verbose=False)
    flat = adata.obsm["scFEA_grad"]
    genes = adata.uns["scFEAann"]["gradient"]["gene_names"]
    modules = adata.uns["scFEAann"]["gradient"]["module_names"]
    assert flat.shape == (adata.n_obs, len(genes) * len(modules))

    one = get_scfeaann_gradient(adata, cell="c0")
    assert one.shape == (len(genes), len(modules))

    all_cells = get_scfeaann_gradient(adata)
    assert all_cells.shape == (adata.n_obs, len(genes), len(modules))
    assert np.array_equal(flat, all_cells.reshape(adata.n_obs, -1))
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_gradient_api.py -q`

Expected: gradient API is missing.

- [ ] **Step 3: Implement gradient API and retrieval**

Port the per-cell refitting loop from `src/scFEA_grad.py`. Store the gradient
using C-order flattening and document the gene/module axes in metadata.
Validate cell selectors with actionable errors.

- [ ] **Step 4: Run gradient tests**

Run: `uv run pytest tests/test_gradient_api.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/scFEAann/api.py src/scFEAann/results.py src/scFEAann/__init__.py tests/test_gradient_api.py tests/helpers.py
git commit -m "feat: add independent AnnData gradient API"
```

### Task 6: H5AD persistence, documentation, and full verification

**Files:**
- Create: `README_scFEAann.md`
- Test: `tests/test_roundtrip.py`

**Interfaces:**
- Consumes all public APIs.
- Produces documented usage and persisted `.h5ad` results.

- [ ] **Step 1: Write failing round-trip test**

```python
from anndata import read_h5ad

from scFEAann import run_scfeaann
from tests.helpers import tiny_adata


def test_results_survive_h5ad_roundtrip(tmp_path):
    adata = tiny_adata()
    run_scfeaann(adata, epochs=1, verbose=False)
    path = tmp_path / "result.h5ad"
    adata.write_h5ad(path)
    loaded = read_h5ad(path)
    assert loaded.obsm["scFEA_flux"].shape == adata.obsm["scFEA_flux"].shape
    assert loaded.uns["scFEAann"]["flux"]["module_names"] == (
        adata.uns["scFEAann"]["flux"]["module_names"]
    )
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_roundtrip.py -q`

Expected: failure until result metadata is H5AD-compatible.

- [ ] **Step 3: Implement persistence fixes and detailed documentation**

Document both APIs with parameter tables, computational cost, AnnData keys,
gradient reconstruction, group semantics, expression sources, uv setup, and a
minimal runnable example.

- [ ] **Step 4: Run full verification**

Run:

```bash
uv run pytest -q
uv run python -c "import scFEAann; from scFEAann import run_scfeaann, calculate_scfeaann_gradient, get_scfeaann_gradient"
```

Expected: all tests pass and imports succeed.

- [ ] **Step 5: Commit**

```bash
git add README_scFEAann.md tests
git commit -m "docs: document scFEAann AnnData API"
```
