# Supermodule-Specific scFEA Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add class-specific scFEA submodels while retaining the global model.

**Architecture:** Reuse one prepared AnnData tensor, slice module blocks for each
supermodule, train sequential independent FLUX models, and assemble their
results into fixed AnnData matrix keys.

**Tech Stack:** Python, AnnData, PyTorch, pandas, NumPy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-23-supermodule-models-design.md`

## Global Constraints

- Keep the existing global 168-module model unchanged.
- Class models are independent and train on all cells.
- Keep all 70 compounds on each class balance axis.
- Do not add per-class gradients in version one.
- Default to `copy=False`.
- Support class selection by ID or class name.
- Store failed class slots as NaN and continue by default.

---

### Task 1: Supermodule mapping and prepared-input slicing

**Files:**
- Create: `src/scFEAann/supermodules.py`
- Copy: `src/scFEAann/data/supermetabolic_modules.csv`
- Test: `tests/test_supermodules.py`

**Interfaces:**
- Produces: `load_supermodules()`
- Produces: `resolve_supermodule_classes(classes=None) -> list[int]`
- Produces: `class_module_names(supermodule_id) -> list[str]`
- Produces: `subset_prepared_inputs(prepared, module_names) -> PreparedInputs`

- [ ] **Step 1: Write failing mapping tests**

```python
def test_supermodule_8_contains_expected_modules():
    assert class_module_names(8) == ["M_48", "M_49", "M_50", "M_51", "M_52"]
```

- [ ] **Step 2: Verify red**

Run: `.venv/bin/python -m pytest tests/test_supermodules.py -q`

Expected: missing module.

- [ ] **Step 3: Implement mapping and slicing**

Validate IDs, class names, module order, tensor width, stoichiometry columns,
and full compound rows.

- [ ] **Step 4: Verify green**

Run: `.venv/bin/python -m pytest tests/test_supermodules.py -q`

Expected: all pass.

### Task 2: Supermodule AnnData result storage and retrieval

**Files:**
- Modify: `src/scFEAann/results.py`
- Test: `tests/test_supermodule_results.py`

**Interfaces:**
- Produces: `write_supermodule_results(...)`
- Produces: `get_scfeaann_supermodule_flux(adata, class_id=None)`
- Produces: `get_scfeaann_supermodule_balance(adata, class_id=None)`

- [ ] **Step 1: Write failing storage tests**

```python
def test_supermodule_balance_has_22_by_70_axes():
    assert adata.obsm["scFEA_supermodule_balance"].shape == (n_obs, 22 * 70)
```

- [ ] **Step 2: Verify red**

Run: `.venv/bin/python -m pytest tests/test_supermodule_results.py -q`

Expected: missing result writers.

- [ ] **Step 3: Implement fixed-slot storage**

Use NaN for failed/unselected classes. Store all axes and run metadata.
Reconstruct one class as a cell-by-module DataFrame and balance as a
cell-by-compound DataFrame.

- [ ] **Step 4: Verify green**

Run: `.venv/bin/python -m pytest tests/test_supermodule_results.py -q`

Expected: all pass.

### Task 3: Public multi-model API

**Files:**
- Modify: `src/scFEAann/api.py`
- Modify: `src/scFEAann/__init__.py`
- Test: `tests/test_supermodule_api.py`

**Interfaces:**
- Produces: `run_scfeaann_supermodules(...) -> AnnData`

- [ ] **Step 1: Write failing API tests**

```python
def test_selected_classes_write_all_result_keys(small_model_adata):
    run_scfeaann_supermodules(
        small_model_adata, classes=[1, 8], epochs=1, verbose=False
    )
    assert "scFEA_supermodule_flux" in small_model_adata.obsm
    assert "scFEA_supermodule_flux_sum" in small_model_adata.obsm
```

- [ ] **Step 2: Verify red**

Run: `.venv/bin/python -m pytest tests/test_supermodule_api.py -q`

Expected: missing API.

- [ ] **Step 3: Implement orchestration**

Prepare inputs once, optionally run the global model, train class models,
assemble outputs, catch class-specific failures, and write metadata.

- [ ] **Step 4: Verify green**

Run: `.venv/bin/python -m pytest tests/test_supermodule_api.py -q`

Expected: all pass.

### Task 4: Documentation, smoke run, and verification

**Files:**
- Modify: `README_scFEAann.md`
- Test: `tests/test_supermodule_roundtrip.py`

- [ ] **Step 1: Add H5AD round-trip and copy tests**

- [ ] **Step 2: Document semantics, keys, selectors, and cost**

- [ ] **Step 3: Run a 200-cell class smoke test**

```bash
.venv/bin/python examples/scfeaann_supermodule_flow.py \
  --input outputs/testdata_scfeaann_sample.h5ad \
  --output outputs/testdata_scfeaann_supermodules_sample.h5ad \
  --epochs 1
```

- [ ] **Step 4: Run the complete test suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all tests pass.
