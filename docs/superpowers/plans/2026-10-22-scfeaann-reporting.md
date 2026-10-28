# scFEAann Scientific Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible SCI-style figures and summary-table export for
scFEAann AnnData results.

**Architecture:** A reporting package separates table calculation, embedding,
style/export, and figure composition. Figure source data are exported before
rendering, and a manifest records all inputs and outputs.

**Tech Stack:** Python 3.12, anndata, pandas, NumPy, matplotlib, seaborn,
scikit-learn, optional umap-learn, openpyxl, and Pillow.

**Spec:** `docs/superpowers/specs/2026-10-22-scfeaann-reporting-design.md`

## Global Constraints

- Do not modify scFEA core algorithms or existing AnnData result keys.
- Use `obsm["scFEA_flux"]` and `obsm["scFEA_balance"]`.
- Use English axis labels and captions.
- Double-column figures are 7.2 inches wide; single-column figures are 3.5
  inches wide.
- Use 6.5–9 pt text and PDF font type 42.
- Use colorblind-safe colors and redundant dataset marker shapes.
- Export PDF, SVG, PNG at 300 dpi, and grayscale PNG.
- Export figure source CSV files before rendering.
- Do not attach cell-level p-values.
- Only summarize gradients when present; never export the full dense gradient.

---

### Task 1: Reporting table calculations

**Files:**
- Create: `src/scFEAann/reporting/__init__.py`
- Create: `src/scFEAann/reporting/tables.py`
- Test: `tests/test_reporting_tables.py`

**Interfaces:**
- Produces: `extract_report_results(adata, dataset_key, cluster_key) -> ReportResults`
- Produces: `summarize_features(values, groups, feature_names) -> pd.DataFrame`
- Produces: `rank_feature_effects(values, groups, feature_names) -> pd.DataFrame`
- Produces: `dataset_consistency(values, dataset, cluster, feature_names) -> pd.DataFrame`

- [ ] **Step 1: Write failing table tests**

```python
import anndata as ad
import numpy as np
import pandas as pd

from scFEAann.reporting.tables import rank_feature_effects


def test_rank_feature_effects_orders_by_between_group_signal():
    values = np.array([
        [0, 0], [0, 0], [10, 0], [10, 0],
    ])
    groups = pd.Series(["a", "a", "b", "b"])
    ranked = rank_feature_effects(values, groups, ["signal", "flat"])
    assert ranked.iloc[0]["feature"] == "signal"
    assert ranked.iloc[0]["effect_ratio"] > ranked.iloc[1]["effect_ratio"]
```

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/test_reporting_tables.py -q`

Expected: import failure for `scFEAann.reporting.tables`.

- [ ] **Step 3: Implement table calculations**

Validate result shapes and grouping columns. Produce long-form grouped summaries,
effect rankings, cross-dataset cluster centroid correlations, group counts, and
optional gradient summaries.

- [ ] **Step 4: Run table tests**

Run: `.venv/bin/python -m pytest tests/test_reporting_tables.py -q`

Expected: all pass.

### Task 2: Embedding coordinates and report exports

**Files:**
- Create: `src/scFEAann/reporting/embedding.py`
- Create: `src/scFEAann/reporting/exports.py`
- Test: `tests/test_reporting_exports.py`

**Interfaces:**
- Produces: `compute_flux_embedding(flux, method="umap", random_state=16) -> tuple[np.ndarray, dict]`
- Produces: `write_summary_tables(report, output_dir) -> dict`
- Produces: `write_cell_tables(report, output_dir) -> dict`
- Produces: `write_figure_source_data(source_data, output_dir) -> dict`

- [ ] **Step 1: Write failing export tests**

```python
def test_flux_embedding_falls_back_to_pca(monkeypatch):
    flux = np.arange(120, dtype=float).reshape(20, 6)
    coords, metadata = compute_flux_embedding(flux, method="pca")
    assert coords.shape == (20, 2)
    assert metadata["method"] == "pca"
```

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/test_reporting_exports.py -q`

Expected: missing module or function.

- [ ] **Step 3: Implement embeddings and exports**

Use scikit-learn PCA. Attempt UMAP only when requested and installed; otherwise
use flux PC1/PC2 and record the fallback. Write compressed cell tables, an Excel
summary workbook, figure source CSV files, and a JSON manifest.

- [ ] **Step 4: Run export tests**

Run: `.venv/bin/python -m pytest tests/test_reporting_exports.py -q`

Expected: all pass.

### Task 3: SCI style and figure composition

**Files:**
- Create: `src/scFEAann/reporting/style.py`
- Create: `src/scFEAann/reporting/figures.py`
- Test: `tests/test_reporting_figures.py`

**Interfaces:**
- Produces: `apply_report_style()`
- Produces: `export_figure(fig, basename, *, size_inches, dpi=300) -> dict`
- Produces: `plot_figure1(report, source_data, output_dir) -> list[Path]`
- Produces: `plot_figure2(report, source_data, output_dir) -> list[Path]`
- Produces: `plot_figure3(report, source_data, output_dir) -> list[Path]`
- Produces: `plot_loss_history(report, source_data, output_dir) -> list[Path]`

- [ ] **Step 1: Write failing figure tests**

```python
def test_main_figures_export_all_formats(tmp_path, synthetic_report):
    paths = plot_figure1(synthetic_report, synthetic_report.source_data, tmp_path)
    suffixes = {path.suffix for path in paths}
    assert {".pdf", ".svg", ".png"} <= suffixes
```

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/test_reporting_figures.py -q`

Expected: missing figure functions.

- [ ] **Step 3: Implement the three main figures and loss figure**

Use fixed cluster colors and dataset color/marker pairs. Use stacked counts,
identity-line centroid scatter, symmetric heatmaps, boxplots, and paired
expression/flux embeddings. Use constrained layout and uniform panel labels.

- [ ] **Step 4: Run figure tests**

Run: `.venv/bin/python -m pytest tests/test_reporting_figures.py -q`

Expected: all pass.

### Task 4: End-to-end report API and visual QA

**Files:**
- Create: `src/scFEAann/reporting/report.py`
- Modify: `src/scFEAann/reporting/__init__.py`
- Create: `examples/scfeaann_report.py`
- Test: `tests/test_reporting_integration.py`

**Interfaces:**
- Produces: `export_scfeaann_report(adata, output_dir, ...) -> dict`

- [ ] **Step 1: Write failing integration test**

```python
def test_report_exports_tables_figures_and_manifest(tmp_path, synthetic_adata):
    manifest = export_scfeaann_report(
        synthetic_adata,
        tmp_path,
        embedding_method="pca",
        dpi=150,
    )
    assert (tmp_path / "report_tables.xlsx").exists()
    assert (tmp_path / "report_manifest.json").exists()
    assert (tmp_path / "Fig1.pdf").exists()
```

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/test_reporting_integration.py -q`

Expected: missing public function.

- [ ] **Step 3: Implement report orchestration and CLI**

Validate inputs, create output structure, compute all tables and source data,
render previews, run layout checks, export final figures, and write the manifest.
The CLI accepts an input H5AD, output directory, grouping keys, top-feature count,
embedding method, and dpi.

- [ ] **Step 4: Run integration tests**

Run: `.venv/bin/python -m pytest tests/test_reporting_integration.py -q`

Expected: all pass.

- [ ] **Step 5: Generate the 1-epoch validation report**

Run:

```bash
.venv/bin/python examples/scfeaann_report.py \
  --input outputs/testdata_scfeaann_full.h5ad \
  --output-dir outputs/report_1ep
```

Expected: three main figures, supplementary loss figure, workbook, source data,
cell-level tables, and manifest.

- [ ] **Step 6: Run visual QA**

Inspect rendered PNGs for text clipping, legend overlap, panel-label alignment,
heatmap readability, and color/marker distinguishability. Fix and re-render
until all checks pass.

### Task 5: Full 100-epoch run and final report

**Files:**
- Output: `outputs/testdata_scfeaann_full_100ep.h5ad`
- Output: `outputs/report_100ep/**`

**Interfaces:**
- Consumes: existing test-data flow and reporting API.

- [ ] **Step 1: Run the full 100-epoch analysis**

```bash
.venv/bin/python examples/scfeaann_testdata_flow.py \
  --output outputs/testdata_scfeaann_full_100ep.h5ad \
  --max-cells 0 \
  --epochs 100 \
  --device auto
```

- [ ] **Step 2: Generate the final report**

```bash
.venv/bin/python examples/scfeaann_report.py \
  --input outputs/testdata_scfeaann_full_100ep.h5ad \
  --output-dir outputs/report_100ep
```

- [ ] **Step 3: Run final verification**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass, all report files exist, and final PNG previews pass
visual inspection.
