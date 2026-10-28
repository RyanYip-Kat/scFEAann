# Supermodule Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add supermodule-level figures and tables to the existing report.

**Architecture:** A new reporting module extracts class features and comparisons,
then shares the existing style, source-data, export, and QA infrastructure.

**Tech Stack:** Python, AnnData, pandas, NumPy, SciPy, matplotlib, seaborn.

**Spec:** `docs/superpowers/specs/2026-10-23-supermodule-reporting-design.md`

## Global Constraints

- Main figures use `flux_mean`.
- `flux_sum` is retained in tables and network node exports.
- Automatically activate only when supermodule result keys exist.
- Reuse 7.2-inch double-column SCI styling.
- Export PDF, SVG, PNG, and grayscale PNG.
- No cell-level p-values.

---

### Task 1: Supermodule statistics

**Files:**
- Create: `src/scFEAann/reporting/supermodules.py`
- Test: `tests/test_reporting_supermodules.py`

**Interfaces:**
- Produces: `extract_supermodule_report(adata) -> SupermoduleReport`

- [ ] **Step 1: Write failing summary/agreement tests**

```python
def test_effect_ranking_uses_mean_features(supermodule_adata):
    report = extract_supermodule_report(supermodule_adata)
    assert report.effects["effect_ratio"].between(0, 1).all()
```

- [ ] **Step 2: Verify red**

Run: `.venv/bin/python -m pytest tests/test_reporting_supermodules.py -q`

Expected: missing module.

- [ ] **Step 3: Implement extraction and comparison**

Validate axes, compute group summaries for mean and sum, effect rankings,
22-class Spearman correlations, and global-versus-class agreement.

- [ ] **Step 4: Verify green**

Run: `.venv/bin/python -m pytest tests/test_reporting_supermodules.py -q`

Expected: all pass.

### Task 2: Supermodule figures and tables

**Files:**
- Modify: `src/scFEAann/reporting/supermodules.py`
- Modify: `src/scFEAann/reporting/figures.py`
- Test: `tests/test_reporting_supermodule_figures.py`

**Interfaces:**
- Produces: `plot_figure4_supermodules(...)`
- Produces: `plot_supp_supermodule_embedding(...)`
- Produces: `write_supermodule_tables(...)`
- Produces: `build_supermodule_source_data(...)`

- [ ] **Step 1: Write failing export tests**

```python
def test_supermodule_figures_export_formats(tmp_path, supermodule_report):
    paths = plot_figure4_supermodules(
        supermodule_report, supermodule_report.source_data, tmp_path
    )
    assert {path.suffix for path in paths} >= {".pdf", ".svg", ".png"}
```

- [ ] **Step 2: Verify red**

Run: `.venv/bin/python -m pytest tests/test_reporting_supermodule_figures.py -q`

Expected: missing plotting functions.

- [ ] **Step 3: Implement figures, workbook sheets, and network CSVs**

Use the shared `export_figure` function and fixed SCI style.

- [ ] **Step 4: Verify green**

Run: `.venv/bin/python -m pytest tests/test_reporting_supermodule_figures.py -q`

Expected: all pass.

### Task 3: Report integration and final validation

**Files:**
- Modify: `src/scFEAann/reporting/report.py`
- Test: `tests/test_reporting_supermodule_integration.py`
- Modify: `README_scFEAann.md`

- [ ] **Step 1: Write failing automatic-detection test**

```python
def test_report_detects_supermodule_results(tmp_path, supermodule_adata):
    manifest = export_scfeaann_report(supermodule_adata, tmp_path)
    assert (tmp_path / "Fig4.pdf").exists()
    assert "supermodule_effects" in pd.ExcelFile(
        tmp_path / "report_tables.xlsx"
    ).sheet_names
```

- [ ] **Step 2: Verify red**

- [ ] **Step 3: Integrate files and manifest entries**

- [ ] **Step 4: Generate the sample report**

```bash
.venv/bin/python examples/scfeaann_report.py \
  --input outputs/testdata_scfeaann_supermodules_sample.h5ad \
  --output-dir outputs/report_supermodules_1ep
```

- [ ] **Step 5: Run visual and full test verification**
