# Supermodule Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reproducible supermodule and factor-graph network visualizations.

**Architecture:** Build network tables from model annotations and AnnData node
features, then render deterministic NetworkX layouts through the existing
scientific figure export layer.

**Tech Stack:** NetworkX, scikit-learn, pandas, NumPy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-supermodule-network-design.md`

## Global Constraints

- Edges come from model annotations, never from AnnData alone.
- Flux node attributes come from AnnData.
- Main node feature is `flux_mean`.
- Default layout seed is 16.
- Export layouts so figures can be reproduced.
- Integrate automatically into reports with supermodule results.

---

### Task 1: Factor graph and supermodule graph tables

**Files:**
- Create: `src/scFEAann/reporting/network.py`
- Test: `tests/test_network_tables.py`

- [ ] **Step 1: Write failing graph tests**

```python
def test_factor_graph_contains_module_and_compound_nodes():
    graph = build_factor_graph()
    assert graph.nodes["M_48"]["node_type"] == "module"
    assert graph.edges
```

- [ ] **Step 2: Verify red**

- [ ] **Step 3: Implement annotated bipartite and class graphs**

- [ ] **Step 4: Verify green**

### Task 2: Network figures

**Files:**
- Modify: `src/scFEAann/reporting/network.py`
- Test: `tests/test_network_figures.py`

- [ ] **Step 1: Write failing figure export tests**

- [ ] **Step 2: Verify red**

- [ ] **Step 3: Implement Figure 5 and Supplementary Figure 3**

- [ ] **Step 4: Verify green**

### Task 3: Report integration and final QA

**Files:**
- Modify: `src/scFEAann/reporting/report.py`
- Modify: `README_scFEAann.md`
- Test: `tests/test_network_integration.py`

- [ ] **Step 1: Write failing integration test**

- [ ] **Step 2: Verify red**

- [ ] **Step 3: Integrate network outputs, manifest, and legends**

- [ ] **Step 4: Generate the sample network report**

- [ ] **Step 5: Run visual, format, and full test verification**
