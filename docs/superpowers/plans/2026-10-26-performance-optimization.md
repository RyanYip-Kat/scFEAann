# scFEAann P0 Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce preprocessing memory, speed up inference, and avoid full tensors
for class-only runs.

**Architecture:** Keep a tested dense reference path, add a sparse fast path and
direct float32 assembly, batch inference, and expose on-demand module blocks.

**Tech Stack:** AnnData, SciPy sparse, NumPy, torch, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-performance-optimization-design.md`

## Global Constraints

- Existing public APIs and result keys remain compatible.
- Training behavior remains unchanged.
- `prediction_batch_size` defaults to 1024 and is persisted in metadata.
- No silent densification of sparse non-MAGIC input.
- Selected-class execution does not build a full X tensor.

---

### Task 1: Preprocessing reference and sparse fast path

**Files:**
- Modify: `src/scFEAann/preprocessing.py`
- Test: `tests/test_performance_preprocessing.py`

**Interfaces:**
- Produces: `prepare_scfea_inputs(..., build_module_blocks=True)`.
- Preserves: `PreparedInputs`.

- [ ] **Step 1: Add failing exact-parity tests for dense/sparse paths.**
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement sparse fast path and direct float32 assembly.**
- [ ] **Step 4: Run parity tests and full preprocessing tests.**

### Task 2: Batched inference

**Files:**
- Modify: `src/scFEAann/core/training.py`
- Modify: `src/scFEAann/api.py`
- Test: `tests/test_inference_batching.py`

**Interfaces:**
- Produces: `predict_scfea(..., prediction_batch_size=1024)`.
- Public APIs accept and record `prediction_batch_size`.

- [ ] **Step 1: Add failing batch-one vs chunked parity test.**
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement chunked inference.**
- [ ] **Step 4: Run inference and API tests.**

### Task 3: On-demand supermodule tensors

**Files:**
- Modify: `src/scFEAann/preprocessing.py`
- Modify: `src/scFEAann/supermodules.py`
- Modify: `src/scFEAann/api.py`
- Test: `tests/test_supermodule_on_demand.py`

**Interfaces:**
- Produces: `materialize_module_blocks(prepared, module_names)`.
- `build_module_blocks=False` keeps global X unbuilt.

- [ ] **Step 1: Add failing class-only equivalence test.**
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement on-demand full and class materialization.**
- [ ] **Step 4: Run supermodule and H5AD tests.**

### Task 4: Benchmark and documentation

**Files:**
- Create: `examples/scfeaann_benchmark.py`
- Modify: `README_scFEAann.md`

- [ ] **Step 1: Add stage timing and RSS reporting.**
- [ ] **Step 2: Record benchmark output for the available sample.**
- [ ] **Step 3: Document `prediction_batch_size` and memory behavior.**
- [ ] **Step 4: Run the full test suite.**
