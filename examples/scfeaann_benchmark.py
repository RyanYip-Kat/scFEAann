#!/usr/bin/env python3
"""Benchmark scFEAann preprocessing, training, and inference.

The default 1,000-cell sample keeps the benchmark practical. Use
``--max-cells 0`` only on a machine sized for the complete tensor.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any

import anndata as ad
import torch

from scFEAann.core.training import fit_scfea, predict_scfea, resolve_device
from scFEAann.preprocessing import (
    PreparedInputs,
    materialize_module_blocks,
    prepare_scfea_inputs,
)
from scFEAann.supermodules import class_module_names

try:
    from scfeaann_testdata_flow import load_testdata
except ImportError:
    from examples.scfeaann_testdata_flow import load_testdata


DEFAULT_INPUT = Path("testdata/testdata.h5ad")
DEFAULT_BATCH_SIZES = (1, 64, 1024)


def _rss_mib() -> tuple[float, float]:
    """Return current and peak resident memory on Linux."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return (float("nan"), float("nan"))
    values: dict[str, int] = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _unit = line.split()
            values[key.rstrip(":")] = int(value)
    return (
        values.get("VmRSS", 0) / 1024,
        values.get("VmHWM", 0) / 1024,
    )


def _memory_snapshot() -> dict[str, float]:
    current, peak = _rss_mib()
    return {"current_rss_mib": current, "peak_rss_mib": peak}


def _tensor_mib(prepared: PreparedInputs, n_modules: int | None = None) -> float:
    module_count = (
        len(prepared.module_names)
        if n_modules is None
        else min(n_modules, len(prepared.module_names))
    )
    values = (
        len(prepared.cell_names)
        * module_count
        * len(prepared.gene_names)
        * 4
    )
    return values / (1024**2)


def _timed_preprocessing(
    adata: ad.AnnData,
    *,
    build_module_blocks: bool,
) -> tuple[PreparedInputs, float, dict[str, float]]:
    started = time.perf_counter()
    prepared = prepare_scfea_inputs(
        adata,
        build_module_blocks=build_module_blocks,
    )
    elapsed = time.perf_counter() - started
    return prepared, elapsed, _memory_snapshot()


def _benchmark_inference(
    fitted,
    prepared: PreparedInputs,
    *,
    device: str,
    batch_sizes: list[int],
    repeats: int,
) -> list[dict[str, float | int]]:
    if prepared.X is None:
        raise RuntimeError("full module X was not materialized")
    warmup = min(64, len(prepared.cell_names))
    if warmup:
        predict_scfea(
            fitted.net,
            prepared.X[:warmup],
            prepared.stoichiometry,
            n_modules=len(prepared.module_names),
            n_genes=len(prepared.gene_names),
            n_comps=prepared.stoichiometry.shape[0],
            device=device,
            prediction_batch_size=warmup,
        )

    results: list[dict[str, float | int]] = []
    for batch_size in batch_sizes:
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            predict_scfea(
                fitted.net,
                prepared.X,
                prepared.stoichiometry,
                n_modules=len(prepared.module_names),
                n_genes=len(prepared.gene_names),
                n_comps=prepared.stoichiometry.shape[0],
                device=device,
                prediction_batch_size=batch_size,
            )
            durations.append(time.perf_counter() - started)
        seconds = mean(durations)
        results.append(
            {
                "prediction_batch_size": batch_size,
                "seconds": seconds,
                "cells_per_second": len(prepared.cell_names) / seconds,
            }
        )
    return results


def run_benchmark(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    max_cells: int = 1000,
    epochs: int = 1,
    device: str = "auto",
    prediction_batch_sizes: list[int] | None = None,
    repeats: int = 3,
    random_state: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run and return a machine-readable performance benchmark."""
    if repeats <= 0:
        raise ValueError("repeats must be greater than zero")
    batch_sizes = list(prediction_batch_sizes or DEFAULT_BATCH_SIZES)
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("prediction batch sizes must be greater than zero")

    input_path = Path(input_path)
    started = time.perf_counter()
    adata = load_testdata(
        input_path,
        max_cells=max_cells,
        random_state=random_state,
    )
    load_seconds = time.perf_counter() - started
    after_load = _memory_snapshot()

    deferred, deferred_seconds, after_deferred = _timed_preprocessing(
        adata, build_module_blocks=False
    )
    representative_modules = class_module_names(1, module_annotation=None)
    materialization_started = time.perf_counter()
    representative = materialize_module_blocks(
        deferred, representative_modules
    )
    materialization_seconds = time.perf_counter() - materialization_started
    representative_x_mib = _tensor_mib(representative)
    del representative
    del deferred
    gc.collect()
    after_deferred_cleanup = _memory_snapshot()

    full, full_seconds, after_full = _timed_preprocessing(
        adata, build_module_blocks=True
    )
    full_x_mib = _tensor_mib(full)

    selected_device = resolve_device(device)
    fit_started = time.perf_counter()
    fitted = fit_scfea(
        full.X,
        full.gene_scale,
        full.module_scale,
        full.stoichiometry,
        n_modules=len(full.module_names),
        n_genes=len(full.gene_names),
        n_comps=full.stoichiometry.shape[0],
        epochs=epochs,
        device=selected_device,
        verbose=verbose,
    )
    fit_wall_seconds = time.perf_counter() - fit_started
    after_fit = _memory_snapshot()
    inference = _benchmark_inference(
        fitted,
        full,
        device=str(selected_device),
        batch_sizes=batch_sizes,
        repeats=repeats,
    )
    after_inference = _memory_snapshot()

    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "anndata": version("anndata"),
            "device": str(selected_device),
            "torch_num_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
        },
        "input": {
            "path": str(input_path),
            "max_cells": max_cells,
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
        },
        "model": {
            "n_modules": len(full.module_names),
            "n_overlap_genes": len(full.gene_names),
            "n_compounds": len(full.compound_names),
            "n_representative_class_modules": len(
                representative_modules
            ),
        },
        "seconds": {
            "load": load_seconds,
            "preprocess_full": full_seconds,
            "preprocess_deferred": deferred_seconds,
            "fit_wall": fit_wall_seconds,
            "fit_reported": float(fitted.elapsed_seconds),
            "materialize_representative_class": materialization_seconds,
        },
        "tensor_mib": {
            "full_module_x": full_x_mib,
            "representative_class_x": representative_x_mib,
        },
        "memory": {
            "after_load": after_load,
            "after_full_preprocessing": after_full,
            "after_deferred_preprocessing": after_deferred,
            "after_deferred_cleanup": after_deferred_cleanup,
            "after_fit": after_fit,
            "after_inference": after_inference,
        },
        "inference": inference,
    }


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be comma-separated positive integers"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "must be comma-separated positive integers"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=1000,
        help="Stratified cell limit; 0 selects every cell.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prediction-batch-sizes",
        type=_parse_batch_sizes,
        default=list(DEFAULT_BATCH_SIZES),
        metavar="1,64,1024",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional destination for the JSON report.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.max_cells < 0:
        parser.error("--max-cells must be zero or greater")

    report = run_benchmark(
        input_path=args.input,
        max_cells=args.max_cells,
        epochs=args.epochs,
        device=args.device,
        prediction_batch_sizes=args.prediction_batch_sizes,
        repeats=args.repeats,
        random_state=args.random_state,
        verbose=not args.quiet,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
