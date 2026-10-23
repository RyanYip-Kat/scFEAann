#!/usr/bin/env python3
"""Run the scFEAann AnnData flow on testdata/testdata.h5ad.

The original scFEA algorithm does not use observation groups to split model
training. This example therefore:

1. preserves ``obs["dataset"]`` and ``obs["cluster"]``;
2. creates ``obs["dataset_cluster"]`` for nested group descriptions;
3. trains one scFEA model on every selected cell;
4. optionally calculates the independent per-cell gradient;
5. saves everything to a new H5AD file.

Use ``--max-cells`` for a deterministic stratified smoke test before running
the default full-cell analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from scFEAann import calculate_scfeaann_gradient, run_scfeaann


DEFAULT_INPUT = Path("testdata/testdata.h5ad")
DEFAULT_OUTPUT = Path("outputs/testdata_scfeaann_flow.h5ad")
DEFAULT_DATASET_COLUMN = "dataset"
DEFAULT_CLUSTER_COLUMN = "cluster"
DEFAULT_GROUP_COLUMN = "dataset_cluster"
GROUP_SEPARATOR = " | "


def add_nested_group_column(
    obs: pd.DataFrame,
    *,
    dataset_key: str = DEFAULT_DATASET_COLUMN,
    cluster_key: str = DEFAULT_CLUSTER_COLUMN,
    group_key: str = DEFAULT_GROUP_COLUMN,
) -> pd.Series:
    """Return a string column representing the ordered dataset-cluster pair.

    The input table is not modified. Missing group values are represented as
    ``"NA"`` so the resulting series remains H5AD-friendly.
    """
    for key in (dataset_key, cluster_key):
        if key not in obs.columns:
            raise ValueError(f"obs column '{key}' is required for nested grouping")
    values = (
        obs[dataset_key].astype("string").fillna("NA")
        + GROUP_SEPARATOR
        + obs[cluster_key].astype("string").fillna("NA")
    )
    values.name = group_key
    return values.astype(str)


def select_observation_positions(
    obs: pd.DataFrame,
    *,
    dataset: str | None = None,
    max_cells: int = 0,
    random_state: int = 0,
    dataset_key: str = DEFAULT_DATASET_COLUMN,
    cluster_key: str = DEFAULT_CLUSTER_COLUMN,
) -> np.ndarray:
    """Select original row positions, optionally stratified by nested groups.

    ``max_cells=0`` selects all rows. A positive limit distributes selected
    cells across every available ``(dataset, cluster)`` pair before filling
    remaining capacity. The original observation order is restored in the
    returned integer positions.
    """
    if max_cells < 0:
        raise ValueError("max_cells must be zero or greater")
    for key in (dataset_key, cluster_key):
        if key not in obs.columns:
            raise ValueError(f"obs column '{key}' is required for group selection")

    positions = np.arange(len(obs), dtype=np.int64)
    if dataset is not None:
        available = set(obs[dataset_key].dropna().astype(str))
        if dataset not in available:
            raise ValueError(
                f"dataset '{dataset}' is not present; available values: "
                f"{sorted(available)}"
            )
        positions = positions[
            obs[dataset_key].astype(str).to_numpy() == dataset
        ]

    if max_cells == 0 or len(positions) <= max_cells:
        return positions

    filtered = obs.iloc[positions]
    grouped_indices = list(
        filtered.groupby(
            [dataset_key, cluster_key], observed=True, sort=True
        ).indices.values()
    )
    generator = np.random.default_rng(random_state)
    shuffled_groups = []
    for indices in grouped_indices:
        shuffled = np.asarray(indices, dtype=np.int64).copy()
        generator.shuffle(shuffled)
        shuffled_groups.append(shuffled)

    per_group = max(1, max_cells // len(shuffled_groups))
    selected = [
        list(indices[:per_group])
        for indices in shuffled_groups
    ]
    selected_count = sum(len(indices) for indices in selected)
    group_offset = 0
    while selected_count < max_cells:
        added = False
        for group_index, indices in enumerate(shuffled_groups):
            candidate_offset = per_group + group_offset
            if candidate_offset < len(indices):
                selected[group_index].append(int(indices[candidate_offset]))
                selected_count += 1
                added = True
                if selected_count == max_cells:
                    break
        if not added:
            break
        group_offset += 1

    nested_positions = np.concatenate(
        [np.asarray(indices, dtype=np.int64) for indices in selected]
    )
    return np.sort(positions[nested_positions])


def load_testdata(
    path: str | Path,
    *,
    dataset: str | None = None,
    max_cells: int = 0,
    random_state: int = 0,
) -> ad.AnnData:
    """Load only selected cells from a backed H5AD file into memory."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"input H5AD does not exist: {path}")

    backed = ad.read_h5ad(path, backed="r")
    try:
        positions = select_observation_positions(
            backed.obs,
            dataset=dataset,
            max_cells=max_cells,
            random_state=random_state,
        )
        subset = backed[positions, :].to_memory()
    finally:
        file_handle = getattr(backed, "file", None)
        if file_handle is not None and hasattr(file_handle, "close"):
            file_handle.close()

    subset.obs[DEFAULT_GROUP_COLUMN] = add_nested_group_column(subset.obs)
    return subset


def run_testdata_flow(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    dataset: str | None = None,
    max_cells: int = 0,
    epochs: int = 1,
    gradient: bool = False,
    device: Any = "auto",
    random_state: int = 0,
    verbose: bool = True,
) -> ad.AnnData:
    """Run flux/balance, optional gradient, and save a new H5AD result.

    Parameters
    ----------
    input_path
        Source H5AD. It is opened in backed mode for cell selection.
    output_path
        Destination H5AD. It must differ from ``input_path``.
    dataset
        Optional value from ``obs["dataset"]``.
    max_cells
        Zero selects every matching cell. A positive value selects a
        deterministic stratified sample across dataset-cluster pairs.
    epochs
        Training epochs. The default of one is intended for smoke testing;
        use 100 for the original scFEA analysis setting.
    gradient
        Run the independent per-cell gradient calculation after flux.
    device
        Torch device forwarded to both APIs.
    random_state
        Seed used only for deterministic cell selection. scFEA training itself
        retains its original internal seed of 16.
    verbose
        Print group counts and progress.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("output_path must differ from input_path")

    subset = load_testdata(
        input_path,
        dataset=dataset,
        max_cells=max_cells,
        random_state=random_state,
    )
    if verbose:
        counts = subset.obs.groupby(
            [DEFAULT_DATASET_COLUMN, DEFAULT_CLUSTER_COLUMN],
            observed=True,
        ).size()
        print(f"Selected cells: {subset.n_obs} x genes: {subset.n_vars}")
        print(counts.to_string())

    run_scfeaann(
        subset,
        group_key=DEFAULT_GROUP_COLUMN,
        epochs=epochs,
        device=device,
        verbose=verbose,
    )
    if gradient:
        calculate_scfeaann_gradient(
            subset,
            group_key=DEFAULT_GROUP_COLUMN,
            epochs=epochs,
            device=device,
            verbose=verbose,
        )

    subset.uns["scFEAann"]["group_columns"] = [
        DEFAULT_DATASET_COLUMN,
        DEFAULT_CLUSTER_COLUMN,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(output_path)
    if verbose:
        print(f"Saved AnnData result: {output_path}")
    return subset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset value such as D1 or D2; default uses all.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="0 uses all cells; positive values create a stratified sample.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs; use 100 for the original analysis setting.",
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Also calculate and store gene-to-module gradients.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_testdata_flow(
        args.input,
        args.output,
        dataset=args.dataset,
        max_cells=args.max_cells,
        epochs=args.epochs,
        gradient=args.gradient,
        device=args.device,
        random_state=args.random_state,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
