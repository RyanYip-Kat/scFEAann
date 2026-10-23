#!/usr/bin/env python3
"""Generate the scFEAann figures and summary tables from an H5AD result."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from scFEAann.reporting.report import export_scfeaann_report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-key", default="dataset")
    parser.add_argument("--cluster-key", default="cluster")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--embedding-method",
        choices=["umap", "tsne", "pca"],
        default="umap",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    adata = ad.read_h5ad(args.input, backed="r")
    try:
        manifest = export_scfeaann_report(
            adata,
            args.output_dir,
            dataset_key=args.dataset_key,
            cluster_key=args.cluster_key,
            top_n=args.top_n,
            embedding_method=args.embedding_method,
            random_state=args.random_state,
            dpi=args.dpi,
        )
    finally:
        file_handle = getattr(adata, "file", None)
        if file_handle is not None and hasattr(file_handle, "close"):
            file_handle.close()
    print(f"Report written to {args.output_dir}")
    print(f"Figures: {len(manifest['figures'])}")


if __name__ == "__main__":
    main()
