#!/usr/bin/env python3
"""Run the global model plus independent human supermodule models."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from scFEAann import run_scfeaann_supermodules


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--classes",
        default="all",
        help="'all' or a comma-separated list of IDs/class names.",
    )
    parser.add_argument("--group-key", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.classes.strip().lower() == "all":
        classes = None
    else:
        classes = [
            token.strip()
            for token in args.classes.split(",")
            if token.strip()
        ]
        classes = [
            int(value) if value.isdigit() else value for value in classes
        ]

    adata = ad.read_h5ad(args.input)
    run_scfeaann_supermodules(
        adata,
        classes=classes,
        group_key=args.group_key,
        epochs=args.epochs,
        device=args.device,
        verbose=not args.quiet,
        continue_on_error=not args.stop_on_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Saved supermodule model results: {args.output}")


if __name__ == "__main__":
    main()
