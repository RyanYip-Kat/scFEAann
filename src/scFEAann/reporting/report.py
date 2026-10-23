"""End-to-end scientific report generation for scFEAann results."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .embedding import compute_flux_embedding, get_transcriptome_embedding
from .exports import (
    build_figure_source_data,
    write_cell_tables,
    write_figure_source_data,
    write_summary_tables,
)
from .figures import (
    plot_figure1,
    plot_figure2,
    plot_figure3,
    plot_figure4_supermodules,
    plot_loss_history,
    plot_supp_supermodule_embedding,
)
from .network import (
    build_factor_graph,
    build_supermodule_network,
    plot_factor_graph,
    plot_supermodule_network,
    write_network_tables,
)
from .supermodules import (
    extract_supermodule_report,
    write_supermodule_tables,
)
from .tables import extract_report_results


def _relative_paths(paths: list[Path], output_dir: Path) -> list[str]:
    return [str(path.relative_to(output_dir)) for path in paths]


def _gradient_tables(
    adata: ad.AnnData,
    output_dir: Path,
) -> dict[str, Path]:
    if "scFEA_grad" not in adata.obsm:
        return {}
    metadata = adata.uns.get("scFEAann", {}).get("gradient", {})
    if not metadata:
        return {}
    shape = tuple(int(value) for value in metadata["shape"])
    gradient = np.asarray(adata.obsm["scFEA_grad"]).reshape(shape)
    gene_names = [str(name) for name in metadata["gene_names"]]
    module_names = [str(name) for name in metadata["module_names"]]
    finite = np.isfinite(gradient)
    finite_count = finite.sum(axis=0)
    absolute = np.abs(gradient)
    with np.errstate(invalid="ignore"):
        mean_absolute = np.nanmean(absolute, axis=0)
        maximum_absolute = np.nanmax(absolute, axis=0)
    rows: list[dict[str, object]] = []
    for gene_index, gene in enumerate(gene_names):
        for module_index, module in enumerate(module_names):
            rows.append(
                {
                    "gene": gene,
                    "module": module,
                    "n_finite": int(finite_count[gene_index, module_index]),
                    "n_cells": int(gradient.shape[0]),
                    "finite_fraction": float(
                        finite_count[gene_index, module_index]
                        / gradient.shape[0]
                    ),
                    "mean_absolute": float(
                        mean_absolute[gene_index, module_index]
                    ),
                    "max_absolute": float(
                        maximum_absolute[gene_index, module_index]
                    ),
                }
            )
    feature_summary = pd.DataFrame(rows)
    feature_path = output_dir / "gradient_feature_summary.csv"
    feature_summary.to_csv(feature_path, index=False)

    labels = (
        adata.obs.get("dataset_cluster")
        if "dataset_cluster" in adata.obs
        else pd.Series(["all"] * adata.n_obs, index=adata.obs_names)
    )
    top_rows: list[pd.DataFrame] = []
    for group in sorted(pd.Series(labels).astype(str).unique()):
        indices = np.flatnonzero(labels.astype(str).to_numpy() == group)
        with np.errstate(invalid="ignore"):
            group_mean = np.nanmean(absolute[indices], axis=0)
        flat_order = np.argsort(np.nan_to_num(group_mean, nan=-np.inf).ravel())[
            -10:
        ][::-1]
        for flat_index in flat_order:
            gene_index, module_index = np.unravel_index(
                flat_index, group_mean.shape
            )
            value = float(group_mean[gene_index, module_index])
            if not np.isfinite(value) or value <= 0:
                continue
            top_rows.append(
                pd.DataFrame(
                    [
                        {
                            "group": group,
                            "gene": gene_names[gene_index],
                            "module": module_names[module_index],
                            "mean_absolute": value,
                        }
                    ]
                )
            )
    top_path = output_dir / "gradient_top_by_group.csv"
    if top_rows:
        pd.concat(top_rows, ignore_index=True).to_csv(top_path, index=False)
    else:
        pd.DataFrame(
            columns=["group", "gene", "module", "mean_absolute"]
        ).to_csv(top_path, index=False)
    return {
        "feature_summary": feature_path,
        "top_by_group": top_path,
    }


def _software_versions() -> dict[str, str]:
    packages = [
        "scfeaann",
        "anndata",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _write_figure_legends(
    report,
    output_dir: Path,
    *,
    transcriptome_metadata: dict[str, Any],
    flux_metadata: dict[str, Any],
) -> Path:
    counts = ", ".join(
        f"{row.dataset} / {row.cluster}: {row.n_cells}"
        for row in report.group_counts.itertuples(index=False)
    )
    transcriptome_method = str(transcriptome_metadata["method"])
    flux_method = str(flux_metadata["method"]).upper()
    total_loss = report.loss_history[
        report.loss_history["loss"] == "total"
    ]["value"]
    if len(total_loss):
        loss_sentence = (
            f"The total-loss first and last values are {total_loss.iloc[0]:.6g} "
            f"and {total_loss.iloc[-1]:.6g}, respectively."
        )
    else:
        loss_sentence = "Total-loss history was unavailable in this AnnData."
    text = f"""# scFEAann figure legends

**Figure 1. Cohort composition and metabolic module structure.**
(a) Cell counts by dataset and cluster. (b) Pearson correlation between
dataset-level mean module-flux centroids within each cluster across
{report.flux.shape[1]} modules. (c) Heatmap of mean module flux for the top
group-variable modules. Values are z-scored across the six dataset-cluster
groups independently for each module; the color scale is centered at zero.
(d) Distributions of the four most group-variable modules. Box plots show the
median, interquartile range, and 1.5 x IQR whiskers; outliers are omitted from
the display, and all cell-level values are provided in the source-data tables.
Cell counts: {counts}.

**Figure 2. Compound-balance structure and feature effects.**
(a) Heatmap of mean compound balance for the most group-variable compounds,
z-scored across nested groups independently for each compound. (b)
Distributions of the four most group-variable compounds, with the same box-plot
definition as Figure 1. (c, d) Module and compound effect rankings. The
effect ratio is between-group variance divided by the sum of between-group and
mean within-group variance. It is descriptive and is not a hypothesis test.

**Figure 3. Transcriptome and flux representations.**
(a, c) Transcriptome embedding obtained from `{transcriptome_method}` and
colored by cluster and dataset, respectively. (b, d) Flux embedding computed
from standardized cell-by-module flux values using {flux_method}, colored by
cluster and dataset. Dataset colors are paired with circle/square markers.
Cluster colors are paired with circle/square/triangle markers.

**Supplementary Figure. Training loss.**
Total scFEA loss and its balance, non-negative, cell-variation, and
module-variation components across training epochs. {loss_sentence}

**Statistics.**
No cell-level p-values are reported. Individual cells are not treated as
independent biological replicates. All group sizes and raw feature summaries
are provided in `report_tables.xlsx`. Exact module reactions, input/output
KEGG identifiers, supermodule identifiers, and member genes are listed in the
`module_annotation` sheet and `module_annotation.csv`.
"""
    path = output_dir / "figure_legends.md"
    path.write_text(text)
    return path


def _append_supermodule_legends(
    path: Path,
    report,
    supermodule_report,
) -> None:
    text = f"""

**Figure 4. Supermodule-specific model results.**
(a) Mean flux for the {supermodule_report.node_mean.shape[1]} supermodule
classes, z-scored across the {len(supermodule_report.group_names)}
dataset-cluster groups independently for each class. (b) Cell-level
distributions of the six most group-variable supermodules; box plots show the
median, interquartile range, and 1.5 x IQR whiskers, with outliers omitted for
display. (c) Spearman correlation between the 22 class-specific node features
across cells. (d) Pearson and Spearman agreement between each independently
trained class model and the same class aggregated from the global 168-module
model. RMSE and MAE for those comparisons are provided in
`supermodule_model_agreement`.

**Supplementary Figure 2. Supermodule feature embedding.**
(a, b) PCA of the 22-dimensional supermodule mean-flux profile, colored by
cluster and dataset. (c) Cell-level comparison between global aggregation and
class-specific flux for representative classes. (d) Ranking of classes by the
between-group variance ratio. All figures use `flux_mean`; `flux_sum` remains
available in the node-feature tables for network analysis.
"""
    path.write_text(path.read_text() + text)


def _append_network_legends(path: Path, network) -> None:
    text = f"""

**Figure 5. Supermodule network.**
Nodes are the 22 supermodule classes; edges connect classes sharing annotated
metabolites, and edge width represents the number of shared compounds. Node
size indicates the number of modules in each class. (a) Colors show
topology-derived spectral clusters. (b) Colors show the standardized mean-flux
difference for {network.contrast_target} versus {network.contrast_reference}.
Continuous standardized differences are shown to avoid arbitrary cutoff-based
categories.

**Supplementary Figure 3. Human metabolic factor graph.**
Bipartite graph connecting metabolic modules (green diamonds) to their input
and output metabolites (pink circles). Edges are derived from
`Human_M168_information.symbols.csv`; AnnData contributes no topology. Every
module node is labeled with its module ID and every compound node with its KEGG
ID. Blue annotations around the perimeter identify the 22 supermodule classes
and are connected to their module centroids by leader lines. Full compound
names, module annotations, and edges are available in
`factor_graph_nodes.csv`, `factor_graph_edges.csv`, and
`factor_graph_supermodule_labels.csv`.
"""
    path.write_text(path.read_text() + text)


def export_scfeaann_report(
    adata: ad.AnnData,
    output_dir: str | Path,
    *,
    dataset_key: str = "dataset",
    cluster_key: str = "cluster",
    top_n: int = 30,
    embedding_method: str = "umap",
    random_state: int = 16,
    dpi: int = 300,
    network_contrast_key: str = "dataset",
    network_contrast: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Export SCI-style figures, summary tables, source data, and a manifest.

    This function is descriptive rather than inferential. It does not calculate
    cell-level p-values or split the scFEA model by group.
    """
    if dpi < 150:
        raise ValueError("dpi must be at least 150")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = extract_report_results(
        adata,
        dataset_key=dataset_key,
        cluster_key=cluster_key,
        top_n=top_n,
    )
    transcriptome_coordinates, transcriptome_metadata = (
        get_transcriptome_embedding(adata)
    )
    flux_coordinates, flux_embedding_metadata = compute_flux_embedding(
        np.asarray(adata.obsm["scFEA_flux"]),
        method=embedding_method,
        random_state=random_state,
    )
    source_data = build_figure_source_data(
        report,
        transcriptome_embedding=transcriptome_coordinates,
        flux_embedding=flux_coordinates,
    )

    summary_paths = write_summary_tables(report, output_dir)
    cell_paths = write_cell_tables(report, output_dir)
    source_paths = write_figure_source_data(source_data, output_dir)
    figure_paths = {
        "Fig1": plot_figure1(report, source_data, output_dir, dpi=dpi),
        "Fig2": plot_figure2(report, source_data, output_dir, dpi=dpi),
        "Fig3": plot_figure3(report, source_data, output_dir, dpi=dpi),
        "Supp_Fig_loss": plot_loss_history(
            report, source_data, output_dir, dpi=dpi
        ),
    }
    supermodule_paths: dict[str, object] = {}
    network_paths: dict[str, object] = {}
    supermodule_report = None
    network_data = None
    factor_graph = None
    if {
        "scFEA_supermodule_flux_mean",
        "scFEA_supermodule_flux_sum",
    } <= set(adata.obsm):
        supermodule_report = extract_supermodule_report(adata)
        supermodule_paths = write_supermodule_tables(
            supermodule_report, output_dir
        )
        figure_paths["Fig4"] = plot_figure4_supermodules(
            supermodule_report, output_dir, dpi=dpi
        )
        figure_paths["Supp_Fig2_supermodules"] = (
            plot_supp_supermodule_embedding(
                supermodule_report, output_dir, dpi=dpi
            )
        )
        network_data = build_supermodule_network(
            adata,
            contrast_key=network_contrast_key,
            contrast=network_contrast,
        )
        factor_graph = build_factor_graph()
        network_paths = write_network_tables(
            network_data, factor_graph, output_dir
        )
        figure_paths["Fig5"] = plot_supermodule_network(
            network_data, output_dir, dpi=dpi
        )
        figure_paths["Supp_Fig3_metabolic_network"] = plot_factor_graph(
            factor_graph, output_dir, dpi=dpi
        )
    gradient_paths = _gradient_tables(adata, output_dir)
    legends_path = _write_figure_legends(
        report,
        output_dir,
        transcriptome_metadata=transcriptome_metadata,
        flux_metadata=flux_embedding_metadata,
    )
    if supermodule_report is not None:
        _append_supermodule_legends(
            legends_path, report, supermodule_report
        )
    if network_data is not None:
        _append_network_legends(legends_path, network_data)

    manifest: dict[str, Any] = {
        "input": {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
        },
        "grouping": {
            "dataset_key": dataset_key,
            "cluster_key": cluster_key,
            "n_datasets": int(report.metadata["dataset"].nunique()),
            "n_clusters": int(report.metadata["cluster"].nunique()),
            "n_nested_groups": int(
                report.metadata["dataset_cluster"].nunique()
            ),
        },
        "features": {
            "n_modules": int(report.flux.shape[1]),
            "n_compounds": int(report.balance.shape[1]),
            "module_annotation_available": not report.module_annotation.empty,
            "top_n": int(top_n),
            "selected_modules": report.top_module_names,
            "selected_compounds": report.top_compound_names,
        },
        "embeddings": {
            "transcriptome": transcriptome_metadata,
            "flux": flux_embedding_metadata,
        },
        "figures": {
            key: _relative_paths(paths, output_dir)
            for key, paths in figure_paths.items()
        },
        "tables": {
            "workbook": str(summary_paths["workbook"].relative_to(output_dir)),
            **{
                key: str(path.relative_to(output_dir))
                for key, path in summary_paths.items()
                if key != "workbook"
            },
            "figure_legends": str(legends_path.relative_to(output_dir)),
            **{
                key: str(path.relative_to(output_dir))
                for key, path in cell_paths.items()
            },
            **{
                (
                    key
                    if key.startswith("supermodule_")
                    else f"supermodule_{key}"
                ): str(path.relative_to(output_dir))
                for key, path in supermodule_paths.items()
                if key not in {"workbook", "source_data"}
            },
            **{
                key: str(path.relative_to(output_dir))
                for key, path in network_paths.items()
            },
        },
        "figure_source_data": {
            key: str(path.relative_to(output_dir))
            for key, path in source_paths.items()
        }
        | {
            key: str(path.relative_to(output_dir))
            for key, path in (
                supermodule_paths.get("source_data", {})
                if isinstance(supermodule_paths.get("source_data", {}), dict)
                else {}
            ).items()
        },
        "gradient_tables": {
            key: str(path.relative_to(output_dir))
            for key, path in gradient_paths.items()
        },
        "software": _software_versions(),
        "warnings": [
            "No cell-level p-values are calculated; cells are not independent "
            "biological replicates.",
            "Heatmaps show group z-scores; raw group means are in the workbook.",
        ],
    }
    if supermodule_report is not None:
        manifest["supermodule_report"] = {
            "n_classes": len(supermodule_report.class_ids),
            "class_ids": supermodule_report.class_ids,
            "class_names": supermodule_report.class_names,
            "primary_feature": "flux_mean",
            "sum_feature_role": "tables_and_network_nodes",
            "top_effect_classes": supermodule_report.effects[
                "class_label"
            ].head(6).tolist(),
        }
    if network_data is not None and factor_graph is not None:
        manifest["networks"] = {
            "contrast_key": network_data.contrast_key,
            "contrast_reference": network_data.contrast_reference,
            "contrast_target": network_data.contrast_target,
            "n_supermodule_nodes": int(network_data.nodes.shape[0]),
            "n_supermodule_edges": int(network_data.edges.shape[0]),
            "n_factor_graph_nodes": int(factor_graph.nodes.shape[0]),
            "n_factor_graph_edges": int(factor_graph.edges.shape[0]),
        }
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
