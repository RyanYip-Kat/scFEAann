"""Table and figure-source-data exports for scFEAann reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tables import ReportResults


def write_summary_tables(
    report: ReportResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the primary report workbook and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "report_tables.xlsx"
    sheets = {
        "group_counts": report.group_counts,
        "dataset_consistency": report.dataset_consistency,
        "module_group_summary": report.module_summary,
        "compound_group_summary": report.compound_summary,
        "module_variance_ranking": report.module_effects,
        "compound_variance_ranking": report.compound_effects,
        "loss_history": report.loss_history,
    }
    module_annotation_path = None
    if not report.module_annotation.empty:
        sheets["module_annotation"] = report.module_annotation.reset_index()
        module_annotation_path = output_dir / "module_annotation.csv"
        report.module_annotation.to_csv(module_annotation_path)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    outputs = {"workbook": workbook_path}
    if module_annotation_path is not None:
        outputs["module_annotation"] = module_annotation_path
    return outputs


def write_cell_tables(
    report: ReportResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write compressed cell metadata, flux, and balance tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cell_metadata": output_dir / "cell_metadata.csv.gz",
        "cell_flux": output_dir / "cell_flux.csv.gz",
        "cell_balance": output_dir / "cell_balance.csv.gz",
    }
    report.metadata.set_index("cell_id").to_csv(paths["cell_metadata"])
    report.flux.to_csv(paths["cell_flux"])
    report.balance.to_csv(paths["cell_balance"])
    return paths


def _consistency_scatter(report: ReportResults) -> pd.DataFrame:
    first = report.dataset_consistency.iloc[0]
    dataset_a = str(first["dataset_a"])
    dataset_b = str(first["dataset_b"])
    rows: list[dict[str, object]] = []
    for cluster in sorted(report.dataset_consistency["cluster"].unique()):
        selection_a = (
            (report.metadata["dataset"] == dataset_a)
            & (report.metadata["cluster"] == cluster)
        )
        selection_b = (
            (report.metadata["dataset"] == dataset_b)
            & (report.metadata["cluster"] == cluster)
        )
        centroid_a = report.flux.loc[selection_a.to_numpy()].mean(axis=0)
        centroid_b = report.flux.loc[selection_b.to_numpy()].mean(axis=0)
        stats = report.dataset_consistency[
            report.dataset_consistency["cluster"] == cluster
        ].iloc[0]
        for module in report.flux.columns:
            rows.append(
                {
                    "cluster": cluster,
                    "module": module,
                    "mean_a": float(centroid_a[module]),
                    "mean_b": float(centroid_b[module]),
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "pearson_r": float(stats["pearson_r"]),
                    "spearman_r": float(stats["spearman_r"]),
                }
            )
    return pd.DataFrame(rows)


def _heatmap_data(
    summary: pd.DataFrame,
    features: list[str],
    group_order: list[str],
) -> pd.DataFrame:
    selected = summary[summary["feature"].isin(features)]
    matrix = selected.pivot(
        index="group", columns="feature", values="mean"
    )
    matrix = matrix.reindex(index=group_order, columns=features)
    matrix.index.name = "group"
    return matrix.reset_index()


def _cell_feature_long(
    report: ReportResults,
    matrix: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    selected = matrix.loc[:, features].copy()
    selected["cell_id"] = report.metadata["cell_id"].to_numpy()
    selected["dataset"] = report.metadata["dataset"].to_numpy()
    selected["cluster"] = report.metadata["cluster"].to_numpy()
    selected["dataset_cluster"] = report.metadata["dataset_cluster"].to_numpy()
    long = selected.melt(
        id_vars=["cell_id", "dataset", "cluster", "dataset_cluster"],
        value_vars=features,
        var_name="feature",
        value_name="value",
    )
    if not report.module_annotation.empty:
        annotation = report.module_annotation.reset_index()[
            ["module_id", "reaction"]
        ].rename(columns={"module_id": "feature", "reaction": "module_reaction"})
        long = long.merge(annotation, on="feature", how="left")
    return long


def build_figure_source_data(
    report: ReportResults,
    *,
    transcriptome_embedding: np.ndarray,
    flux_embedding: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Build the exact data frames consumed by each figure panel."""
    transcriptome_embedding = np.asarray(transcriptome_embedding, dtype=float)
    flux_embedding = np.asarray(flux_embedding, dtype=float)
    if transcriptome_embedding.shape != (report.metadata.shape[0], 2):
        raise ValueError("transcriptome_embedding must have shape (n_obs, 2)")
    if flux_embedding.shape != (report.metadata.shape[0], 2):
        raise ValueError("flux_embedding must have shape (n_obs, 2)")
    group_order = report.group_counts["dataset_cluster"].tolist()
    top_modules = report.top_module_names[:4]
    top_compounds = report.top_compound_names[:4]
    embedding_metadata = report.metadata[
        ["cell_id", "dataset", "cluster", "dataset_cluster"]
    ].copy()
    transcriptome = embedding_metadata.copy()
    transcriptome["Embedding1"] = transcriptome_embedding[:, 0]
    transcriptome["Embedding2"] = transcriptome_embedding[:, 1]
    flux_coordinates = embedding_metadata.copy()
    flux_coordinates["Flux1"] = flux_embedding[:, 0]
    flux_coordinates["Flux2"] = flux_embedding[:, 1]

    return {
        "Fig1a_group_counts": report.group_counts.copy(),
        "Fig1b_dataset_consistency": _consistency_scatter(report),
        "Fig1c_module_heatmap": _heatmap_data(
            report.module_summary, report.top_module_names, group_order
        ),
        "Fig1d_top_modules": _cell_feature_long(
            report, report.flux, top_modules
        ),
        "Fig2a_balance_heatmap": _heatmap_data(
            report.compound_summary, report.top_compound_names, group_order
        ),
        "Fig2b_top_compounds": _cell_feature_long(
            report, report.balance, top_compounds
        ),
        "Fig2c_module_effects": report.module_effects.copy(),
        "Fig2d_compound_effects": report.compound_effects.copy(),
        "Fig3a_transcriptome_embedding": transcriptome,
        "Fig3b_flux_embedding": flux_coordinates,
        "Supp_loss_history": report.loss_history.copy(),
    }


def write_figure_source_data(
    source_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write one CSV file per figure panel."""
    source_dir = Path(output_dir) / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, frame in source_data.items():
        path = source_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = path
    return outputs
