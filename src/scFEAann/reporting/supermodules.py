"""Statistics and source data for supermodule-specific model results."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..modules import get_module_annotation
from ..supermodules import load_supermodules
from .embedding import compute_flux_embedding
from .exports import write_figure_source_data
from .tables import rank_feature_effects, summarize_features


@dataclass
class SupermoduleReport:
    node_mean: pd.DataFrame
    node_sum: pd.DataFrame
    global_class_mean: pd.DataFrame
    group_summary: pd.DataFrame
    group_summary_by_class: pd.DataFrame
    effects: pd.DataFrame
    agreement: pd.DataFrame
    correlation: pd.DataFrame
    class_annotation: pd.DataFrame
    embedding: pd.DataFrame
    source_data: dict[str, pd.DataFrame]
    class_ids: list[int]
    class_names: list[str]
    class_labels: list[str]
    group_names: list[str]


def _safe_correlations(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return np.nan, np.nan
    first_finite = first[finite]
    second_finite = second[finite]
    if np.var(first_finite) == 0 or np.var(second_finite) == 0:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            float(pearsonr(first_finite, second_finite).statistic),
            float(spearmanr(first_finite, second_finite).statistic),
        )


def _mean_absolute_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(np.abs(finite))) if finite.size else np.nan


def _group_zscore(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    mean = frame.groupby(groups, observed=True).mean()
    standard_deviation = mean.std(axis=0, ddof=0).replace(0, np.nan)
    return ((mean - mean.mean(axis=0)) / standard_deviation).fillna(0.0)


def extract_supermodule_report(adata: ad.AnnData) -> SupermoduleReport:
    """Calculate supermodule statistics and exact panel source data."""
    required = {
        "scFEA_flux",
        "scFEA_supermodule_flux_mean",
        "scFEA_supermodule_flux_sum",
    }
    missing = required - set(adata.obsm)
    if missing:
        raise ValueError(f"supermodule report requires obsm keys: {sorted(missing)}")
    metadata = adata.uns.get("scFEAann", {}).get("supermodules", {})
    if not metadata:
        raise ValueError("supermodule metadata is missing from adata.uns")
    dataset_key = "dataset"
    cluster_key = "cluster"
    if dataset_key not in adata.obs or cluster_key not in adata.obs:
        raise ValueError("supermodule report requires obs['dataset'] and obs['cluster']")

    class_ids = [int(value) for value in metadata["class_ids"]]
    class_names = [str(value) for value in metadata["class_names"]]
    class_labels = [
        f"SM{class_id:02d} {class_name}"
        for class_id, class_name in zip(class_ids, class_names, strict=True)
    ]
    node_mean = pd.DataFrame(
        np.asarray(adata.obsm["scFEA_supermodule_flux_mean"], dtype=float),
        index=adata.obs_names,
        columns=class_ids,
    )
    node_sum = pd.DataFrame(
        np.asarray(adata.obsm["scFEA_supermodule_flux_sum"], dtype=float),
        index=adata.obs_names,
        columns=class_ids,
    )
    groups = (
        adata.obs[dataset_key].astype(str)
        + " | "
        + adata.obs[cluster_key].astype(str)
    )
    group_names = sorted(groups.unique())

    annotation = get_module_annotation()
    global_flux = pd.DataFrame(
        np.asarray(adata.obsm["scFEA_flux"], dtype=float),
        index=adata.obs_names,
        columns=[str(name) for name in metadata["module_names"]],
    )
    global_class_values = np.full((adata.n_obs, 22), np.nan, dtype=float)
    agreement_rows: list[dict[str, object]] = []
    for slot, (class_id, class_name) in enumerate(
        zip(class_ids, class_names, strict=True)
    ):
        class_modules = annotation.index[
            annotation["supermodule_id"] == class_id
        ].tolist()
        available = [
            module for module in class_modules if module in global_flux.columns
        ]
        if available:
            global_values = global_flux.loc[:, available].mean(axis=1).to_numpy()
            global_class_values[:, slot] = global_values
        else:
            global_values = np.full(adata.n_obs, np.nan)
        class_values = node_mean.iloc[:, slot].to_numpy()
        pearson_r, spearman_r = _safe_correlations(
            global_values, class_values
        )
        finite = np.isfinite(global_values) & np.isfinite(class_values)
        if finite.any():
            difference = global_values[finite] - class_values[finite]
            rmse = float(np.sqrt(np.mean(difference**2)))
            mae = float(np.mean(np.abs(difference)))
        else:
            rmse = mae = np.nan
        agreement_rows.append(
            {
                "supermodule_id": class_id,
                "supermodule_class": class_name,
                "class_label": class_labels[slot],
                "n_modules": len(available),
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
                "rmse": rmse,
                "mae": mae,
                "global_mean_absolute": _mean_absolute_or_nan(global_values),
                "class_mean_absolute": _mean_absolute_or_nan(class_values),
            }
        )
    global_class_mean = pd.DataFrame(
        global_class_values,
        index=adata.obs_names,
        columns=class_ids,
    )
    agreement = pd.DataFrame(agreement_rows)

    mean_summary = summarize_features(
        node_mean.to_numpy(),
        groups,
        [str(class_id) for class_id in class_ids],
    )
    mean_summary.insert(0, "metric", "mean")
    sum_summary = summarize_features(
        node_sum.to_numpy(),
        groups,
        [str(class_id) for class_id in class_ids],
    )
    sum_summary.insert(0, "metric", "sum")
    group_summary = pd.concat(
        [mean_summary, sum_summary], ignore_index=True
    )
    class_lookup = {
        str(class_id): (class_name, class_labels[index])
        for index, (class_id, class_name) in enumerate(
            zip(class_ids, class_names, strict=True)
        )
    }
    group_summary_by_class = group_summary.copy()
    group_summary_by_class["supermodule_id"] = (
        group_summary_by_class["feature"].astype(int)
    )
    group_summary_by_class["supermodule_class"] = (
        group_summary_by_class["feature"].map(
            {key: value[0] for key, value in class_lookup.items()}
        )
    )
    group_summary_by_class["class_label"] = (
        group_summary_by_class["feature"].map(
            {key: value[1] for key, value in class_lookup.items()}
        )
    )

    effects = rank_feature_effects(
        node_mean.to_numpy(),
        groups,
        [str(class_id) for class_id in class_ids],
    )
    effects["supermodule_id"] = effects["feature"].astype(int)
    effects["supermodule_class"] = effects["feature"].map(
        {key: value[0] for key, value in class_lookup.items()}
    )
    effects["class_label"] = effects["feature"].map(
        {key: value[1] for key, value in class_lookup.items()}
    )

    correlation = node_mean.corr(method="spearman")
    correlation.index.name = "supermodule_id"
    correlation.columns.name = "target_supermodule_id"
    correlation_long = correlation.reset_index().melt(
        id_vars="supermodule_id",
        var_name="target_supermodule_id",
        value_name="spearman_r",
    )

    class_annotation_rows: list[dict[str, object]] = []
    supermodule_table = load_supermodules()
    for class_id, class_name in zip(class_ids, class_names, strict=True):
        class_modules = annotation.index[
            annotation["supermodule_id"] == class_id
        ].tolist()
        genes = {
            gene
            for module in class_modules
            for gene in str(
                annotation.loc[module, "module_genes"]
            ).split(";")
            if gene
        }
        class_annotation_rows.append(
            {
                "supermodule_id": class_id,
                "supermodule_class": class_name,
                "published_number_of_modules": int(
                    supermodule_table.loc[class_id, "Number of modules"]
                ),
                "current_number_of_modules": len(class_modules),
                "published_number_of_genes": int(
                    supermodule_table.loc[class_id, "Number of genes"]
                ),
                "current_unique_genes": len(genes),
                "module_ids": ";".join(class_modules),
                "unique_genes": ";".join(sorted(genes)),
            }
        )
    class_annotation = pd.DataFrame(class_annotation_rows)

    group_mean = node_mean.groupby(groups, observed=True).mean()
    group_zscore = _group_zscore(node_mean, groups)
    top_effect_ids = effects["feature"].head(6).astype(str).tolist()
    top_distribution = node_mean.loc[:, [int(value) for value in top_effect_ids]].copy()
    top_distribution["cell_id"] = adata.obs_names
    top_distribution["dataset"] = adata.obs[dataset_key].astype(str).to_numpy()
    top_distribution["cluster"] = adata.obs[cluster_key].astype(str).to_numpy()
    top_distribution["dataset_cluster"] = groups.to_numpy()
    top_distribution = top_distribution.melt(
        id_vars=["cell_id", "dataset", "cluster", "dataset_cluster"],
        var_name="supermodule_id",
        value_name="mean_flux",
    )

    embedding_values, embedding_metadata = compute_flux_embedding(
        node_mean.to_numpy(), method="pca"
    )
    embedding = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "dataset": adata.obs[dataset_key].astype(str).to_numpy(),
            "cluster": adata.obs[cluster_key].astype(str).to_numpy(),
            "supermodule_PC1": embedding_values[:, 0],
            "supermodule_PC2": embedding_values[:, 1],
        }
    )

    scatter_candidates = agreement[np.isfinite(agreement["pearson_r"])].sort_values(
        "pearson_r"
    ).head(3)
    scatter_frames: list[pd.DataFrame] = []
    for _, row in scatter_candidates.iterrows():
        class_id = int(row["supermodule_id"])
        slot = class_ids.index(class_id)
        global_values = global_class_mean.iloc[:, slot].to_numpy()
        class_values = node_mean.iloc[:, slot].to_numpy()
        global_std = np.nanstd(global_values)
        class_std = np.nanstd(class_values)
        global_z = (
            (global_values - np.nanmean(global_values)) / global_std
            if np.isfinite(global_std) and global_std > 0
            else np.zeros_like(global_values)
        )
        class_z = (
            (class_values - np.nanmean(class_values)) / class_std
            if np.isfinite(class_std) and class_std > 0
            else np.zeros_like(class_values)
        )
        scatter_frames.append(
            pd.DataFrame(
                {
                    "cell_id": adata.obs_names,
                    "supermodule_id": class_id,
                    "class_label": row["class_label"],
                    "dataset": adata.obs[dataset_key].astype(str).to_numpy(),
                    "cluster": adata.obs[cluster_key].astype(str).to_numpy(),
                    "global_mean": global_values,
                    "class_mean": class_values,
                    "global_z": global_z,
                    "class_z": class_z,
                    "pearson_r": float(row["pearson_r"]),
                }
            )
        )
    scatter = pd.concat(scatter_frames, ignore_index=True)

    heatmap = group_zscore.reset_index()
    heatmap.columns = ["group", *class_ids]
    source_data = {
        "Fig4a_group_heatmap": heatmap,
        "Fig4b_top_class_distributions": top_distribution,
        "Fig4c_class_correlation": correlation_long,
        "Fig4d_model_agreement": agreement,
        "SuppFig2a_class_embedding": embedding,
        "SuppFig2c_global_class_scatter": scatter,
        "SuppFig2d_effect_ranking": effects,
        "SuppFig2_group_means_raw": group_mean.reset_index(),
    }
    return SupermoduleReport(
        node_mean=node_mean,
        node_sum=node_sum,
        global_class_mean=global_class_mean,
        group_summary=group_summary,
        group_summary_by_class=group_summary_by_class,
        effects=effects,
        agreement=agreement,
        correlation=correlation,
        class_annotation=class_annotation,
        embedding=embedding,
        source_data=source_data,
        class_ids=class_ids,
        class_names=class_names,
        class_labels=class_labels,
        group_names=group_names,
    )


def write_supermodule_tables(
    report: SupermoduleReport,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write supermodule workbook sheets, network nodes, and panel source data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook = output_dir / "report_tables.xlsx"
    sheets = {
        "supermodule_group_summary": report.group_summary_by_class,
        "supermodule_effects": report.effects,
        "supermodule_model_agreement": report.agreement,
        "supermodule_correlation": report.correlation.reset_index(),
        "supermodule_annotation": report.class_annotation,
    }
    mode = "a" if workbook.exists() else "w"
    with pd.ExcelWriter(
        workbook,
        engine="openpyxl",
        mode=mode,
        if_sheet_exists="replace" if mode == "a" else None,
    ) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    short_names = [
        f"SM{class_id:02d}" for class_id in report.class_ids
    ]
    node_cells = report.embedding[
        ["cell_id", "dataset", "cluster"]
    ].set_index("cell_id")
    for index, short_name in enumerate(short_names):
        node_cells[f"{short_name}_mean_flux"] = report.node_mean.iloc[:, index].to_numpy()
        node_cells[f"{short_name}_sum_flux"] = report.node_sum.iloc[:, index].to_numpy()
    node_cells_path = output_dir / "supermodule_node_features_by_cell.csv.gz"
    node_cells.to_csv(node_cells_path)

    mean_metric = report.group_summary_by_class[
        report.group_summary_by_class["metric"] == "mean"
    ][["group", "supermodule_id", "class_label", "n", "mean"]].rename(
        columns={"n": "n_cells", "mean": "mean_flux"}
    )
    sum_metric = report.group_summary_by_class[
        report.group_summary_by_class["metric"] == "sum"
    ][["group", "supermodule_id", "mean"]].rename(
        columns={"mean": "sum_flux"}
    )
    node_groups = mean_metric.merge(
        sum_metric,
        on=["group", "supermodule_id"],
        how="left",
    )
    node_groups_path = output_dir / "supermodule_node_features_by_group.csv"
    node_groups.to_csv(node_groups_path, index=False)

    annotation_path = output_dir / "supermodule_annotation.csv"
    report.class_annotation.to_csv(annotation_path, index=False)
    correlation_path = output_dir / "supermodule_correlation.csv"
    report.correlation.to_csv(correlation_path)
    source_paths = write_figure_source_data(report.source_data, output_dir)
    return {
        "workbook": workbook,
        "node_features_by_cell": node_cells_path,
        "node_features_by_group": node_groups_path,
        "supermodule_annotation": annotation_path,
        "supermodule_correlation": correlation_path,
        "source_data": source_paths,
    }
