"""Statistical tables derived from scFEAann AnnData results."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..modules import get_module_annotation


FLUX_KEY = "scFEA_flux"
BALANCE_KEY = "scFEA_balance"


@dataclass
class ReportResults:
    """Calculated report tables and aligned result matrices."""

    metadata: pd.DataFrame
    flux: pd.DataFrame
    balance: pd.DataFrame
    group_counts: pd.DataFrame
    module_summary: pd.DataFrame
    compound_summary: pd.DataFrame
    module_effects: pd.DataFrame
    compound_effects: pd.DataFrame
    dataset_consistency: pd.DataFrame
    loss_history: pd.DataFrame
    module_annotation: pd.DataFrame
    top_module_names: list[str]
    top_compound_names: list[str]
    dataset_key: str
    cluster_key: str


def _feature_names(
    adata: ad.AnnData,
    *,
    result_name: str,
    uns_section: str,
    prefix: str,
    width: int,
) -> list[str]:
    section = adata.uns.get("scFEAann", {}).get(uns_section, {})
    names = section.get("module_names" if result_name == FLUX_KEY else "compound_names")
    if names is None:
        return [f"{prefix}{index + 1}" for index in range(width)]
    names = [str(name) for name in names]
    if len(names) != width:
        raise ValueError(
            f"metadata for {result_name} has {len(names)} names but obsm has "
            f"{width} columns"
        )
    return names


def summarize_features(
    values: np.ndarray,
    groups: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return long-form per-group distribution statistics for each feature."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if len(groups) != values.shape[0]:
        raise ValueError("groups length must match values rows")
    if len(feature_names) != values.shape[1]:
        raise ValueError("feature_names length must match values columns")
    group_values = pd.Series(groups).astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for group in sorted(np.unique(group_values)):
        group_array = values[group_values == group]
        for feature_index, feature in enumerate(feature_names):
            sample = np.asarray(group_array[:, feature_index], dtype=float)
            finite = sample[np.isfinite(sample)]
            count = len(finite)
            if count:
                mean = float(np.mean(finite))
                standard_deviation = float(np.std(finite, ddof=1)) if count > 1 else np.nan
                sem = (
                    standard_deviation / np.sqrt(count)
                    if count > 1
                    else np.nan
                )
                q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
                minimum = float(np.min(finite))
                maximum = float(np.max(finite))
            else:
                mean = standard_deviation = sem = np.nan
                q25 = median = q75 = np.nan
                minimum = maximum = np.nan
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "n": count,
                    "mean": mean,
                    "sd": standard_deviation,
                    "sem": sem,
                    "median": float(median) if np.isfinite(median) else np.nan,
                    "q25": float(q25) if np.isfinite(q25) else np.nan,
                    "q75": float(q75) if np.isfinite(q75) else np.nan,
                    "min": minimum,
                    "max": maximum,
                }
            )
    return pd.DataFrame(rows)


def rank_feature_effects(
    values: np.ndarray,
    groups: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by between-group variance relative to total variance."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if len(groups) != values.shape[0]:
        raise ValueError("groups length must match values rows")
    if len(feature_names) != values.shape[1]:
        raise ValueError("feature_names length must match values columns")
    group_values = pd.Series(groups).astype(str).to_numpy()
    unique_groups = sorted(np.unique(group_values))
    if len(unique_groups) < 2:
        raise ValueError("at least two groups are required for effect ranking")

    group_means = []
    group_variances = []
    for group in unique_groups:
        sample = values[group_values == group]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            group_means.append(np.nanmean(sample, axis=0))
            group_variances.append(np.nanvar(sample, axis=0, ddof=1))
    between = np.nanvar(np.vstack(group_means), axis=0, ddof=1)
    within = np.nanmean(np.vstack(group_variances), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        effect_ratio = between / (between + within)
    effect_ratio = np.where(np.isfinite(effect_ratio), effect_ratio, 0.0)
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "between_variance": between,
            "within_variance": within,
            "effect_ratio": effect_ratio,
        }
    )
    return result.sort_values(
        ["effect_ratio", "between_variance"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def dataset_consistency(
    values: np.ndarray,
    datasets: pd.Series,
    clusters: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compare feature centroids between every pair of datasets per cluster."""
    values = np.asarray(values, dtype=float)
    datasets = pd.Series(datasets).astype(str)
    clusters = pd.Series(clusters).astype(str)
    if len(datasets) != values.shape[0] or len(clusters) != values.shape[0]:
        raise ValueError("datasets and clusters must align with values rows")
    if len(feature_names) != values.shape[1]:
        raise ValueError("feature_names length must match values columns")

    dataset_values = sorted(datasets.unique())
    rows: list[dict[str, object]] = []
    for dataset_a, dataset_b in itertools.combinations(dataset_values, 2):
        shared_clusters = sorted(
            set(clusters[datasets == dataset_a])
            & set(clusters[datasets == dataset_b])
        )
        for cluster in shared_clusters:
            mask_a = (datasets == dataset_a) & (clusters == cluster)
            mask_b = (datasets == dataset_b) & (clusters == cluster)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centroid_a = np.nanmean(values[mask_a.to_numpy()], axis=0)
                centroid_b = np.nanmean(values[mask_b.to_numpy()], axis=0)
            finite = np.isfinite(centroid_a) & np.isfinite(centroid_b)
            if np.sum(finite) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pearson_r = float(pearsonr(centroid_a[finite], centroid_b[finite]).statistic)
                spearman_r = float(spearmanr(centroid_a[finite], centroid_b[finite]).statistic)
            rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "cluster": cluster,
                    "n_cells_a": int(mask_a.sum()),
                    "n_cells_b": int(mask_b.sum()),
                    "n_features": int(finite.sum()),
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                }
            )
    return pd.DataFrame(rows)


def _loss_history(adata: ad.AnnData) -> pd.DataFrame:
    metadata = adata.uns.get("scFEAann", {})
    flux_metadata = metadata.get("flux", {}) if isinstance(metadata, dict) else {}
    history = flux_metadata.get("loss_history", {})
    if not history:
        return pd.DataFrame(columns=["epoch", "loss", "value"])
    rows: list[dict[str, object]] = []
    for loss_name, values in history.items():
        for epoch, value in enumerate(np.asarray(values), start=1):
            rows.append(
                {
                    "epoch": epoch,
                    "loss": str(loss_name),
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def _human_module_annotation(
    adata: ad.AnnData,
    module_names: list[str],
) -> pd.DataFrame:
    metadata = adata.uns.get("scFEAann", {})
    model_metadata = metadata.get("model", {}) if isinstance(metadata, dict) else {}
    module_file = Path(str(model_metadata.get("module_gene_file", ""))).name
    if module_file != "module_gene_m168.csv":
        return pd.DataFrame()
    annotation = get_module_annotation()
    if not set(module_names).issubset(annotation.index):
        return pd.DataFrame()
    return annotation


def _merge_module_annotation(
    frame: pd.DataFrame,
    annotation: pd.DataFrame,
) -> pd.DataFrame:
    if annotation.empty:
        return frame
    columns = [
        "reaction",
        "input_names",
        "input_kegg_ids",
        "output_names",
        "output_kegg_ids",
        "supermodule_id",
        "n_genes",
        "external_input",
        "external_output",
        "modeled_in_balance",
    ]
    joined = annotation.reset_index()[["module_id", *columns]]
    return frame.merge(
        joined,
        left_on="feature",
        right_on="module_id",
        how="left",
    ).drop(columns="module_id")


def extract_report_results(
    adata: ad.AnnData,
    *,
    dataset_key: str = "dataset",
    cluster_key: str = "cluster",
    top_n: int = 30,
) -> ReportResults:
    """Validate scFEAann results and calculate all summary report tables."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if FLUX_KEY not in adata.obsm:
        raise ValueError(f"adata.obsm['{FLUX_KEY}'] is required")
    if BALANCE_KEY not in adata.obsm:
        raise ValueError(f"adata.obsm['{BALANCE_KEY}'] is required")
    for key in (dataset_key, cluster_key):
        if key not in adata.obs.columns:
            raise ValueError(f"obs column '{key}' is required")
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    flux_values = np.asarray(adata.obsm[FLUX_KEY], dtype=float)
    balance_values = np.asarray(adata.obsm[BALANCE_KEY], dtype=float)
    if flux_values.shape[0] != adata.n_obs or balance_values.shape[0] != adata.n_obs:
        raise ValueError("scFEA result matrices must align with AnnData observations")
    module_names = _feature_names(
        adata,
        result_name=FLUX_KEY,
        uns_section="flux",
        prefix="M_",
        width=flux_values.shape[1],
    )
    compound_names = _feature_names(
        adata,
        result_name=BALANCE_KEY,
        uns_section="balance",
        prefix="C_",
        width=balance_values.shape[1],
    )

    datasets = adata.obs[dataset_key].astype(str)
    clusters = adata.obs[cluster_key].astype(str)
    if datasets.nunique() < 2:
        raise ValueError("at least two dataset levels are required")
    if clusters.nunique() < 2:
        raise ValueError("at least two cluster levels are required")

    nested = datasets + " | " + clusters
    metadata = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "dataset": datasets.to_numpy(),
            "cluster": clusters.to_numpy(),
            "dataset_cluster": nested.to_numpy(),
        }
    )
    flux = pd.DataFrame(
        flux_values, index=adata.obs_names, columns=module_names
    )
    balance = pd.DataFrame(
        balance_values, index=adata.obs_names, columns=compound_names
    )
    group_counts = (
        metadata.groupby(["dataset", "cluster"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    group_counts["dataset_cluster"] = (
        group_counts["dataset"] + " | " + group_counts["cluster"]
    )

    module_summary = summarize_features(
        flux_values, nested, module_names
    )
    compound_summary = summarize_features(
        balance_values, nested, compound_names
    )
    module_effects = rank_feature_effects(
        flux_values, nested, module_names
    )
    compound_effects = rank_feature_effects(
        balance_values, nested, compound_names
    )
    consistency = dataset_consistency(
        flux_values, datasets, clusters, module_names
    )
    if consistency.empty:
        raise ValueError(
            "datasets must share at least one cluster with two finite features"
        )
    top_modules = module_effects["feature"].head(top_n).astype(str).tolist()
    top_compounds = compound_effects["feature"].head(top_n).astype(str).tolist()
    module_annotation = _human_module_annotation(adata, module_names)
    module_summary = _merge_module_annotation(
        module_summary, module_annotation
    )
    module_effects = _merge_module_annotation(
        module_effects, module_annotation
    )

    return ReportResults(
        metadata=metadata,
        flux=flux,
        balance=balance,
        group_counts=group_counts,
        module_summary=module_summary,
        compound_summary=compound_summary,
        module_effects=module_effects,
        compound_effects=compound_effects,
        dataset_consistency=consistency,
        loss_history=_loss_history(adata),
        module_annotation=module_annotation,
        top_module_names=top_modules,
        top_compound_names=top_compounds,
        dataset_key=dataset_key,
        cluster_key=cluster_key,
    )
