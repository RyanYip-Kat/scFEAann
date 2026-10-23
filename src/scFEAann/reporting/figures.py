"""SCI-style figures for scFEAann reports."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
from PIL import Image, ImageOps

from .style import (
    apply_report_style,
    cluster_color_map,
    dataset_style_map,
)
from .qa import audit_layout
from .supermodules import SupermoduleReport
from .tables import ReportResults


OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
CLUSTER_MARKERS = ["o", "s", "^"]


def cluster_display_label_map(labels: list[str]) -> dict[str, str]:
    """Return compact, unique figure labels for arbitrary cluster names."""
    known = {
        "corneal basal epithelium": "Basal",
        "corneal superficial epithelium": "Superficial",
        "corneal suprabasal epithelial cells": "Suprabasal",
    }
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for raw_label in labels:
        label = str(raw_label)
        base = known.get(label, label)
        if len(base) > 16:
            base = base[:14].rstrip(" _-")
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base[:13]}_{suffix}"
            suffix += 1
        mapping[label] = candidate
        used.add(candidate)
    return mapping


def _short_cluster_label(
    label: str,
    display_labels: dict[str, str],
) -> str:
    return display_labels[str(label)]


def _short_nested_label(
    label: str,
    display_labels: dict[str, str],
) -> str:
    dataset, cluster = str(label).split(" | ", maxsplit=1)
    return f"{dataset}\n{_short_cluster_label(cluster, display_labels)}"


def _cluster_marker_map(labels: list[str]) -> dict[str, str]:
    return {
        label: CLUSTER_MARKERS[index % len(CLUSTER_MARKERS)]
        for index, label in enumerate(labels)
    }


def _panel_labels(
    fig,
    axes: Iterable,
    labels: str,
    *,
    x_offset: float = -0.014,
    y_offset: float = 0.018,
) -> None:
    fig.canvas.draw()
    for axis, label in zip(axes, labels, strict=True):
        bbox = axis.get_position()
        fig.text(
            bbox.x0 + x_offset,
            min(bbox.y1 + y_offset, 0.995),
            label,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="top",
        )


def _panel_labels_above(fig, axes: Iterable, labels: str) -> None:
    fig.canvas.draw()
    for axis, label in zip(axes, labels, strict=True):
        bbox = axis.get_position()
        fig.text(
            bbox.x0,
            min(bbox.y1 + 0.022, 0.99),
            label,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="top",
        )


def _standardized_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    group_column = frame.columns[0]
    matrix = frame.set_index(group_column)
    standard_deviation = matrix.std(axis=0, ddof=0).replace(0, np.nan)
    standardized = (matrix - matrix.mean(axis=0)) / standard_deviation
    return standardized.fillna(0.0)


def _heatmap(
    axis,
    frame: pd.DataFrame,
    *,
    colorbar_label: str,
    display_labels: dict[str, str],
    max_features: int = 12,
) -> None:
    group_column = frame.columns[0]
    feature_columns = list(frame.columns[1:])[:max_features]
    frame = frame[[group_column, *feature_columns]]
    matrix = _standardized_heatmap(frame)
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    limit = max(limit, 1.0)
    row_labels = [
        _short_nested_label(label, display_labels)
        for label in matrix.index
    ]
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.3,
        linecolor="white",
        yticklabels=row_labels,
        xticklabels=True,
        cbar_kws={"label": colorbar_label, "shrink": 0.62, "pad": 0.015},
    )
    axis.tick_params(axis="x", rotation=90, labelsize=6)
    axis.tick_params(axis="y", rotation=0, labelsize=6)
    axis.set_xlabel("")
    axis.set_ylabel("")


def _top_feature_panels(
    parent_axis,
    frame: pd.DataFrame,
    *,
    feature_column: str,
    dataset_palette: dict[str, str],
    display_labels: dict[str, str],
    value_label: str,
) -> None:
    parent_axis.set_axis_off()
    work = frame.copy()
    work["group"] = (
        work["dataset"].astype(str)
        + " "
        + work["cluster"].map(display_labels)
    )
    features = list(pd.unique(frame[feature_column]))[:4]
    grid = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=parent_axis.get_subplotspec(),
        wspace=0.18,
        hspace=0.28,
    )
    for index, feature in enumerate(features):
        mini = parent_axis.figure.add_subplot(grid[index])
        subset = work[work[feature_column] == feature]
        maximum = float(np.nanmax(np.abs(subset["value"].to_numpy())))
        exponent = int(np.floor(np.log10(maximum))) if maximum > 0 else 0
        scale = 10.0**exponent
        subset = subset.assign(value=subset["value"] / scale)
        sns.boxplot(
            data=subset,
            x="group",
            y="value",
            hue="dataset",
            palette=dataset_palette,
            legend=False,
            showfliers=False,
            linewidth=0.55,
            width=0.72,
            ax=mini,
        )
        scale_note = "" if scale == 1.0 else f"($\\times10^{{{exponent}}}$)"
        if scale_note and len(str(feature)) > 10:
            title = f"{feature}\n{scale_note}"
        else:
            title = f"{feature}{' ' + scale_note if scale_note else ''}"
        mini.set_title(title, fontsize=7, pad=1.5)
        mini.set_xlabel("")
        mini.set_ylabel(value_label if index % 2 == 0 else "", fontsize=6)
        mini.tick_params(axis="x", rotation=90, labelsize=6, pad=1.5)
        mini.tick_params(axis="y", labelsize=6)
        if index < 2:
            mini.tick_params(axis="x", labelbottom=False)


def export_figure(
    fig,
    basename: str | Path,
    *,
    size_inches: tuple[float, float],
    formats: list[str] | tuple[str, ...] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Export a figure to vector/raster formats and a grayscale PNG preview."""
    formats = list(formats)
    if not formats:
        raise ValueError("formats must contain at least one output format")
    allowed = {"pdf", "svg", "png"}
    unsupported = set(formats) - allowed
    if unsupported:
        raise ValueError(f"unsupported figure formats: {sorted(unsupported)}")
    basename = Path(basename)
    basename.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*size_inches)
    issues = audit_layout(fig)
    failures = [message for severity, message in issues if severity == "FAIL"]
    if failures:
        raise RuntimeError("figure layout audit failed: " + " | ".join(failures))
    outputs: list[Path] = []
    for suffix in formats:
        output = basename.with_suffix(f".{suffix}")
        fig.savefig(
            output,
            dpi=dpi,
            bbox_inches=None,
            facecolor="white",
        )
        outputs.append(output)

    if "png" in formats:
        color_png = basename.with_suffix(".png")
        with Image.open(color_png) as image:
            grayscale = ImageOps.grayscale(image)
            grayscale_path = basename.parent / f"{basename.name}_grayscale.png"
            grayscale.save(grayscale_path, dpi=(dpi, dpi))
    else:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_png = Path(temporary) / "preview.png"
            fig.savefig(temporary_png, dpi=dpi, facecolor="white")
            with Image.open(temporary_png) as image:
                grayscale = ImageOps.grayscale(image)
                grayscale_path = basename.parent / f"{basename.name}_grayscale.png"
                grayscale.save(grayscale_path, dpi=(dpi, dpi))
    outputs.append(grayscale_path)
    return outputs


def plot_figure1(
    report: ReportResults,
    source_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot cohort structure, cross-dataset consistency, and module flux."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    cluster_labels = sorted(report.metadata["cluster"].unique())
    cluster_colors = cluster_color_map(cluster_labels)
    display_labels = cluster_display_label_map(cluster_labels)
    dataset_labels = sorted(report.metadata["dataset"].unique())
    dataset_styles = dataset_style_map(dataset_labels)
    dataset_palette = {
        dataset: style[0] for dataset, style in dataset_styles.items()
    }

    counts = source_data["Fig1a_group_counts"]
    count_matrix = counts.pivot(
        index="dataset", columns="cluster", values="n_cells"
    ).reindex(columns=cluster_labels, fill_value=0)
    count_matrix.plot(
        kind="bar",
        stacked=True,
        color=[cluster_colors[label] for label in count_matrix.columns],
        width=0.72,
        linewidth=0.4,
        edgecolor="white",
        ax=axes[0, 0],
    )
    for container, hatch in zip(
        axes[0, 0].containers,
        ["", "///", "\\\\\\"],
        strict=False,
    ):
        for patch in container.patches:
            patch.set_hatch(hatch)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Cells (n)")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].legend(
        title="Cluster",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=6,
        title_fontsize=6,
        labels=[
            _short_cluster_label(label, display_labels)
            for label in count_matrix.columns
        ],
    )

    consistency = source_data["Fig1b_dataset_consistency"]
    axes_b = axes[0, 1]
    correlation_summary = (
        consistency.groupby("cluster", observed=True)
        .agg(pearson_r=("pearson_r", "first"))
        .reset_index()
    )
    cluster_markers = _cluster_marker_map(cluster_labels)
    for _, row in correlation_summary.iterrows():
        cluster = row["cluster"]
        axes_b.scatter(
            row["pearson_r"],
            _short_cluster_label(cluster, display_labels),
            s=28,
            alpha=0.9,
            color=cluster_colors[cluster],
            marker=cluster_markers[cluster],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        axes_b.text(
            row["pearson_r"],
            _short_cluster_label(cluster, display_labels),
            f"  {row['pearson_r']:.5f}",
            va="center",
            fontsize=5.5,
        )
    minimum = max(0.0, float(correlation_summary["pearson_r"].min()) - 0.01)
    axes_b.set_xlim(minimum, 1.0001)
    axes_b.set_xlabel(
        f"Pearson r: {consistency['dataset_a'].iloc[0]} vs "
        f"{consistency['dataset_b'].iloc[0]}"
    )
    axes_b.set_ylabel("")

    _heatmap(
        axes[1, 0],
        source_data["Fig1c_module_heatmap"],
        colorbar_label="Module flux (group z-score)",
        display_labels=display_labels,
    )
    _top_feature_panels(
        axes[1, 1],
        source_data["Fig1d_top_modules"],
        feature_column="feature",
        dataset_palette=dataset_palette,
        display_labels=display_labels,
        value_label="Module flux",
    )
    _panel_labels(fig, axes.flat, "abcd")
    outputs = export_figure(
        fig,
        output_dir / "Fig1",
        size_inches=(7.2, 6.2),
        dpi=dpi,
    )
    plt.close(fig)
    return outputs


def plot_figure2(
    report: ReportResults,
    source_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot balance patterns and feature-level effect rankings."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    display_labels = cluster_display_label_map(
        sorted(report.metadata["cluster"].unique())
    )
    dataset_labels = sorted(report.metadata["dataset"].unique())
    dataset_styles = dataset_style_map(dataset_labels)
    dataset_palette = {
        dataset: style[0] for dataset, style in dataset_styles.items()
    }

    _heatmap(
        axes[0, 0],
        source_data["Fig2a_balance_heatmap"],
        colorbar_label="Balance (group z-score)",
        display_labels=display_labels,
    )
    _top_feature_panels(
        axes[0, 1],
        source_data["Fig2b_top_compounds"],
        feature_column="feature",
        dataset_palette=dataset_palette,
        display_labels=display_labels,
        value_label="Balance",
    )
    for axis, key, label in (
        (axes[1, 0], "Fig2c_module_effects", "Module flux"),
        (axes[1, 1], "Fig2d_compound_effects", "Compound balance"),
    ):
        frame = source_data[key].head(15).iloc[::-1]
        axis.barh(
            frame["feature"],
            frame["effect_ratio"],
            color=OKABE[0],
            edgecolor="black",
            linewidth=0.35,
        )
        axis.set_xlabel("Between-group variance ratio")
        axis.set_ylabel("")
        axis.set_title(label, fontsize=8)
        axis.tick_params(axis="y", labelsize=6)
    _panel_labels(fig, axes.flat, "abcd")
    outputs = export_figure(
        fig,
        output_dir / "Fig2",
        size_inches=(7.2, 6.2),
        dpi=dpi,
    )
    plt.close(fig)
    return outputs


def plot_figure3(
    report: ReportResults,
    source_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot transcriptome and flux embeddings with consistent encodings."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    cluster_labels = sorted(report.metadata["cluster"].unique())
    cluster_colors = cluster_color_map(cluster_labels)
    cluster_markers = _cluster_marker_map(cluster_labels)
    display_labels = cluster_display_label_map(cluster_labels)
    dataset_labels = sorted(report.metadata["dataset"].unique())
    dataset_styles = dataset_style_map(dataset_labels)
    transcriptome = source_data["Fig3a_transcriptome_embedding"]
    flux = source_data["Fig3b_flux_embedding"]

    for axis, frame, x, y, title in (
        (
            axes[0, 0],
            transcriptome,
            "Embedding1",
            "Embedding2",
            "Transcriptome embedding",
        ),
        (axes[0, 1], flux, "Flux1", "Flux2", "Flux embedding"),
    ):
        for cluster in cluster_labels:
            subset = frame[frame["cluster"] == cluster]
            axis.scatter(
                subset[x],
                subset[y],
                s=1.2,
                alpha=0.55,
                color=cluster_colors[cluster],
                marker=cluster_markers[cluster],
                linewidths=0,
                label=_short_cluster_label(cluster, display_labels),
            )
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_title(title, fontsize=8)
    axes[0, 0].legend(
        title="Cluster",
        frameon=False,
        markerscale=5,
        fontsize=5.5,
        title_fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )

    for axis, frame, x, y, title in (
        (
            axes[1, 0],
            transcriptome,
            "Embedding1",
            "Embedding2",
            "Transcriptome embedding",
        ),
        (axes[1, 1], flux, "Flux1", "Flux2", "Flux embedding"),
    ):
        for dataset in dataset_labels:
            subset = frame[frame["dataset"] == dataset]
            color, marker = dataset_styles[dataset]
            axis.scatter(
                subset[x],
                subset[y],
                s=3.0,
                alpha=0.65,
                color=color,
                marker=marker,
                linewidths=0,
                label=dataset,
            )
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_title(title, fontsize=8)
    axes[1, 0].legend(
        title="Dataset",
        frameon=False,
        markerscale=3,
        fontsize=6,
        title_fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    _panel_labels(fig, axes.flat, "abcd")
    outputs = export_figure(
        fig,
        output_dir / "Fig3",
        size_inches=(7.2, 6.4),
        dpi=dpi,
    )
    plt.close(fig)
    return outputs


def plot_loss_history(
    report: ReportResults,
    source_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot the original five scFEA training-loss components."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axis = plt.subplots(figsize=(3.5, 2.6))
    history = source_data["Supp_loss_history"]
    if history.empty:
        axis.text(
            0.5,
            0.5,
            "Loss history unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        styles = {
            "total": ("-", "o"),
            "balance": ("--", "s"),
            "negative": ("-.", "^"),
            "cell": (":", "D"),
            "module": ((0, (5, 1)), "v"),
        }
        for index, (loss_name, subset) in enumerate(
            history.groupby("loss", sort=False)
        ):
            line_style, marker = styles.get(loss_name, ("-", "o"))
            axis.plot(
                subset["epoch"],
                subset["value"],
                color=OKABE[index % len(OKABE)],
                linestyle=line_style,
                marker=marker,
                markevery=max(1, len(subset) // 8),
                markersize=2.5,
                linewidth=0.8,
                label=loss_name.capitalize(),
            )
        values = history.loc[history["value"] > 0, "value"]
        if not values.empty and values.max() / values.min() > 100:
            axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend(frameon=False, fontsize=6, ncol=2)
    output = export_figure(
        fig,
        output_dir / "Supp_Fig_loss",
        size_inches=(3.5, 2.6),
        dpi=dpi,
    )
    plt.close(fig)
    return output


def _supermodule_class_labels(class_ids: list[int]) -> list[str]:
    return [f"SM{class_id:02d}" for class_id in class_ids]


def _supermodule_group_labels(groups: list[str]) -> list[str]:
    labels = []
    for group in groups:
        dataset, cluster = str(group).split(" | ", maxsplit=1)
        labels.append(f"{dataset}\n{_short_cluster_label(cluster, cluster_display_label_map([cluster]))}")
    return labels


def plot_figure4_supermodules(
    report: SupermoduleReport,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot supermodule group patterns, distributions, similarity, and agreement."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    class_labels = _supermodule_class_labels(report.class_ids)
    group_labels = _supermodule_group_labels(report.group_names)

    heatmap = report.source_data["Fig4a_group_heatmap"].set_index("group")
    heatmap = heatmap.loc[report.group_names]
    matrix = heatmap.T
    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    sns.heatmap(
        matrix,
        ax=axes[0, 0],
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        xticklabels=group_labels,
        yticklabels=class_labels,
        cbar_kws={
            "label": "Supermodule flux mean (group z-score)",
            "shrink": 0.68,
        },
        linewidths=0.25,
        linecolor="white",
    )
    axes[0, 0].tick_params(axis="x", labelsize=5.8, rotation=0)
    axes[0, 0].tick_params(axis="y", labelsize=5.8, rotation=0)

    distributions = report.source_data["Fig4b_top_class_distributions"].copy()
    distributions["group_label"] = distributions["dataset"].astype(str) + "\n" + (
        distributions["cluster"].map(
            cluster_display_label_map(
                sorted(distributions["cluster"].astype(str).unique())
            )
        )
    )
    distributions["supermodule_label"] = distributions["supermodule_id"].map(
        dict(zip(report.class_ids, class_labels, strict=True))
    )
    top_ids = report.effects["supermodule_id"].head(6).tolist()
    distributions["group_label"] = distributions["group_label"].str.replace(
        "\n", " ", regex=False
    )
    fig.canvas.draw()
    parent_bbox = axes[0, 1].get_position()
    horizontal_gap = parent_bbox.width * 0.08
    top_margin = parent_bbox.height * 0.08
    bottom_margin = parent_bbox.height * 0.08
    vertical_gap = parent_bbox.height * 0.06
    mini_width = (parent_bbox.width - horizontal_gap) / 2
    mini_height = (
        parent_bbox.height
        - top_margin
        - bottom_margin
        - 2 * vertical_gap
    ) / 3
    top = parent_bbox.y1 - top_margin
    axes[0, 1].set_axis_off()
    for index, class_id in enumerate(top_ids):
        row, column = divmod(index, 2)
        mini = fig.add_axes(
            [
                parent_bbox.x0 + column * (mini_width + horizontal_gap),
                top
                - row * (mini_height + vertical_gap)
                - mini_height,
                mini_width,
                mini_height,
            ]
        )
        mini.set_in_layout(False)
        subset = distributions[
            distributions["supermodule_id"].astype(int) == int(class_id)
        ]
        maximum = float(np.nanmax(np.abs(subset["mean_flux"].to_numpy())))
        exponent = int(np.floor(np.log10(maximum))) if maximum > 0 else 0
        scale = 10.0**exponent
        subset = subset.assign(mean_flux=subset["mean_flux"] / scale)
        sns.boxplot(
            data=subset,
            x="group_label",
            y="mean_flux",
            color=OKABE[index % len(OKABE)],
            showfliers=False,
            linewidth=0.55,
            width=0.72,
            ax=mini,
        )
        scale_note = "" if scale == 1.0 else f"\n($\\times10^{{{exponent}}}$)"
        mini.set_title(
            f"SM{int(class_id):02d}{scale_note}", fontsize=7, pad=1.5
        )
        mini.set_xlabel("")
        mini.set_ylabel(
            "Supermodule mean flux" if index % 2 == 0 else "", fontsize=6
        )
        mini.tick_params(axis="x", rotation=90, labelsize=5.8)
        mini.tick_params(axis="y", labelsize=6)
        mini.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        mini.yaxis.get_offset_text().set_visible(False)
        if index < 4:
            mini.tick_params(axis="x", labelbottom=False)

    correlation = report.correlation.copy()
    correlation.index = class_labels
    correlation.columns = class_labels
    sns.heatmap(
        correlation,
        ax=axes[1, 0],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Spearman rho", "shrink": 0.68},
        linewidths=0,
    )
    axes[1, 0].tick_params(axis="x", rotation=90, labelsize=6)
    axes[1, 0].tick_params(axis="y", rotation=0, labelsize=6)

    agreement = report.agreement[
        np.isfinite(report.agreement["pearson_r"])
    ].sort_values("pearson_r")
    agreement_labels = [
        f"SM{int(class_id):02d}"
        for class_id in agreement["supermodule_id"]
    ]
    y_positions = np.arange(len(agreement))
    axes[1, 1].hlines(
        y_positions,
        0,
        agreement["pearson_r"],
        color="#B0B0B0",
        linewidth=0.6,
    )
    axes[1, 1].scatter(
        agreement["pearson_r"],
        y_positions,
        marker="o",
        s=18,
        color=OKABE[0],
        label="Pearson",
        zorder=3,
    )
    axes[1, 1].scatter(
        agreement["spearman_r"],
        y_positions,
        marker="s",
        s=16,
        color=OKABE[1],
        label="Spearman",
        zorder=3,
    )
    axes[1, 1].axvline(0, color="#666666", linestyle="--", linewidth=0.6)
    axes[1, 1].set_yticks(y_positions, agreement_labels, fontsize=6)
    axes[1, 1].set_xlim(-1.0, 1.0)
    axes[1, 1].set_xlabel("Global vs class-specific correlation")
    axes[1, 1].set_ylabel("")
    axes[1, 1].legend(frameon=False, fontsize=6, loc="lower right")

    flat_axes = list(axes.flat)
    _panel_labels_above(fig, flat_axes[2:], "cd")
    flat_axes[0].set_title(
        "a", loc="left", fontsize=8, fontweight="bold", pad=2
    )
    fig.text(
        parent_bbox.x0,
        min(parent_bbox.y1 + 0.006, 0.985),
        "b",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )
    outputs = export_figure(
        fig,
        output_dir / "Fig4",
        size_inches=(7.2, 7.0),
        dpi=dpi,
    )
    plt.close(fig)
    return outputs


def plot_supp_supermodule_embedding(
    report: SupermoduleReport,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot supermodule feature embedding and global-model comparisons."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    embedding = report.source_data["SuppFig2a_class_embedding"]
    cluster_labels = sorted(embedding["cluster"].astype(str).unique())
    cluster_colors = cluster_color_map(cluster_labels)
    display_labels = cluster_display_label_map(cluster_labels)
    cluster_markers = _cluster_marker_map(cluster_labels)
    dataset_labels = sorted(embedding["dataset"].astype(str).unique())
    dataset_styles = dataset_style_map(dataset_labels)

    for axis, color_column in (
        (axes[0, 0], "cluster"),
        (axes[0, 1], "dataset"),
    ):
        if color_column == "cluster":
            for label in cluster_labels:
                subset = embedding[embedding["cluster"] == label]
                axis.scatter(
                    subset["supermodule_PC1"],
                    subset["supermodule_PC2"],
                    color=cluster_colors[label],
                    marker=cluster_markers[label],
                    s=16,
                    alpha=0.8,
                    label=_short_cluster_label(label, display_labels),
                    linewidths=0,
                )
            axis.legend(title="Cluster", frameon=False, fontsize=5.5)
        else:
            for label in dataset_labels:
                subset = embedding[embedding["dataset"] == label]
                color, marker = dataset_styles[label]
                axis.scatter(
                    subset["supermodule_PC1"],
                    subset["supermodule_PC2"],
                    color=color,
                    marker=marker,
                    s=16,
                    alpha=0.8,
                    label=label,
                    linewidths=0,
                )
            axis.legend(title="Dataset", frameon=False, fontsize=5.5)
        axis.set_xlabel("Supermodule PC1")
        axis.set_ylabel("Supermodule PC2")

    parent = axes[1, 0]
    parent.set_axis_off()
    scatter = report.source_data["SuppFig2c_global_class_scatter"]
    class_ids = sorted(scatter["supermodule_id"].unique())
    inner = GridSpecFromSubplotSpec(
        1,
        len(class_ids),
        subplot_spec=parent.get_subplotspec(),
        wspace=0.32,
    )
    for index, class_id in enumerate(class_ids):
        mini = fig.add_subplot(inner[index])
        subset = scatter[scatter["supermodule_id"] == class_id]
        mini.scatter(
            subset["global_z"],
            subset["class_z"],
            s=10,
            color=OKABE[index % len(OKABE)],
            alpha=0.75,
            linewidths=0,
        )
        limits = [
            min(subset["global_z"].min(), subset["class_z"].min()),
            max(subset["global_z"].max(), subset["class_z"].max()),
        ]
        mini.plot(limits, limits, linestyle="--", color="#777777", linewidth=0.6)
        mini.set_title(f"SM{int(class_id):02d}", fontsize=7)
        mini.set_xlabel("Global (z)")
        if index == 0:
            mini.set_ylabel("Class (z)")
        mini.tick_params(labelsize=6)
        mini.text(
            0.04,
            0.96,
            f"r={subset['pearson_r'].iloc[0]:.2f}",
            transform=mini.transAxes,
            fontsize=5.5,
            va="top",
        )

    effects = report.effects.head(12).iloc[::-1]
    axes[1, 1].barh(
        [f"SM{int(value):02d}" for value in effects["supermodule_id"]],
        effects["effect_ratio"],
        color=OKABE[0],
        edgecolor="black",
        linewidth=0.35,
    )
    axes[1, 1].set_xlabel("Between-group variance ratio")
    axes[1, 1].set_ylabel("")
    axes[1, 1].tick_params(axis="y", labelsize=6)
    _panel_labels(fig, axes.flat, "abcd")
    outputs = export_figure(
        fig,
        output_dir / "Supp_Fig2_supermodules",
        size_inches=(7.2, 6.2),
        dpi=dpi,
    )
    plt.close(fig)
    return outputs
