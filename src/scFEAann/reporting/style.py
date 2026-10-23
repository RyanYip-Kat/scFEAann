"""English SCI figure style for scFEAann reports."""

from __future__ import annotations

import warnings

import matplotlib as mpl
import seaborn as sns
from matplotlib import MatplotlibDeprecationWarning


CLUSTER_COLORS = {
    "default": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"],
    "corneal basal epithelium": "#0072B2",
    "corneal superficial epithelium": "#D55E00",
    "corneal suprabasal epithelial cells": "#009E73",
}
DATASET_COLORS = {
    "default": ["#0072B2", "#D55E00", "#009E73", "#CC79A7"],
}
DATASET_MARKERS = ["o", "s", "^", "D"]


def apply_report_style() -> None:
    """Apply compact, colorblind-safe, English-publication matplotlib settings."""
    warnings.filterwarnings(
        "ignore",
        message="The set_bad function will be deprecated.*",
        category=PendingDeprecationWarning,
        module=r"seaborn\.matrix",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"vert: bool was deprecated.*",
        category=MatplotlibDeprecationWarning,
        module=r"seaborn\.categorical",
    )
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "figure.titlesize": 9,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def cluster_color_map(labels: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    fallback = CLUSTER_COLORS["default"]
    for index, label in enumerate(labels):
        mapping[label] = CLUSTER_COLORS.get(label, fallback[index % len(fallback)])
    return mapping


def dataset_style_map(labels: list[str]) -> dict[str, tuple[str, str]]:
    fallback = DATASET_COLORS["default"]
    return {
        label: (
            fallback[index % len(fallback)],
            DATASET_MARKERS[index % len(DATASET_MARKERS)],
        )
        for index, label in enumerate(labels)
    }
