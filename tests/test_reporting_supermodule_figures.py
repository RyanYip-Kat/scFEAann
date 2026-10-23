from pathlib import Path

import pandas as pd

from scFEAann.reporting.figures import (
    plot_figure4_supermodules,
    plot_supp_supermodule_embedding,
)
from scFEAann.reporting.supermodules import (
    extract_supermodule_report,
    write_supermodule_tables,
)


def test_supermodule_figures_export_all_formats(
    tmp_path, supermodule_adata
) -> None:
    report = extract_supermodule_report(supermodule_adata)

    figure4 = plot_figure4_supermodules(report, tmp_path, dpi=150)
    supplement = plot_supp_supermodule_embedding(
        report, tmp_path, dpi=150
    )

    assert {
        "Fig4.pdf",
        "Fig4.svg",
        "Fig4.png",
        "Fig4_grayscale.png",
    } <= {path.name for path in figure4}
    assert {
        "Supp_Fig2_supermodules.pdf",
        "Supp_Fig2_supermodules.svg",
        "Supp_Fig2_supermodules.png",
        "Supp_Fig2_supermodules_grayscale.png",
    } <= {path.name for path in supplement}


def test_supermodule_tables_and_network_exports(
    tmp_path, supermodule_adata
) -> None:
    report = extract_supermodule_report(supermodule_adata)

    outputs = write_supermodule_tables(report, tmp_path)

    workbook = tmp_path / "report_tables.xlsx"
    sheets = pd.ExcelFile(workbook).sheet_names
    assert {
        "supermodule_group_summary",
        "supermodule_effects",
        "supermodule_model_agreement",
        "supermodule_correlation",
        "supermodule_annotation",
    } <= set(sheets)
    assert outputs["node_features_by_cell"].exists()
    assert outputs["node_features_by_group"].exists()
    node_cells = pd.read_csv(outputs["node_features_by_cell"], index_col=0)
    assert node_cells.shape[0] == supermodule_adata.n_obs
    assert "SM01_mean_flux" in node_cells.columns
    assert "SM01_sum_flux" in node_cells.columns


def test_supermodule_tables_write_figure_source_data(
    tmp_path, supermodule_adata
) -> None:
    report = extract_supermodule_report(supermodule_adata)

    outputs = write_supermodule_tables(report, tmp_path)

    source_dir = Path(tmp_path) / "figure_source_data"
    assert outputs["source_data"]["Fig4a_group_heatmap"].exists()
    for name in report.source_data:
        assert (source_dir / f"{name}.csv").exists()
