import numpy as np
import pandas as pd

from scFEAann.reporting.exports import (
    build_figure_source_data,
    write_cell_tables,
    write_figure_source_data,
    write_summary_tables,
)
from scFEAann.reporting.embedding import compute_flux_embedding
from scFEAann.reporting.tables import extract_report_results


def test_flux_embedding_pca_returns_two_coordinates() -> None:
    flux = np.arange(120, dtype=float).reshape(20, 6)

    coordinates, metadata = compute_flux_embedding(flux, method="pca")

    assert coordinates.shape == (20, 2)
    assert metadata["method"] == "pca"
    assert metadata["standardized"] is True


def test_summary_tables_write_expected_workbook_sheets(
    tmp_path, reporting_adata
) -> None:
    report = extract_report_results(reporting_adata, top_n=2)

    outputs = write_summary_tables(report, tmp_path)

    workbook = tmp_path / "report_tables.xlsx"
    assert workbook.exists()
    assert "module_group_summary" in pd.ExcelFile(workbook).sheet_names
    assert outputs["workbook"] == workbook


def test_cell_tables_write_compressed_matrices(
    tmp_path, reporting_adata
) -> None:
    report = extract_report_results(reporting_adata, top_n=2)

    outputs = write_cell_tables(report, tmp_path)

    loaded_metadata = pd.read_csv(outputs["cell_metadata"], index_col=0)
    loaded_flux = pd.read_csv(outputs["cell_flux"], index_col=0)
    assert len(loaded_metadata) == reporting_adata.n_obs
    assert loaded_flux.shape == reporting_adata.obsm["scFEA_flux"].shape


def test_figure_source_data_contains_all_panels(tmp_path, reporting_adata) -> None:
    report = extract_report_results(reporting_adata, top_n=2)
    coordinates, _ = compute_flux_embedding(
        reporting_adata.obsm["scFEA_flux"], method="pca"
    )
    source_data = build_figure_source_data(
        report,
        transcriptome_embedding=np.asarray(reporting_adata.obsm["X_umap"]),
        flux_embedding=coordinates,
    )

    outputs = write_figure_source_data(source_data, tmp_path)

    expected = {
        "Fig1a_group_counts",
        "Fig1b_dataset_consistency",
        "Fig1c_module_heatmap",
        "Fig1d_top_modules",
        "Fig2a_balance_heatmap",
        "Fig2b_top_compounds",
        "Fig2c_module_effects",
        "Fig2d_compound_effects",
        "Fig3a_transcriptome_embedding",
        "Fig3b_flux_embedding",
        "Supp_loss_history",
    }
    assert expected == set(outputs)
