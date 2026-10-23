import json

from scFEAann.reporting.report import export_scfeaann_report


def test_report_exports_tables_figures_and_manifest(
    tmp_path, reporting_adata
) -> None:
    manifest = export_scfeaann_report(
        reporting_adata,
        tmp_path,
        embedding_method="pca",
        dpi=150,
    )

    assert (tmp_path / "report_tables.xlsx").exists()
    assert (tmp_path / "report_manifest.json").exists()
    assert (tmp_path / "Fig1.pdf").exists()
    assert (tmp_path / "Fig2.pdf").exists()
    assert (tmp_path / "Fig3.pdf").exists()
    assert (tmp_path / "Supp_Fig_loss.pdf").exists()
    assert (tmp_path / "figure_legends.md").exists()
    assert len(list((tmp_path / "figure_source_data").glob("*.csv"))) == 11
    assert manifest["grouping"]["dataset_key"] == "dataset"
    assert manifest["grouping"]["cluster_key"] == "cluster"
    assert manifest["embeddings"]["flux"]["method"] == "pca"

    stored = json.loads((tmp_path / "report_manifest.json").read_text())
    assert stored["input"]["n_obs"] == reporting_adata.n_obs
    assert "report_tables.xlsx" in stored["tables"]["workbook"]
    legends = (tmp_path / "figure_legends.md").read_text()
    assert "median" in legends
    assert "No cell-level p-values" in legends


def test_report_does_not_overwrite_input_h5ad_contents(
    tmp_path, reporting_adata
) -> None:
    input_path = tmp_path / "input.h5ad"
    reporting_adata.write_h5ad(input_path)
    original_bytes = input_path.stat().st_size

    export_scfeaann_report(
        reporting_adata,
        tmp_path / "report",
        embedding_method="pca",
        dpi=150,
    )

    assert input_path.stat().st_size == original_bytes
