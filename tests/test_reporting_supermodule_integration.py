import json

import pandas as pd

from scFEAann.reporting.report import export_scfeaann_report


def test_report_detects_supermodule_results(
    tmp_path, supermodule_adata
) -> None:
    manifest = export_scfeaann_report(
        supermodule_adata,
        tmp_path,
        embedding_method="pca",
        dpi=150,
    )

    assert (tmp_path / "Fig4.pdf").exists()
    assert (tmp_path / "Fig4.png").exists()
    assert (tmp_path / "Supp_Fig2_supermodules.pdf").exists()
    sheets = pd.ExcelFile(tmp_path / "report_tables.xlsx").sheet_names
    assert {
        "supermodule_group_summary",
        "supermodule_effects",
        "supermodule_model_agreement",
        "supermodule_correlation",
        "supermodule_annotation",
    } <= set(sheets)
    assert (tmp_path / "supermodule_node_features_by_cell.csv.gz").exists()
    assert (tmp_path / "supermodule_node_features_by_group.csv").exists()
    assert "supermodule_report" in manifest
    assert "Fig4" in manifest["figures"]
    assert "supermodule_node_features_by_cell" in manifest["tables"]

    stored = json.loads((tmp_path / "report_manifest.json").read_text())
    assert stored["supermodule_report"]["n_classes"] == 22
    legends = (tmp_path / "figure_legends.md").read_text()
    assert "Figure 4. Supermodule" in legends
