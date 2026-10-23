import json

import pandas as pd

from scFEAann.reporting.report import export_scfeaann_report


def test_report_adds_network_figures_and_tables(
    tmp_path, supermodule_adata
) -> None:
    manifest = export_scfeaann_report(
        supermodule_adata,
        tmp_path,
        embedding_method="pca",
        dpi=150,
        network_contrast_key="dataset",
        network_contrast=("dataset_A", "dataset_B"),
    )

    assert (tmp_path / "Fig5.pdf").exists()
    assert (tmp_path / "Fig5.png").exists()
    assert (tmp_path / "Supp_Fig3_metabolic_network.pdf").exists()
    assert (tmp_path / "network_nodes.csv").exists()
    assert (tmp_path / "network_edges.csv").exists()
    assert (tmp_path / "factor_graph_edges.csv").exists()
    assert pd.read_csv(tmp_path / "network_nodes.csv").shape[0] == 22
    assert "networks" in manifest
    assert manifest["networks"]["contrast_reference"] == "dataset_A"
    assert manifest["networks"]["contrast_target"] == "dataset_B"
    assert "Fig5" in manifest["figures"]
    assert "network_nodes" in manifest["tables"]

    stored = json.loads((tmp_path / "report_manifest.json").read_text())
    assert stored["networks"]["n_supermodule_edges"] > 0
    legends = (tmp_path / "figure_legends.md").read_text()
    assert "Figure 5. Supermodule network" in legends
    assert "Supplementary Figure 3. Human metabolic factor graph" in legends
