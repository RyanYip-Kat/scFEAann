import pandas as pd

from scFEAann.reporting.network import (
    build_factor_graph,
    build_supermodule_network,
    plot_factor_graph,
    plot_supermodule_network,
    write_network_tables,
)


def test_supermodule_network_exports_all_formats(
    tmp_path, supermodule_adata
) -> None:
    network = build_supermodule_network(supermodule_adata)

    paths = plot_supermodule_network(network, tmp_path, dpi=150)

    assert {
        "Fig5.pdf",
        "Fig5.svg",
        "Fig5.png",
        "Fig5_grayscale.png",
    } <= {path.name for path in paths}


def test_factor_graph_exports_all_formats(tmp_path) -> None:
    factor = build_factor_graph()

    paths = plot_factor_graph(factor, tmp_path, dpi=150)

    assert {
        "Supp_Fig3_metabolic_network.pdf",
        "Supp_Fig3_metabolic_network.svg",
        "Supp_Fig3_metabolic_network.png",
        "Supp_Fig3_metabolic_network_grayscale.png",
    } <= {path.name for path in paths}


def test_network_tables_export_nodes_edges_and_layouts(
    tmp_path, supermodule_adata
) -> None:
    network = build_supermodule_network(supermodule_adata)
    factor = build_factor_graph()

    outputs = write_network_tables(network, factor, tmp_path)

    assert pd.read_csv(outputs["network_nodes"]).shape[0] == 22
    assert pd.read_csv(outputs["network_edges"]).shape[0] > 0
    assert pd.read_csv(outputs["network_layout"]).shape[0] == 22
    assert pd.read_csv(outputs["factor_graph_nodes"]).shape[0] == (
        factor.nodes.shape[0]
    )
    assert pd.read_csv(outputs["factor_graph_edges"]).shape[0] == (
        factor.edges.shape[0]
    )
    labels = pd.read_csv(outputs["factor_graph_supermodule_labels"])
    assert labels.shape[0] == 22
    assert {"anchor_x", "anchor_y", "label_x", "label_y"} <= set(labels.columns)
