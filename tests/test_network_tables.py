import numpy as np
import pytest

from scFEAann.reporting.network import (
    build_factor_graph,
    build_supermodule_network,
    factor_graph_label_map,
    factor_graph_supermodule_annotations,
)


def test_factor_graph_contains_module_compound_and_direction() -> None:
    graph_data = build_factor_graph()

    assert "module:M_48" in graph_data.nodes.index
    assert "compound:Glutamate" in graph_data.nodes.index
    edge = graph_data.edges[
        (graph_data.edges["module_id"] == "M_48")
        & (graph_data.edges["compound_name"] == "Glutamate")
    ].iloc[0]
    assert edge["direction"] == "input"
    assert edge["sign"] == -1
    assert graph_data.layout.shape[0] == graph_data.nodes.shape[0]
    assert graph_data.nodes.loc["module:M_48", "label"] == "M_48"
    assert graph_data.nodes.loc["compound:Glutamate", "label"] == "C00025"


def test_factor_graph_label_map_covers_every_node() -> None:
    graph_data = build_factor_graph()
    labels = factor_graph_label_map(graph_data)

    assert len(labels) == graph_data.nodes.shape[0]
    assert labels["module:M_48"] == "M_48"
    assert labels["compound:Glutamate"] == "C00025"


def test_factor_graph_has_22_supermodule_class_annotations() -> None:
    graph_data = build_factor_graph()

    annotations = factor_graph_supermodule_annotations(graph_data)

    assert annotations.shape[0] == 22
    assert annotations["supermodule_class"].nunique() == 22
    assert np.isfinite(
        annotations[["anchor_x", "anchor_y", "label_x", "label_y"]].to_numpy()
    ).all()


def test_factor_graph_layout_is_reproducible() -> None:
    first = build_factor_graph()
    second = build_factor_graph()

    np.testing.assert_allclose(
        first.layout[["x", "y"]].to_numpy(),
        second.layout[["x", "y"]].to_numpy(),
    )


def test_supermodule_network_contains_22_nodes_and_flux_effect(
    supermodule_adata,
) -> None:
    network = build_supermodule_network(
        supermodule_adata,
        contrast_key="dataset",
        contrast=("dataset_A", "dataset_B"),
    )

    assert network.nodes.shape[0] == 22
    assert network.edges.shape[0] > 0
    assert network.layout.shape[0] == 22
    assert network.nodes["topology_cluster"].nunique() == 5
    assert np.isfinite(network.nodes["effect_z"]).all()
    class8 = network.nodes[network.nodes["supermodule_id"] == 8].iloc[0]
    assert class8["contrast_reference"] == "dataset_A"
    assert class8["contrast_target"] == "dataset_B"


def test_supermodule_network_rejects_unknown_contrast_label(
    supermodule_adata,
) -> None:
    with pytest.raises(ValueError, match="contrast"):
        build_supermodule_network(
            supermodule_adata,
            contrast_key="dataset",
            contrast=("missing", "dataset_B"),
        )
