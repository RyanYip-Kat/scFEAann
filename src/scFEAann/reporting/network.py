"""Network construction and tables for scFEA metabolic graphs."""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.cluster import KMeans, SpectralClustering

from ..modules import get_module_annotation
from ..supermodules import load_supermodules
from .figures import export_figure
from .style import apply_report_style, CLUSTER_COLORS


@dataclass
class FactorGraphData:
    graph: nx.Graph
    nodes: pd.DataFrame
    edges: pd.DataFrame
    layout: pd.DataFrame


@dataclass
class SupermoduleNetworkData:
    graph: nx.Graph
    nodes: pd.DataFrame
    edges: pd.DataFrame
    layout: pd.DataFrame
    contrast_reference: str
    contrast_target: str
    contrast_key: str


def _split_annotation_names(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [
        token.strip()
        for token in str(value).split(" + ")
        if token.strip()
    ]


def _split_kegg_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [
        token.strip()
        for token in str(value).split("+")
        if token.strip() and token.strip().lower() != "nan"
    ]


def build_factor_graph(
    annotation: pd.DataFrame | None = None,
) -> FactorGraphData:
    """Build the undirected module-compound factor graph."""
    annotation = (
        get_module_annotation() if annotation is None else annotation
    )
    graph = nx.Graph()
    supermodule_names = load_supermodules()["Supermodule class"].to_dict()
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for module_id, row in annotation.iterrows():
        module_node = f"module:{module_id}"
        graph.add_node(
            module_node,
            node_type="module",
            label=str(module_id),
            module_id=str(module_id),
            supermodule_id=int(row["supermodule_id"]),
            supermodule_class=str(
                supermodule_names.get(int(row["supermodule_id"]), "")
            ),
        )
        node_rows.append(graph.nodes[module_node] | {"node_id": module_node})
        for direction, name_column, id_column, sign in (
            ("input", "input_names", "input_kegg_ids", -1),
            ("output", "output_names", "output_kegg_ids", 1),
        ):
            compound_names = _split_annotation_names(row[name_column])
            kegg_ids = _split_kegg_ids(row[id_column])
            if len(compound_names) != len(kegg_ids):
                raise ValueError(
                    f"compound name/ID mismatch for {module_id} {direction}"
                )
            for compound_name, kegg_id in zip(
                compound_names, kegg_ids, strict=True
            ):
                compound_node = f"compound:{compound_name}"
                if compound_node not in graph:
                    graph.add_node(
                        compound_node,
                        node_type="compound",
                        label=kegg_id or compound_name,
                        compound_name=compound_name,
                        kegg_id=kegg_id,
                    )
                    node_rows.append(
                        graph.nodes[compound_node] | {"node_id": compound_node}
                    )
                if graph.has_edge(module_node, compound_node):
                    existing = graph.edges[module_node, compound_node]
                    roles = set(existing["roles"].split(";"))
                    roles.add(direction)
                    existing["roles"] = ";".join(sorted(roles))
                    existing["sign"] = 0
                else:
                    graph.add_edge(
                        module_node,
                        compound_node,
                        roles=direction,
                        sign=sign,
                    )
                edge_rows.append(
                    {
                        "source": module_node,
                        "target": compound_node,
                        "module_id": str(module_id),
                        "compound_name": compound_name,
                        "compound_kegg_id": kegg_id,
                        "direction": direction,
                        "sign": sign,
                    }
                )
    nodes = pd.DataFrame(node_rows).set_index("node_id")
    nodes["degree"] = [
        graph.degree(node) for node in nodes.index
    ]
    edges = pd.DataFrame(edge_rows).drop_duplicates(
        ["source", "target", "direction"]
    )
    positions = nx.kamada_kawai_layout(graph)
    layout = pd.DataFrame(
        [
            {
                "node_id": node,
                "x": float(position[0]),
                "y": float(position[1]),
            }
            for node, position in positions.items()
        ]
    ).set_index("node_id")
    return FactorGraphData(
        graph=graph, nodes=nodes, edges=edges, layout=layout
    )


def factor_graph_label_map(
    factor_graph: FactorGraphData,
) -> dict[str, str]:
    """Return one in-node label for every factor-graph node."""
    return {
        node: str(attributes.get("label", node))
        for node, attributes in factor_graph.graph.nodes(data=True)
    }


def factor_graph_supermodule_annotations(
    factor_graph: FactorGraphData,
) -> pd.DataFrame:
    """Return perimeter label positions and module centroids per class."""
    modules = factor_graph.nodes[
        factor_graph.nodes["node_type"] == "module"
    ]
    provisional: list[dict[str, object]] = []
    for supermodule_id, group in modules.groupby(
        "supermodule_id", sort=True
    ):
        node_ids = group.index.tolist()
        positions = factor_graph.layout.loc[node_ids, ["x", "y"]]
        class_name = str(group["supermodule_class"].iloc[0])
        readable_name = class_name.replace("+", " + ").replace(
            "metabolism/synthesis", "metabolism/\nsynthesis"
        )
        provisional.append(
            {
                "supermodule_id": int(supermodule_id),
                "supermodule_class": class_name,
                "n_modules": len(node_ids),
                "anchor_x": float(positions["x"].mean()),
                "anchor_y": float(positions["y"].mean()),
                "label": textwrap.fill(readable_name, width=20),
            }
        )
    annotation = pd.DataFrame(provisional)
    center_x = float(annotation["anchor_x"].mean())
    center_y = float(annotation["anchor_y"].mean())
    radii = np.hypot(
        annotation["anchor_x"] - center_x,
        annotation["anchor_y"] - center_y,
    )
    base_radius = float(radii.quantile(0.9))
    angles = np.arctan2(
        annotation["anchor_y"] - center_y,
        annotation["anchor_x"] - center_x,
    )
    annotation["_angle"] = angles
    annotation = annotation.sort_values("_angle").reset_index(drop=True)
    count = len(annotation)
    placed_angles = (
        np.arange(count, dtype=float) * (2 * np.pi / max(count, 1))
        - np.pi / 2
    )
    annotation["label_x"] = center_x + np.cos(placed_angles) * base_radius * 1.18
    annotation["label_y"] = center_y + np.sin(placed_angles) * base_radius * 1.18
    annotation.loc[
        np.arange(1, count, 2), "label_x"
    ] = (
        center_x
        + np.cos(placed_angles[1::2])
        * base_radius
        * 1.34
    )
    annotation.loc[
        np.arange(1, count, 2), "label_y"
    ] = (
        center_y
        + np.sin(placed_angles[1::2])
        * base_radius
        * 1.34
    )
    return annotation.drop(columns="_angle")


def _topology_clusters(
    graph: nx.Graph,
    *,
    n_clusters: int = 5,
) -> dict[str, int]:
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return {nodes[0]: 0} if nodes else {}
    n_clusters = max(1, min(n_clusters, len(nodes) - 1))
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
    adjacency = np.maximum(adjacency, adjacency.T)
    try:
        labels = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=16,
        ).fit_predict(adjacency)
    except Exception:
        labels = KMeans(
            n_clusters=n_clusters, random_state=16, n_init=20
        ).fit_predict(adjacency)
    return {
        node: int(label) + 1
        for node, label in zip(nodes, labels, strict=True)
    }


def build_supermodule_network(
    adata: ad.AnnData,
    *,
    contrast_key: str = "dataset",
    contrast: tuple[str, str] | None = None,
) -> SupermoduleNetworkData:
    """Build the class-level network and contrast node attributes."""
    required = {
        "scFEA_supermodule_flux_mean",
        "scFEA_supermodule_flux_sum",
    }
    missing = required - set(adata.obsm)
    if missing:
        raise ValueError(f"supermodule results are missing: {sorted(missing)}")
    metadata = adata.uns.get("scFEAann", {}).get("supermodules", {})
    if not metadata:
        raise ValueError("supermodule metadata is missing from adata.uns")
    if contrast_key not in adata.obs:
        raise ValueError(f"contrast column '{contrast_key}' is missing")
    contrast_values = adata.obs[contrast_key].astype(str)
    available = sorted(contrast_values.unique())
    if contrast is None:
        if len(available) < 2:
            raise ValueError("at least two contrast levels are required")
        contrast = (available[0], available[1])
    if len(contrast) != 2 or contrast[0] not in available or contrast[1] not in available:
        raise ValueError(
            f"contrast must contain two values from {available}; got {contrast}"
        )
    reference, target = contrast
    class_ids = [int(value) for value in metadata["class_ids"]]
    class_names = [str(value) for value in metadata["class_names"]]
    annotation = get_module_annotation()
    class_compounds: dict[int, set[str]] = {}
    class_modules: dict[int, list[str]] = {}
    for class_id in class_ids:
        rows = annotation[annotation["supermodule_id"] == class_id]
        class_modules[class_id] = rows.index.astype(str).tolist()
        compounds: set[str] = set()
        for _, row in rows.iterrows():
            compounds.update(_split_annotation_names(row["input_names"]))
            compounds.update(_split_annotation_names(row["output_names"]))
        class_compounds[class_id] = compounds

    graph = nx.Graph()
    for class_id, class_name in zip(class_ids, class_names, strict=True):
        graph.add_node(
            class_id,
            label=f"SM{class_id:02d}",
            supermodule_class=class_name,
            n_modules=len(class_modules[class_id]),
        )
    edge_rows: list[dict[str, object]] = []
    for first, second in combinations(class_ids, 2):
        shared = sorted(
            class_compounds[first] & class_compounds[second]
        )
        if not shared:
            continue
        graph.add_edge(
            first,
            second,
            weight=len(shared),
            shared_compounds=";".join(shared),
        )
        edge_rows.append(
            {
                "source_class_id": first,
                "target_class_id": second,
                "source_label": f"SM{first:02d}",
                "target_label": f"SM{second:02d}",
                "weight": len(shared),
                "shared_compounds": ";".join(shared),
            }
        )
    topology_labels = _topology_clusters(graph)
    node_mean = pd.DataFrame(
        np.asarray(adata.obsm["scFEA_supermodule_flux_mean"], dtype=float),
        index=adata.obs_names,
        columns=class_ids,
    )
    node_sum = pd.DataFrame(
        np.asarray(adata.obsm["scFEA_supermodule_flux_sum"], dtype=float),
        index=adata.obs_names,
        columns=class_ids,
    )
    mask_reference = contrast_values.to_numpy() == reference
    mask_target = contrast_values.to_numpy() == target
    node_rows: list[dict[str, object]] = []
    for slot, class_id in enumerate(class_ids):
        mean_values = node_mean.iloc[:, slot].to_numpy()
        reference_values = mean_values[mask_reference]
        target_values = mean_values[mask_target]
        reference_mean = float(np.mean(reference_values))
        target_mean = float(np.mean(target_values))
        difference = target_mean - reference_mean
        pooled_standard_deviation = math.sqrt(
            (
                np.var(reference_values, ddof=1)
                * max(len(reference_values) - 1, 0)
                + np.var(target_values, ddof=1)
                * max(len(target_values) - 1, 0)
            )
            / max(len(reference_values) + len(target_values) - 2, 1)
        )
        effect_z = (
            difference / pooled_standard_deviation
            if pooled_standard_deviation > 0
            else 0.0
        )
        row = {
            "supermodule_id": class_id,
            "class_label": f"SM{class_id:02d}",
            "supermodule_class": class_names[slot],
            "n_modules": len(class_modules[class_id]),
            "topology_cluster": topology_labels[class_id],
            "mean_flux": float(np.mean(mean_values)),
            "mean_absolute_flux": float(np.mean(np.abs(mean_values))),
            "sum_flux": float(node_sum.iloc[:, slot].mean()),
            "contrast_key": contrast_key,
            "contrast_reference": reference,
            "contrast_target": target,
            "reference_mean": reference_mean,
            "target_mean": target_mean,
            "effect_difference": difference,
            "effect_z": float(effect_z),
            "degree": graph.degree(class_id, weight="weight"),
        }
        node_rows.append(row)
        for key in (
            "topology_cluster",
            "effect_z",
            "effect_difference",
            "mean_flux",
        ):
            graph.nodes[class_id][key] = row[key]
    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(
        edge_rows,
        columns=[
            "source_class_id",
            "target_class_id",
            "source_label",
            "target_label",
            "weight",
            "shared_compounds",
        ],
    )
    positions = nx.kamada_kawai_layout(graph, weight="weight")
    layout = pd.DataFrame(
        [
            {
                "supermodule_id": class_id,
                "x": float(position[0]),
                "y": float(position[1]),
            }
            for class_id, position in positions.items()
        ]
    ).set_index("supermodule_id")
    return SupermoduleNetworkData(
        graph=graph,
        nodes=nodes,
        edges=edges,
        layout=layout,
        contrast_reference=reference,
        contrast_target=target,
        contrast_key=contrast_key,
    )


def write_network_tables(
    network: SupermoduleNetworkData,
    factor_graph: FactorGraphData,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export node, edge, and layout tables for both network figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "network_nodes": output_dir / "network_nodes.csv",
        "network_edges": output_dir / "network_edges.csv",
        "network_layout": output_dir / "network_layout.csv",
        "network_supermodule_summary": (
            output_dir / "network_supermodule_summary.csv"
        ),
        "factor_graph_nodes": output_dir / "factor_graph_nodes.csv",
        "factor_graph_edges": output_dir / "factor_graph_edges.csv",
        "factor_graph_layout": output_dir / "factor_graph_layout.csv",
        "factor_graph_supermodule_labels": (
            output_dir / "factor_graph_supermodule_labels.csv"
        ),
    }
    network.nodes.to_csv(paths["network_nodes"], index=False)
    network.edges.to_csv(paths["network_edges"], index=False)
    network.layout.reset_index().to_csv(
        paths["network_layout"], index=False
    )
    network.nodes.to_csv(
        paths["network_supermodule_summary"], index=False
    )
    factor_graph.nodes.reset_index().to_csv(
        paths["factor_graph_nodes"], index=False
    )
    factor_graph.edges.to_csv(paths["factor_graph_edges"], index=False)
    factor_graph.layout.reset_index().to_csv(
        paths["factor_graph_layout"], index=False
    )
    factor_graph_supermodule_annotations(factor_graph).to_csv(
        paths["factor_graph_supermodule_labels"], index=False
    )
    return paths


def _network_positions(layout: pd.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(node_id): np.asarray([row["x"], row["y"]], dtype=float)
        for node_id, row in layout.iterrows()
    }


def plot_supermodule_network(
    network: SupermoduleNetworkData,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot topology clusters and flux-effect views of the class network."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
    positions = _network_positions(network.layout)
    node_ids = network.nodes["supermodule_id"].astype(int).tolist()
    sizes = [
        360 + 30 * math.sqrt(max(count, 1))
        for count in network.nodes["n_modules"]
    ]
    edge_widths = [
        0.35 + 0.45 * math.log1p(weight)
        for weight in network.edges["weight"]
    ]
    segments = [
        [positions[int(row.source_class_id)], positions[int(row.target_class_id)]]
        for row in network.edges.itertuples(index=False)
    ]

    cluster_values = network.nodes["topology_cluster"].astype(int)
    cluster_ids = sorted(cluster_values.unique())
    palette = CLUSTER_COLORS["default"]
    cluster_colors = {
        cluster: palette[index % len(palette)]
        for index, cluster in enumerate(cluster_ids)
    }
    effects = network.nodes["effect_z"].to_numpy(dtype=float)
    effect_limit = max(
        0.5,
        float(np.nanpercentile(np.abs(effects), 95)),
    )
    norm = Normalize(vmin=-effect_limit, vmax=effect_limit)
    cmap = plt.get_cmap("RdBu_r")
    effect_colors = cmap(norm(effects))

    titles = (
        "Supermodule topology clusters",
        f"Flux effect: {network.contrast_target} vs "
        f"{network.contrast_reference}",
    )
    for axis, title in zip(axes, titles, strict=True):
        axis.add_collection(
            LineCollection(
                segments,
                colors="#B0B0B0",
                linewidths=edge_widths,
                alpha=0.7,
                zorder=1,
            )
        )
        axis.set_title(title, fontsize=8)
        axis.set_axis_off()

    size_by_node = dict(zip(node_ids, sizes, strict=True))
    cluster_markers = ["o", "s", "^", "D", "P"]
    for index, cluster in enumerate(cluster_ids):
        cluster_nodes = network.nodes.loc[
            cluster_values == cluster, "supermodule_id"
        ].astype(int).tolist()
        nx.draw_networkx_nodes(
            network.graph,
            positions,
            nodelist=cluster_nodes,
            node_size=[size_by_node[node] for node in cluster_nodes],
            node_color=cluster_colors[cluster],
            node_shape=cluster_markers[index % len(cluster_markers)],
            edgecolors="black",
            linewidths=0.45,
            ax=axes[0],
        )
    nx.draw_networkx_nodes(
        network.graph,
        positions,
        nodelist=node_ids,
        node_size=sizes,
        node_color=effect_colors,
        edgecolors="black",
        linewidths=0.45,
        ax=axes[1],
    )
    labels = {
        node: network.graph.nodes[node]["label"]
        for node in node_ids
    }
    for axis in axes:
        nx.draw_networkx_labels(
            network.graph,
            positions,
            labels=labels,
            font_size=5.5,
            font_color="black",
            ax=axis,
        )

    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker=cluster_markers[index % len(cluster_markers)],
            linestyle="",
            markerfacecolor=cluster_colors[cluster],
            markeredgecolor="black",
            markersize=6,
            label=f"Topology cluster {cluster}",
        )
        for index, cluster in enumerate(cluster_ids)
    ]
    axes[0].legend(
        handles=cluster_handles,
        frameon=False,
        fontsize=5.5,
        loc="upper left",
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[1],
        shrink=0.72,
        pad=0.02,
    )
    colorbar.set_label("Standardized flux difference", fontsize=6)
    colorbar.ax.tick_params(labelsize=6)
    fig.text(
        0.005,
        0.995,
        "a",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.51,
        0.995,
        "b",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )
    outputs = export_figure(
        fig,
        output_dir / "Fig5",
        size_inches=(7.2, 4.2),
        dpi=dpi,
    )
    plt.close(fig)
    return outputs


def plot_factor_graph(
    factor_graph: FactorGraphData,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Plot the full module-metabolite factor graph."""
    apply_report_style()
    output_dir = Path(output_dir)
    fig, axis = plt.subplots(figsize=(7.2, 7.0))
    positions = {
        node_id: np.asarray([row["x"], row["y"]], dtype=float)
        for node_id, row in factor_graph.layout.iterrows()
    }
    nx.draw_networkx_edges(
        factor_graph.graph,
        positions,
        edge_color="#B8B8B8",
        width=0.35,
        alpha=0.65,
        ax=axis,
    )
    modules = factor_graph.nodes[
        factor_graph.nodes["node_type"] == "module"
    ]
    compounds = factor_graph.nodes[
        factor_graph.nodes["node_type"] == "compound"
    ]
    module_sizes = 70 + 14 * modules["degree"].to_numpy()
    compound_sizes = 70 + 14 * compounds["degree"].to_numpy()
    nx.draw_networkx_nodes(
        factor_graph.graph,
        positions,
        nodelist=modules.index.tolist(),
        node_size=module_sizes,
        node_color="#66BB55",
        node_shape="D",
        edgecolors="black",
        linewidths=0.25,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        factor_graph.graph,
        positions,
        nodelist=compounds.index.tolist(),
        node_size=compound_sizes,
        node_color="#F4A6A6",
        node_shape="o",
        edgecolors="black",
        linewidths=0.25,
        ax=axis,
    )
    labels = factor_graph_label_map(factor_graph)
    nx.draw_networkx_labels(
        factor_graph.graph,
        positions,
        labels=labels,
        font_size=4.2,
        font_color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.1},
        ax=axis,
    )
    annotations = factor_graph_supermodule_annotations(factor_graph)
    for row in annotations.itertuples(index=False):
        axis.plot(
            [row.anchor_x, row.label_x],
            [row.anchor_y, row.label_y],
            color="#173B8F",
            linewidth=0.45,
            alpha=0.35,
            zorder=4,
        )
        text = axis.text(
            row.label_x,
            row.label_y,
            row.label,
            color="#173B8F",
            fontsize=6,
            ha="center",
            va="center",
            zorder=6,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.8, foreground="white"),
                path_effects.Normal(),
            ]
        )
    axis.margins(0.16)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#F4A6A6",
                markeredgecolor="black",
                markersize=7,
                label="Metabolite",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                markerfacecolor="#66BB55",
                markeredgecolor="black",
                markersize=7,
                label="Metabolic module",
            ),
        ],
        frameon=False,
        fontsize=6,
        loc="upper left",
    )
    axis.set_title("Human metabolic factor graph", fontsize=9)
    axis.set_axis_off()
    outputs = export_figure(
        fig,
        output_dir / "Supp_Fig3_metabolic_network",
        size_inches=(7.2, 7.0),
        dpi=dpi,
    )
    plt.close(fig)
    return outputs
