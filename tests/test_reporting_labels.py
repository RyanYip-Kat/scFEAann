from scFEAann.reporting.figures import cluster_display_label_map
from scFEAann.reporting.report import export_scfeaann_report


def test_known_cluster_names_use_compact_publication_labels() -> None:
    mapping = cluster_display_label_map(
        [
            "corneal basal epithelium",
            "corneal superficial epithelium",
            "corneal suprabasal epithelial cells",
        ]
    )

    assert list(mapping.values()) == ["Basal", "Superficial", "Suprabasal"]


def test_arbitrary_cluster_names_remain_unique() -> None:
    labels = [
        "cell_type_1",
        "cell_type_2",
        "cell_type_3",
        "a_very_long_cell_type_name_alpha",
        "a_very_long_cell_type_name_beta",
    ]

    mapping = cluster_display_label_map(labels)

    assert len(set(mapping.values())) == len(labels)
    assert mapping["cell_type_1"] == "cell_type_1"
    assert mapping["cell_type_2"] == "cell_type_2"


def test_report_runs_with_arbitrary_dataset_and_cluster_names(
    tmp_path, reporting_adata
) -> None:
    adata = reporting_adata.copy()
    adata.obs["dataset"] = adata.obs["dataset"].map(
        {"D1": "batch_alpha", "D2": "batch_beta"}
    )
    cluster_mapping = {
        label: f"cell_type_{index + 1}"
        for index, label in enumerate(sorted(adata.obs["cluster"].unique()))
    }
    adata.obs["cluster"] = adata.obs["cluster"].map(cluster_mapping)
    adata.obs["dataset_cluster"] = (
        adata.obs["dataset"].astype(str)
        + " | "
        + adata.obs["cluster"].astype(str)
    )

    manifest = export_scfeaann_report(
        adata,
        tmp_path,
        embedding_method="pca",
        dpi=150,
    )

    assert manifest["grouping"]["n_datasets"] == 2
    assert manifest["grouping"]["n_clusters"] == 2
    assert (tmp_path / "Fig1.png").exists()
    assert (tmp_path / "Fig2.png").exists()
