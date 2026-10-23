import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def reporting_adata() -> ad.AnnData:
    datasets = ["D1"] * 6 + ["D2"] * 6
    clusters = (["A"] * 3 + ["B"] * 3) * 2
    obs = pd.DataFrame(
        {
            "dataset": datasets,
            "cluster": clusters,
            "dataset_cluster": [
                f"{dataset} | {cluster}"
                for dataset, cluster in zip(datasets, clusters, strict=True)
            ],
        },
        index=[f"cell_{index}" for index in range(12)],
    )
    adata = ad.AnnData(
        np.ones((12, 4), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    adata.obsm["X_umap"] = np.column_stack(
        [
            np.linspace(-2, 2, 12),
            np.tile([-1.0, 1.0], 6),
        ]
    )
    adata.obsm["scFEA_flux"] = np.column_stack(
        [
            np.tile([10.0, 10.0, 10.0, 0.0, 0.0, 0.0], 2),
            np.tile([2.0, 2.0, 2.0, -1.0, -1.0, -1.0], 2),
            np.tile([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], 2),
            np.zeros(12),
        ]
    ).astype(np.float32)
    adata.obsm["scFEA_balance"] = np.column_stack(
        [
            np.r_[np.zeros(6), np.full(6, 4.0)],
            np.linspace(1, -1, 12),
            np.tile([0.5, -0.5], 6),
        ]
    ).astype(np.float32)
    adata.uns["scFEAann"] = {
        "group_key": "dataset_cluster",
        "expression_source": "X",
        "model": {
            "module_gene_file": "module_gene_m168.csv",
        },
        "flux": {
            "module_names": ["M_1", "M_2", "M_3", "M_4"],
            "loss_history": {
                "total": np.array([5.0, 4.0, 3.0]),
                "balance": np.array([1.0, 0.8, 0.6]),
                "negative": np.array([2.0, 1.5, 1.0]),
                "cell": np.array([1.0, 0.9, 0.8]),
                "module": np.array([1.0, 0.8, 0.6]),
            },
        },
        "balance": {
            "compound_names": ["C1", "C2", "C3"],
        },
    }
    return adata
