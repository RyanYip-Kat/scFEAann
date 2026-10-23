import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scFEAann.results import write_supermodule_results
from scFEAann.supermodules import load_supermodules


@pytest.fixture
def supermodule_adata() -> ad.AnnData:
    rng = np.random.default_rng(31)
    n_cells = 12
    datasets = ["dataset_A"] * 6 + ["dataset_B"] * 6
    clusters = ["cell_type_A"] * 3 + ["cell_type_B"] * 3
    clusters = clusters * 2
    obs = pd.DataFrame(
        {
            "dataset": datasets,
            "cluster": clusters,
            "dataset_cluster": [
                f"{dataset} | {cluster}"
                for dataset, cluster in zip(datasets, clusters, strict=True)
            ],
        },
        index=[f"cell_{index}" for index in range(n_cells)],
    )
    module_names = ["M_1", "M_48", "M_77", "M_169"]
    compound_names = ["Glutamate", "Cholesterol"]
    adata = ad.AnnData(
        rng.normal(3, 1, (n_cells, 4)).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    adata.obsm["scFEA_flux"] = rng.normal(
        0, 0.01, (n_cells, len(module_names))
    ).astype(np.float32)
    adata.obsm["scFEA_balance"] = rng.normal(
        0, 0.01, (n_cells, len(compound_names))
    ).astype(np.float32)
    class_table = load_supermodules()
    class_ids = class_table.index.astype(int).tolist()
    class_names = class_table["Supermodule class"].astype(str).tolist()
    node_mean = rng.normal(0, 1, (n_cells, 22)).astype(np.float32)
    node_mean[:, 0] += np.where(
        np.asarray(datasets) == "dataset_B", 2.0, 0.0
    )
    node_mean[:, 7] -= np.where(
        np.asarray(clusters) == "cell_type_B", 1.5, 0.0
    )
    node_sum = node_mean * rng.uniform(2, 8, size=(1, 22))
    super_flux = rng.normal(0, 0.01, (n_cells, 4)).astype(np.float32)
    super_balance = rng.normal(
        0, 0.01, (n_cells, 22 * len(compound_names))
    ).astype(np.float32)
    class_results = {
        str(class_id): {
            "status": "success",
            "module_names": [
                module
                for module, module_class in {
                    "M_1": 1,
                    "M_48": 8,
                    "M_77": 12,
                    "M_169": 22,
                }.items()
                if module_class == class_id
            ],
        }
        for class_id in class_ids
    }
    write_supermodule_results(
        adata,
        supermodule_flux=super_flux,
        supermodule_balance=super_balance,
        flux_sum=node_sum,
        flux_mean=node_mean,
        class_ids=class_ids,
        class_names=class_names,
        module_names=module_names,
        compound_names=compound_names,
        class_results=class_results,
        total_requested=True,
        total_completed=True,
    )
    adata.uns["scFEAann"]["flux"] = {
        "module_names": module_names,
        "loss_history": {
            "total": np.array([5.0, 4.0]),
        },
    }
    adata.uns["scFEAann"]["balance"] = {
        "compound_names": compound_names,
    }
    return adata
