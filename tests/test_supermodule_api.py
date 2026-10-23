import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scFEAann import run_scfeaann_supermodules
from scFEAann.modules import get_module_annotation


def full_model_adata(n_cells: int = 2) -> AnnData:
    module_genes = pd.read_csv("data/module_gene_m168.csv", index_col=0)
    genes = sorted(
        {
            str(gene)
            for gene in module_genes.stack().dropna().tolist()
        }
    )
    rng = np.random.default_rng(11)
    values = rng.normal(4.0, 1.0, size=(n_cells, len(genes))).astype(np.float32)
    return AnnData(
        values,
        obs=pd.DataFrame(
            {
                "dataset": ["dataset_A", "dataset_B"][:n_cells],
                "cluster": ["cell_type_A"] * n_cells,
            },
            index=[f"cell_{index}" for index in range(n_cells)],
        ),
        var=pd.DataFrame(index=genes),
    )


def test_selected_supermodules_write_fixed_ann_data_layout() -> None:
    adata = full_model_adata()
    annotation = get_module_annotation()

    returned = run_scfeaann_supermodules(
        adata,
        classes=[1, 8],
        include_total=True,
        epochs=1,
        device="cpu",
        verbose=False,
    )

    assert returned is adata
    assert adata.obsm["scFEA_flux"].shape == (2, 168)
    assert adata.obsm["scFEA_supermodule_flux"].shape == (2, 168)
    assert adata.obsm["scFEA_supermodule_balance"].shape == (2, 22 * 70)
    assert adata.obsm["scFEA_supermodule_flux_sum"].shape == (2, 22)
    class1_modules = annotation.index[
        annotation["supermodule_id"] == 1
    ].tolist()
    class8_modules = annotation.index[
        annotation["supermodule_id"] == 8
    ].tolist()
    assert np.isfinite(
        adata.obsm["scFEA_supermodule_flux"][:, : len(class1_modules)]
    ).all()
    class8_index = adata.uns["scFEAann"]["supermodules"]["module_names"].index(
        class8_modules[0]
    )
    assert np.isfinite(
        adata.obsm["scFEA_supermodule_flux"][:, class8_index]
    ).all()
    assert np.isnan(
        adata.obsm["scFEA_supermodule_flux_sum"][:, 1]
    ).all()
    metadata = adata.uns["scFEAann"]["supermodules"]
    assert metadata["class_results"]["1"]["status"] == "success"
    assert metadata["class_results"]["8"]["status"] == "success"
    assert metadata["class_results"]["2"]["status"] == "not_requested"
    assert metadata["total_model"]["completed"] is True


def test_all_22_classes_fill_every_supermodule_slot() -> None:
    adata = full_model_adata(n_cells=2)

    run_scfeaann_supermodules(
        adata,
        classes=None,
        include_total=False,
        epochs=1,
        device="cpu",
        verbose=False,
    )

    metadata = adata.uns["scFEAann"]["supermodules"]
    assert all(
        result["status"] == "success"
        for result in metadata["class_results"].values()
    )
    assert np.isfinite(adata.obsm["scFEA_supermodule_flux"]).all()
    assert np.isfinite(adata.obsm["scFEA_supermodule_balance"]).all()
    assert np.isfinite(adata.obsm["scFEA_supermodule_flux_sum"]).all()
    assert metadata["total_model"]["requested"] is False
    assert metadata["class_results"]["3"]["module_loss_disabled"] is True
    assert metadata["class_results"]["1"]["module_loss_disabled"] is False


def test_copy_true_leaves_original_adata_unchanged() -> None:
    adata = full_model_adata()

    returned = run_scfeaann_supermodules(
        adata,
        classes=[1],
        include_total=False,
        epochs=1,
        device="cpu",
        verbose=False,
        copy=True,
    )

    assert returned is not adata
    assert "scFEA_supermodule_flux" not in adata.obsm
    assert "scFEA_supermodule_flux" in returned.obsm


def test_unknown_supermodule_selector_raises() -> None:
    with pytest.raises(ValueError, match="unknown supermodule"):
        run_scfeaann_supermodules(
            full_model_adata(),
            classes=[999],
            include_total=False,
            epochs=1,
            verbose=False,
        )
