import numpy as np
import pandas as pd

from scFEAann.results import (
    get_scfeaann_supermodule_balance,
    get_scfeaann_supermodule_flux,
    write_supermodule_results,
)


def make_test_adata():
    import anndata as ad

    adata = ad.AnnData(
        np.ones((5, 2)),
        obs=pd.DataFrame(index=[f"c{index}" for index in range(5)]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.uns["scFEAann"] = {}
    return adata


def test_supermodule_results_use_fixed_class_slots() -> None:
    adata = make_test_adata()
    class_ids = list(range(1, 23))
    supermodule_flux = np.arange(15, dtype=np.float32).reshape(5, 3)
    supermodule_balance = np.arange(5 * 22 * 2, dtype=np.float32).reshape(
        5, 22 * 2
    )
    flux_sum = np.arange(5 * 22, dtype=np.float32).reshape(5, 22)
    flux_mean = flux_sum / 2

    write_supermodule_results(
        adata,
        supermodule_flux=supermodule_flux,
        supermodule_balance=supermodule_balance,
        flux_sum=flux_sum,
        flux_mean=flux_mean,
        class_ids=class_ids,
        class_names=[f"Class {index}" for index in class_ids],
        module_names=["M_1", "M_48", "M_77"],
        compound_names=["A", "B"],
        class_results={
            "1": {"status": "success", "module_names": ["M_1"]},
            "8": {"status": "success", "module_names": ["M_48"]},
        },
        total_requested=True,
        total_completed=True,
    )

    assert adata.obsm["scFEA_supermodule_flux"].shape == (5, 3)
    assert adata.obsm["scFEA_supermodule_balance"].shape == (5, 44)
    assert adata.obsm["scFEA_supermodule_flux_sum"].shape == (5, 22)
    assert adata.obsm["scFEA_supermodule_flux_mean"].shape == (5, 22)
    metadata = adata.uns["scFEAann"]["supermodules"]
    assert metadata["balance_axis_order"] == [
        "supermodule_class",
        "compound",
    ]
    assert metadata["total_model"]["completed"] is True


def test_supermodule_retrieval_restores_class_dataframes() -> None:
    adata = make_test_adata()
    class_ids = list(range(1, 23))
    flux = np.arange(15, dtype=np.float32).reshape(5, 3)
    balance = np.arange(5 * 22 * 2, dtype=np.float32).reshape(5, 44)
    write_supermodule_results(
        adata,
        supermodule_flux=flux,
        supermodule_balance=balance,
        flux_sum=np.zeros((5, 22), dtype=np.float32),
        flux_mean=np.zeros((5, 22), dtype=np.float32),
        class_ids=class_ids,
        class_names=[f"Class {index}" for index in class_ids],
        module_names=["M_1", "M_48", "M_77"],
        compound_names=["A", "B"],
        class_results={
            "1": {"status": "success", "module_names": ["M_1"]},
            "8": {"status": "success", "module_names": ["M_48"]},
        },
        total_requested=True,
        total_completed=True,
    )

    class_flux = get_scfeaann_supermodule_flux(adata, class_id=8)
    class_balance = get_scfeaann_supermodule_balance(adata, class_id=8)

    assert class_flux.shape == (5, 1)
    assert class_flux.columns.tolist() == ["M_48"]
    assert class_balance.shape == (5, 2)
    assert class_balance.columns.tolist() == ["A", "B"]
    np.testing.assert_array_equal(
        class_balance.to_numpy(),
        balance.reshape(5, 22, 2)[:, 7, :],
    )
