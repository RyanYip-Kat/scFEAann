import anndata as ad
import numpy as np
import pandas as pd
from anndata import read_h5ad

from scFEAann import (
    calculate_scfeaann_gradient,
    get_scfeaann_gradient,
    run_scfeaann,
)


def tiny_adata() -> ad.AnnData:
    return ad.AnnData(
        (np.arange(15, dtype=np.float32) + 1).reshape(5, 3),
        obs=pd.DataFrame(
            {"cell_type": ["a", "a", "b", "b", "b"]},
            index=[f"c{index}" for index in range(5)],
        ),
        var=pd.DataFrame(index=["ADPGK", "G6PC", "ALDOA"]),
    )


def test_flux_and_gradient_survive_h5ad_roundtrip(tmp_path) -> None:
    adata = tiny_adata()
    run_scfeaann(
        adata,
        group_key="cell_type",
        epochs=1,
        verbose=False,
    )
    calculate_scfeaann_gradient(
        adata,
        group_key="cell_type",
        epochs=1,
        verbose=False,
    )
    path = tmp_path / "result.h5ad"
    adata.write_h5ad(path)

    loaded = read_h5ad(path)

    np.testing.assert_array_equal(
        loaded.obsm["scFEA_flux"], adata.obsm["scFEA_flux"]
    )
    np.testing.assert_array_equal(
        loaded.obsm["scFEA_grad"], adata.obsm["scFEA_grad"]
    )
    assert list(loaded.uns["scFEAann"]["flux"]["module_names"]) == (
        list(adata.uns["scFEAann"]["flux"]["module_names"])
    )
    assert loaded.uns["scFEAann"]["gradient"]["flatten_order"] == "C"
    restored = get_scfeaann_gradient(loaded, cell="c1")
    assert restored.shape == (len(loaded.uns["scFEAann"]["gradient"]["gene_names"]), 168)
