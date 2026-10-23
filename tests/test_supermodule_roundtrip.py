import numpy as np
from anndata import read_h5ad

from scFEAann import run_scfeaann_supermodules
from scFEAann.results import (
    get_scfeaann_supermodule_balance,
    get_scfeaann_supermodule_flux,
)
from tests.test_supermodule_api import full_model_adata


def test_supermodule_results_survive_h5ad_roundtrip(tmp_path) -> None:
    adata = full_model_adata(n_cells=2)
    run_scfeaann_supermodules(
        adata,
        classes=[1, 3],
        include_total=True,
        epochs=1,
        device="cpu",
        verbose=False,
    )
    path = tmp_path / "supermodules.h5ad"
    adata.write_h5ad(path)

    loaded = read_h5ad(path)

    assert loaded.obsm["scFEA_supermodule_flux"].shape == (2, 168)
    assert loaded.obsm["scFEA_supermodule_balance"].shape == (2, 1540)
    assert get_scfeaann_supermodule_flux(loaded, class_id=3).shape[1] == 1
    assert get_scfeaann_supermodule_balance(loaded, class_id=3).shape == (
        2,
        70,
    )
    class3 = loaded.uns["scFEAann"]["supermodules"]["class_results"]["3"]
    assert class3["status"] == "success"
    assert class3["module_loss_disabled"] is True
    assert np.isfinite(loaded.obsm["scFEA_supermodule_flux"]).sum() > 0
