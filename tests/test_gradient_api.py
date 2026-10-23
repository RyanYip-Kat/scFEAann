import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scFEAann import calculate_scfeaann_gradient, get_scfeaann_gradient


def gradient_adata() -> ad.AnnData:
    genes = ["ADPGK", "G6PC", "ALDOA"]
    return ad.AnnData(
        (np.arange(15, dtype=np.float32) + 1).reshape(5, 3),
        obs=pd.DataFrame(
            {"cell_type": ["a", "a", "b", "b", "b"]},
            index=[f"c{index}" for index in range(5)],
        ),
        var=pd.DataFrame(index=genes),
    )


def test_gradient_is_stored_flat_and_can_be_restored() -> None:
    adata = gradient_adata()

    returned = calculate_scfeaann_gradient(
        adata, group_key="cell_type", epochs=1, verbose=False
    )

    assert returned is adata
    assert "scFEA_flux" not in adata.obsm
    assert adata.obs["cell_type"].tolist() == ["a", "a", "b", "b", "b"]
    flat = adata.obsm["scFEA_grad"]
    metadata = adata.uns["scFEAann"]["gradient"]
    genes = metadata["gene_names"]
    modules = metadata["module_names"]
    assert flat.shape == (adata.n_obs, len(genes) * len(modules))
    assert metadata["flatten_order"] == "C"
    assert metadata["axis_order"] == ["cell", "gene", "module"]

    one_cell = get_scfeaann_gradient(adata, cell="c0")
    assert one_cell.shape == (len(genes), len(modules))
    assert one_cell.index.tolist() == genes
    assert one_cell.columns.tolist() == modules

    all_cells = get_scfeaann_gradient(adata)
    assert all_cells.shape == (adata.n_obs, len(genes), len(modules))
    np.testing.assert_array_equal(flat, all_cells.reshape(adata.n_obs, -1))


def test_gradient_does_not_clobber_existing_flux_metadata() -> None:
    from scFEAann import run_scfeaann

    adata = gradient_adata()
    run_scfeaann(adata, epochs=1, verbose=False)
    flux = adata.obsm["scFEA_flux"].copy()

    calculate_scfeaann_gradient(adata, epochs=1, verbose=False)

    assert np.array_equal(adata.obsm["scFEA_flux"], flux)
    assert "flux" in adata.uns["scFEAann"]
    assert "gradient" in adata.uns["scFEAann"]
    assert adata.uns["scFEAann"]["flux"]["parameters"]["epochs"] == 1
    assert adata.uns["scFEAann"]["flux"]["expression_source"] == "X"
    assert adata.uns["scFEAann"]["flux"]["model"]["module_gene_file"].endswith(
        "module_gene_m168.csv"
    )


def test_flux_does_not_clobber_existing_gradient_metadata() -> None:
    from scFEAann import run_scfeaann

    adata = gradient_adata()
    calculate_scfeaann_gradient(adata, epochs=1, verbose=False)
    gradient = adata.obsm["scFEA_grad"].copy()

    run_scfeaann(adata, epochs=1, verbose=False)

    np.testing.assert_array_equal(adata.obsm["scFEA_grad"], gradient)
    assert adata.uns["scFEAann"]["gradient"]["parameters"]["epochs"] == 1
    assert adata.uns["scFEAann"]["gradient"]["expression_source"] == "X"


@pytest.mark.parametrize("cell_index", [-1, 5])
def test_gradient_helper_rejects_out_of_range_integer(cell_index) -> None:
    adata = gradient_adata()
    calculate_scfeaann_gradient(adata, epochs=1, verbose=False)

    with pytest.raises(ValueError, match="out of range"):
        get_scfeaann_gradient(adata, cell=cell_index)
