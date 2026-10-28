import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scFEAann import run_scfeaann


def tiny_adata(group_key: str = "cell_type") -> ad.AnnData:
    genes = ["ADPGK", "G6PC", "ALDOA"]
    return ad.AnnData(
        np.arange(15, dtype=np.float32).reshape(5, 3),
        obs=pd.DataFrame(
            {group_key: ["a", "a", "b", "b", "b"]},
            index=[f"c{index}" for index in range(5)],
        ),
        var=pd.DataFrame(index=genes),
    )


def test_flux_api_writes_results_and_preserves_group() -> None:
    adata = tiny_adata()

    returned = run_scfeaann(
        adata, group_key="cell_type", epochs=1, verbose=False
    )

    assert returned is adata
    assert adata.obsm["scFEA_flux"].shape == (5, 168)
    assert adata.obsm["scFEA_balance"].shape == (5, 70)
    assert adata.obs["cell_type"].tolist() == ["a", "a", "b", "b", "b"]
    metadata = adata.uns["scFEAann"]
    assert metadata["group_key"] == "cell_type"
    assert metadata["flux"]["obsm_key"] == "scFEA_flux"
    assert metadata["balance"]["obsm_key"] == "scFEA_balance"
    assert len(metadata["flux"]["module_names"]) == 168
    assert len(metadata["balance"]["compound_names"]) == 70
    assert metadata["expression_source"] == "X"
    assert metadata["flux"]["loss_history"]["total"].shape == (1,)


def test_copy_true_does_not_modify_input() -> None:
    adata = tiny_adata()

    returned = run_scfeaann(
        adata, epochs=1, copy=True, verbose=False
    )

    assert returned is not adata
    assert "scFEA_flux" not in adata.obsm
    assert "scFEAann" not in adata.uns
    assert "scFEA_flux" in returned.obsm
    assert np.array_equal(
        adata.obs_names.to_numpy(), returned.obs_names.to_numpy()
    )


def test_missing_group_column_raises_before_training() -> None:
    with pytest.raises(ValueError, match="group_key"):
        run_scfeaann(
            tiny_adata(), group_key="missing", epochs=1, verbose=False
        )


def test_layer_source_is_recorded() -> None:
    adata = tiny_adata()
    adata.layers["counts"] = np.asarray(adata.X) * 2

    run_scfeaann(
        adata, layer="counts", epochs=1, verbose=False
    )

    assert adata.uns["scFEAann"]["expression_source"] == "layer:counts"


def test_invalid_prediction_batch_size_fails_before_preprocessing(
    monkeypatch,
) -> None:
    import scFEAann.api as api

    def unexpected_preprocessing(*args, **kwargs):
        raise AssertionError("preprocessing should not start")

    monkeypatch.setattr(api, "prepare_scfea_inputs", unexpected_preprocessing)

    with pytest.raises(ValueError, match="prediction_batch_size"):
        run_scfeaann(
            tiny_adata(),
            epochs=1,
            verbose=False,
            prediction_batch_size=0,
        )
