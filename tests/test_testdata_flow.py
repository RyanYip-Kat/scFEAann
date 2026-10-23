from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad

from examples.scfeaann_testdata_flow import (
    add_nested_group_column,
    run_testdata_flow,
    select_observation_positions,
)


TESTDATA = Path("testdata/testdata.h5ad")


def test_nested_group_column_combines_dataset_and_cluster() -> None:
    obs = pd.DataFrame(
        {
            "dataset": ["D1", "D1", "D2"],
            "cluster": ["a", "b", "a"],
        },
        index=["c1", "c2", "c3"],
    )

    groups = add_nested_group_column(obs)

    assert groups.tolist() == ["D1 | a", "D1 | b", "D2 | a"]
    assert "dataset_cluster" not in obs.columns


def test_sampling_covers_every_dataset_cluster_combination() -> None:
    obs = pd.DataFrame(
        {
            "dataset": ["D1"] * 6 + ["D2"] * 6,
            "cluster": ["a", "a", "b", "b", "c", "c"] * 2,
        },
        index=[f"c{index}" for index in range(12)],
    )

    first = select_observation_positions(
        obs, max_cells=6, random_state=7
    )
    second = select_observation_positions(
        obs, max_cells=6, random_state=7
    )

    np.testing.assert_array_equal(first, second)
    selected = obs.iloc[first]
    assert selected.groupby(["dataset", "cluster"]).ngroups == 6
    assert len(first) == 6


def test_dataset_filter_rejects_unknown_dataset() -> None:
    obs = pd.DataFrame(
        {"dataset": ["D1"], "cluster": ["a"]}, index=["c1"]
    )

    with pytest.raises(ValueError, match="dataset"):
        select_observation_positions(obs, dataset="D3")


@pytest.mark.skipif(not TESTDATA.exists(), reason="testdata H5AD is unavailable")
def test_real_testdata_flow_with_gradient(tmp_path) -> None:
    output = tmp_path / "testdata_scfeaann.h5ad"

    result = run_testdata_flow(
        TESTDATA,
        output,
        dataset="D2",
        max_cells=6,
        epochs=1,
        gradient=True,
        device="cpu",
        verbose=False,
    )

    assert result.n_obs == 6
    assert set(result.obs["dataset"]) == {"D2"}
    assert result.obs["dataset_cluster"].nunique() == 3
    assert result.obsm["scFEA_flux"].shape == (6, 168)
    assert result.obsm["scFEA_balance"].shape == (6, 70)
    assert result.uns["scFEAann"]["group_key"] == "dataset_cluster"
    assert result.obsm["scFEA_grad"].ndim == 2
    assert output.exists()

    loaded = read_h5ad(output)
    assert loaded.obsm["scFEA_flux"].shape == (6, 168)
    assert loaded.obs["cluster"].nunique() == 3
