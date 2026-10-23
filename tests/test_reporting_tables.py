import numpy as np
import pandas as pd
import pytest

from scFEAann.reporting.tables import (
    extract_report_results,
    rank_feature_effects,
    summarize_features,
)


def test_summarize_features_reports_group_distribution() -> None:
    values = np.array([[1.0], [3.0], [10.0], [14.0]])
    groups = pd.Series(["A", "A", "B", "B"])

    summary = summarize_features(values, groups, ["feature"])

    assert summary["group"].tolist() == ["A", "B"]
    assert summary["n"].tolist() == [2, 2]
    assert summary["mean"].tolist() == [2.0, 12.0]
    assert summary["sem"].tolist() == [1.0, 2.0]
    assert summary["median"].tolist() == [2.0, 12.0]


def test_rank_feature_effects_orders_by_between_group_signal() -> None:
    values = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    groups = pd.Series(["A", "A", "B", "B"])

    ranked = rank_feature_effects(values, groups, ["signal", "flat"])

    assert ranked.iloc[0]["feature"] == "signal"
    assert ranked.iloc[0]["effect_ratio"] > ranked.iloc[1]["effect_ratio"]


def test_extract_report_results_builds_nested_group_tables(reporting_adata) -> None:
    report = extract_report_results(
        reporting_adata,
        dataset_key="dataset",
        cluster_key="cluster",
        top_n=2,
    )

    assert report.group_counts["n_cells"].sum() == reporting_adata.n_obs
    assert set(report.metadata.columns) >= {
        "cell_id",
        "dataset",
        "cluster",
        "dataset_cluster",
    }
    assert report.module_effects.iloc[0]["feature"] == "M_1"
    assert report.compound_effects.iloc[0]["feature"] == "C1"
    assert report.top_module_names == ["M_1", "M_2"]
    assert set(report.dataset_consistency["cluster"]) == {"A", "B"}
    assert set(report.dataset_consistency["n_cells_a"]) == {3}
    assert report.dataset_consistency["pearson_r"].min() > 0.5


def test_extract_report_results_rejects_missing_result() -> None:
    import anndata as ad

    adata = ad.AnnData(
        np.ones((2, 2)),
        obs=pd.DataFrame(
            {"dataset": ["D1", "D2"], "cluster": ["A", "A"]},
            index=["c1", "c2"],
        ),
    )
    with pytest.raises(ValueError, match="scFEA_flux"):
        extract_report_results(adata)
