import numpy as np

from scFEAann.reporting.supermodules import extract_supermodule_report


def test_supermodule_report_builds_group_and_effect_tables(
    supermodule_adata,
) -> None:
    report = extract_supermodule_report(supermodule_adata)

    assert report.node_mean.shape == (12, 22)
    assert report.node_sum.shape == (12, 22)
    assert set(report.group_summary["metric"]) == {"mean", "sum"}
    assert report.group_summary["n"].gt(0).all()
    assert report.effects["effect_ratio"].between(0, 1).all()
    assert len(report.group_summary_by_class) == 22 * 4 * 2


def test_supermodule_agreement_uses_global_module_aggregation(
    supermodule_adata,
) -> None:
    report = extract_supermodule_report(supermodule_adata)

    class1 = report.agreement[
        report.agreement["supermodule_id"] == 1
    ].iloc[0]
    assert class1["n_modules"] == 1
    assert np.isfinite(class1["pearson_r"])
    assert np.isfinite(class1["rmse"])
    assert class1["pearson_r"] <= 1.0


def test_supermodule_correlation_is_square_and_has_unit_diagonal(
    supermodule_adata,
) -> None:
    report = extract_supermodule_report(supermodule_adata)

    assert report.correlation.shape == (22, 22)
    assert np.isfinite(report.correlation.to_numpy()).all()
    np.testing.assert_allclose(np.diag(report.correlation), 1.0)


def test_supermodule_source_data_contains_main_and_supp_panels(
    supermodule_adata,
) -> None:
    report = extract_supermodule_report(supermodule_adata)

    assert {
        "Fig4a_group_heatmap",
        "Fig4b_top_class_distributions",
        "Fig4c_class_correlation",
        "Fig4d_model_agreement",
        "SuppFig2a_class_embedding",
        "SuppFig2c_global_class_scatter",
        "SuppFig2d_effect_ranking",
    } <= set(report.source_data)
