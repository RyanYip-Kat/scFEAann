import pandas as pd

from scFEAann.modules import get_module_annotation


def test_human_module_annotation_contains_all_168_modules() -> None:
    annotation = get_module_annotation()

    assert annotation.shape[0] == 168
    assert annotation.index.is_unique
    assert annotation.loc["M_1", "module_number"] == 1
    assert annotation.loc["M_171", "module_number"] == 171
    assert {"M_82", "M_86", "M_104"}.isdisjoint(annotation.index)


def test_named_modules_have_reaction_and_compound_mapping() -> None:
    annotation = get_module_annotation()

    assert annotation.loc["M_1", "reaction"] == "Glucose -> G6P"
    assert annotation.loc["M_1", "input_kegg_ids"] == "C00267"
    assert annotation.loc["M_1", "output_kegg_ids"] == "C00668"
    assert annotation.loc["M_48", "reaction"] == "Glutamate -> Glutamine"
    assert annotation.loc["M_77", "reaction"] == "Oxaloacetate_in -> Oxaloacetate"
    assert annotation.loc["M_77", "external_input"] is True or (
        annotation.loc["M_77", "external_input"] == True  # noqa: E712
    )
    assert annotation.loc["M_77", "modeled_in_balance"] is True or (
        annotation.loc["M_77", "modeled_in_balance"] == True  # noqa: E712
    )


def test_all_modules_match_information_and_stoichiometry_signs() -> None:
    annotation = get_module_annotation()

    assert annotation["stoichiometry_match"].all()
    assert annotation["n_genes"].gt(0).all()
    assert annotation["module_genes"].str.contains(";").any()


def test_module_annotation_is_exported_in_report_workbook(
    tmp_path, reporting_adata
) -> None:
    from scFEAann.reporting.report import export_scfeaann_report

    export_scfeaann_report(
        reporting_adata,
        tmp_path,
        embedding_method="pca",
        dpi=150,
    )

    sheets = pd.ExcelFile(tmp_path / "report_tables.xlsx").sheet_names
    assert "module_annotation" in sheets
    assert (tmp_path / "module_annotation.csv").exists()
