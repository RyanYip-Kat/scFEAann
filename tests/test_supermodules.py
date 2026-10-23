import torch

from scFEAann.preprocessing import PreparedInputs
from scFEAann.supermodules import (
    class_module_names,
    load_supermodules,
    resolve_supermodule_classes,
    subset_prepared_inputs,
)


def test_supermodule_table_contains_22_classes() -> None:
    table = load_supermodules()

    assert table.shape[0] == 22
    assert table.index.tolist() == list(range(1, 23))
    assert table.loc[8, "Supermodule class"] == "Glutamate metabolism"


def test_supermodule_selectors_accept_ids_and_names() -> None:
    assert resolve_supermodule_classes(None) == list(range(1, 23))
    assert resolve_supermodule_classes([8, "Glycolysis+TCA cycle"]) == [1, 8]


def test_supermodule_8_contains_expected_modules() -> None:
    assert class_module_names(8) == [
        "M_48",
        "M_49",
        "M_50",
        "M_51",
        "M_52",
    ]


def test_subset_prepared_inputs_selects_consistent_module_blocks() -> None:
    cells, modules, genes, compounds = 3, 3, 2, 2
    prepared = PreparedInputs(
        X=torch.arange(cells * modules * genes).reshape(
            cells, modules * genes
        ),
        gene_expression=torch.ones(cells, genes),
        module_masks=torch.tensor([[True, False], [False, True], [True, True]]),
        gene_scale=torch.ones(cells),
        module_scale=torch.tensor([[1.0, 2.0, 3.0]] * cells),
        stoichiometry=torch.tensor(
            [[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]
        ),
        gene_names=["G1", "G2"],
        cell_names=["C1", "C2", "C3"],
        module_names=["M_1", "M_48", "M_77"],
        compound_names=["A", "B"],
        expression_source="X",
        model_metadata={},
    )

    subset = subset_prepared_inputs(prepared, ["M_1", "M_77"])

    expected_x = torch.cat(
        [
            prepared.X.reshape(cells, modules, genes)[:, 0, :],
            prepared.X.reshape(cells, modules, genes)[:, 2, :],
        ],
        dim=1,
    )
    assert subset.module_names == ["M_1", "M_77"]
    assert torch.equal(subset.X, expected_x)
    assert torch.equal(
        subset.module_scale, prepared.module_scale[:, [0, 2]]
    )
    assert torch.equal(
        subset.stoichiometry, prepared.stoichiometry[:, [0, 2]]
    )
