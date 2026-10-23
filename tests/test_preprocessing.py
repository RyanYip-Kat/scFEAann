import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scFEAann.preprocessing import prepare_scfea_inputs


def csv_to_adata(path: str) -> ad.AnnData:
    expression = pd.read_csv(path, index_col=0)
    return ad.AnnData(
        expression.T.astype(float),
        obs=pd.DataFrame(index=expression.columns.astype(str)),
        var=pd.DataFrame(index=expression.index.astype(str)),
    )


def test_prepared_inputs_match_legacy_preprocessing() -> None:
    expression = pd.read_csv("input/Melissa_full.csv", index_col=0)
    adata = csv_to_adata("input/Melissa_full.csv")

    prepared = prepare_scfea_inputs(adata)

    modules = pd.read_csv("data/module_gene_m168.csv", index_col=0)
    module_genes = set(modules.stack().dropna().astype(str))
    expected_genes = sorted(set(expression.index.astype(str)) & module_genes)
    assert prepared.gene_names == expected_genes
    assert prepared.cell_names == adata.obs_names.tolist()
    assert prepared.X.shape == (adata.n_obs, modules.shape[0] * len(expected_genes))

    legacy = expression.T * 1.0
    if legacy.max().max() > 50:
        legacy = (legacy + 1).apply(np.log2)
    expected_gene_scale = legacy.sum(axis=1)
    expected_gene_scale = expected_gene_scale / expected_gene_scale.mean()
    np.testing.assert_allclose(
        prepared.gene_scale.numpy(),
        expected_gene_scale.to_numpy(dtype=np.float32),
        rtol=0,
        atol=0,
    )

    first_module_genes = set(modules.iloc[0].dropna().astype(str))
    expected_first_block = legacy[expected_genes].copy()
    expected_first_block.loc[
        :, [gene for gene in expected_genes if gene not in first_module_genes]
    ] = 0
    np.testing.assert_allclose(
        prepared.X[:, : len(expected_genes)].numpy(),
        expected_first_block.to_numpy(dtype=np.float32),
        rtol=0,
        atol=0,
    )

    expected_module_scale = np.zeros((adata.n_obs, modules.shape[0]))
    module_lengths = modules.notna().sum(axis=1).to_numpy()
    for module_index in range(modules.shape[0]):
        genes = set(modules.iloc[module_index].dropna().astype(str))
        selected = [
            gene for gene in expected_genes if gene in genes
        ]
        if selected:
            expected_module_scale[:, module_index] = (
                legacy[selected].sum(axis=1) / module_lengths[module_index]
            )
    np.testing.assert_allclose(
        prepared.module_scale.numpy(),
        expected_module_scale.astype(np.float32),
        rtol=0,
        atol=0,
    )


def test_rejects_no_overlap_genes() -> None:
    adata = ad.AnnData(
        np.ones((2, 2)),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["not_a_module_gene_a", "not_a_module_gene_b"]),
    )

    with pytest.raises(ValueError, match="overlap"):
        prepare_scfea_inputs(adata)


def test_rejects_non_finite_expression() -> None:
    adata = csv_to_adata("input/Melissa_full.csv")
    adata.X = adata.X.copy()
    adata.X[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        prepare_scfea_inputs(adata)


def test_sparse_and_dense_inputs_prepare_identically() -> None:
    dense = csv_to_adata("input/Melissa_full.csv")
    sparse_adata = ad.AnnData(
        sparse.csr_matrix(np.asarray(dense.X)),
        obs=pd.DataFrame(index=dense.obs_names),
        var=pd.DataFrame(index=dense.var_names),
    )

    dense_prepared = prepare_scfea_inputs(dense)
    sparse_prepared = prepare_scfea_inputs(sparse_adata)

    assert sparse_prepared.gene_names == dense_prepared.gene_names
    np.testing.assert_array_equal(
        sparse_prepared.X.numpy(), dense_prepared.X.numpy()
    )
    np.testing.assert_array_equal(
        sparse_prepared.gene_scale.numpy(),
        dense_prepared.gene_scale.numpy(),
    )
    np.testing.assert_array_equal(
        sparse_prepared.module_scale.numpy(),
        dense_prepared.module_scale.numpy(),
    )
