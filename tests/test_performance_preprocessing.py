import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scFEAann.preprocessing import prepare_scfea_inputs
from scFEAann.supermodules import (
    materialize_module_blocks,
    subset_prepared_inputs,
)


def csv_to_adata(path: str) -> ad.AnnData:
    expression = pd.read_csv(path, index_col=0)
    return ad.AnnData(
        expression.T.astype(float),
        obs=pd.DataFrame(index=expression.columns.astype(str)),
        var=pd.DataFrame(index=expression.index.astype(str)),
    )


def test_deferred_module_blocks_keep_x_unbuilt() -> None:
    adata = csv_to_adata("input/Melissa_full.csv")

    prepared = prepare_scfea_inputs(adata, build_module_blocks=False)

    assert prepared.X is None
    assert prepared.gene_expression.shape[0] == adata.n_obs
    assert prepared.module_masks.shape[0] == 168

    class1 = materialize_module_blocks(prepared, prepared.module_names[:3])
    assert class1.X is not None
    assert class1.X.shape == (
        adata.n_obs,
        3 * len(prepared.gene_names),
    )
    assert class1.gene_expression is prepared.gene_expression


def test_deferred_class_blocks_match_full_tensor_subset() -> None:
    adata = csv_to_adata("input/Melissa_full.csv")
    full = prepare_scfea_inputs(adata)
    deferred = prepare_scfea_inputs(adata, build_module_blocks=False)
    selected = [full.module_names[0], full.module_names[17], full.module_names[-1]]

    expected = subset_prepared_inputs(full, selected)
    observed = materialize_module_blocks(deferred, selected)

    np.testing.assert_array_equal(observed.X.numpy(), expected.X.numpy())
    np.testing.assert_array_equal(
        observed.module_masks.numpy(),
        expected.module_masks.numpy(),
    )
    np.testing.assert_array_equal(
        observed.module_scale.numpy(),
        expected.module_scale.numpy(),
    )
    np.testing.assert_array_equal(
        observed.stoichiometry.numpy(),
        expected.stoichiometry.numpy(),
    )
    assert observed.module_names == expected.module_names


def test_sparse_fast_path_matches_dense_for_preprocessing() -> None:
    dense = csv_to_adata("input/Melissa_full.csv")
    sparse_adata = ad.AnnData(
        sparse.csr_matrix(np.asarray(dense.X)),
        obs=pd.DataFrame(index=dense.obs_names),
        var=pd.DataFrame(index=dense.var_names),
    )

    dense_prepared = prepare_scfea_inputs(dense)
    sparse_prepared = prepare_scfea_inputs(sparse_adata)

    np.testing.assert_array_equal(
        dense_prepared.X.numpy(), sparse_prepared.X.numpy()
    )
    np.testing.assert_allclose(
        dense_prepared.gene_scale.numpy(),
        sparse_prepared.gene_scale.numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        dense_prepared.module_scale.numpy(),
        sparse_prepared.module_scale.numpy(),
        rtol=0,
        atol=0,
    )
