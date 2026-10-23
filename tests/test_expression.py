import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scFEAann.expression import resolve_expression


def make_adata() -> ad.AnnData:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    adata.layers["counts"] = matrix * 10
    adata.raw = adata
    return adata


def test_defaults_to_x() -> None:
    adata = make_adata()

    matrix, genes, source = resolve_expression(adata)

    assert np.array_equal(np.asarray(matrix), np.asarray(adata.X))
    assert genes.tolist() == ["g1", "g2"]
    assert source == "X"


def test_selects_layer() -> None:
    adata = make_adata()

    matrix, genes, source = resolve_expression(adata, layer="counts")

    assert np.array_equal(np.asarray(matrix), np.asarray(adata.layers["counts"]))
    assert genes.tolist() == ["g1", "g2"]
    assert source == "layer:counts"


def test_selects_raw() -> None:
    adata = make_adata()

    matrix, genes, source = resolve_expression(adata, use_raw=True)

    assert np.array_equal(np.asarray(matrix), np.asarray(adata.raw.X))
    assert genes.tolist() == ["g1", "g2"]
    assert source == "raw"


def test_rejects_layer_and_raw_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_expression(make_adata(), layer="counts", use_raw=True)


def test_rejects_missing_layer() -> None:
    with pytest.raises(ValueError, match="layer 'missing'"):
        resolve_expression(make_adata(), layer="missing")


def test_rejects_raw_when_unavailable() -> None:
    adata = make_adata()
    adata.raw = None

    with pytest.raises(ValueError, match="raw"):
        resolve_expression(adata, use_raw=True)


def test_rejects_duplicate_gene_names() -> None:
    adata = make_adata()
    adata.var_names = ["g1", "g1"]

    with pytest.raises(ValueError, match="var_names"):
        resolve_expression(adata)
