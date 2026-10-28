import numpy as np
import pytest
import torch

from scFEAann.core.flux import FLUX
from scFEAann.core.training import predict_scfea


def test_chunked_prediction_matches_batch_one() -> None:
    torch.manual_seed(16)
    net = FLUX(None, n_modules=3, f_in=4, f_out=1)
    expression = torch.arange(120, dtype=torch.float32).reshape(10, 12)
    stoichiometry = torch.tensor(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]
    )

    flux_one, balance_one = predict_scfea(
        net,
        expression,
        stoichiometry,
        n_modules=3,
        n_genes=4,
        n_comps=2,
        device="cpu",
        prediction_batch_size=1,
    )
    flux_chunked, balance_chunked = predict_scfea(
        net,
        expression,
        stoichiometry,
        n_modules=3,
        n_genes=4,
        n_comps=2,
        device="cpu",
        prediction_batch_size=4,
    )

    np.testing.assert_allclose(
        flux_chunked, flux_one, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        balance_chunked, balance_one, rtol=1e-5, atol=1e-6
    )


def test_prediction_batch_size_must_be_positive() -> None:
    net = FLUX(None, n_modules=1, f_in=2, f_out=1)
    expression = torch.ones(1, 2)
    stoichiometry = torch.ones(1, 1)

    with pytest.raises(ValueError, match="prediction_batch_size"):
        predict_scfea(
            net,
            expression,
            stoichiometry,
            n_modules=1,
            n_genes=2,
            n_comps=1,
            device="cpu",
            prediction_batch_size=0,
        )
