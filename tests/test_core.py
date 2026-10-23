import numpy as np
import pytest
import torch

from scFEAann.core.dataset import MyDataset
from scFEAann.core.flux import FLUX
from scFEAann.core.loss import scfea_loss
from scFEAann.core.training import fit_scfea, predict_scfea


def test_dataset_returns_three_aligned_tensors() -> None:
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    scale = torch.arange(4, dtype=torch.float32)
    module_scale = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    x, gene_scale, module_value = MyDataset(data, scale, module_scale)[2]

    assert torch.equal(x, data[2])
    assert gene_scale.item() == 2
    assert torch.equal(module_value, module_scale[2])


def test_flux_shapes_and_balance_formula() -> None:
    stoichiometry = torch.tensor([[1.0, -1.0]])
    network = FLUX(None, n_modules=2, f_in=3, f_out=1)
    expression = torch.ones(4, 6)

    flux, compound_balance = network(
        expression, 2, 3, 1, stoichiometry
    )

    assert flux.shape == (4, 2)
    assert compound_balance.shape == (4, 1)
    assert torch.equal(
        compound_balance[:, 0], flux[:, 0] - flux[:, 1]
    )


def test_loss_has_five_components_and_nonnegative_penalty() -> None:
    flux = torch.tensor(
        [[1.0, 2.0], [-3.0, 4.0]], requires_grad=True
    )
    balance = torch.zeros(2, 1)

    loss, balance_loss, negative_loss, cell_loss, module_loss = scfea_loss(
        flux,
        balance,
        gene_scale=torch.tensor([3.0, 1.0]),
        module_scale=torch.tensor([[1.0, 2.0], [2.0, 1.0]]),
    )

    assert loss.item() >= 0
    assert balance_loss.item() == 0
    assert negative_loss.item() == pytest.approx(1.2)
    assert cell_loss.item() == pytest.approx(0.0)
    assert module_loss.item() >= -1e-6


def test_loss_can_skip_undefined_module_correlation() -> None:
    flux = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    balance = torch.zeros(2, 1)
    constant_module_scale = torch.ones(2, 2)

    unsafe = scfea_loss(
        flux,
        balance,
        gene_scale=torch.tensor([3.0, 3.0]),
        module_scale=constant_module_scale,
    )
    safe = scfea_loss(
        flux,
        balance,
        gene_scale=torch.tensor([3.0, 3.0]),
        module_scale=constant_module_scale,
        safe_module_correlation=True,
    )

    assert not torch.isfinite(unsafe[0])
    assert torch.isfinite(safe[0])
    assert safe[4].item() == pytest.approx(0.0)


def test_fit_and_predict_return_legacy_shapes() -> None:
    cells, modules, genes, compounds = 3, 2, 4, 1
    expression = torch.arange(
        cells * modules * genes, dtype=torch.float32
    ).reshape(cells, modules * genes)
    gene_scale = torch.ones(cells)
    module_scale = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [1.0, 3.0]]
    )
    stoichiometry = torch.tensor([[1.0, -1.0]])

    fitted = fit_scfea(
        expression,
        gene_scale,
        module_scale,
        stoichiometry,
        n_modules=modules,
        n_genes=genes,
        n_comps=compounds,
        epochs=1,
        device="cpu",
        verbose=False,
    )
    flux, balance = predict_scfea(
        fitted.net,
        expression,
        stoichiometry,
        n_modules=modules,
        n_genes=genes,
        n_comps=compounds,
        device="cpu",
    )

    assert fitted.total_losses.shape == (1,)
    assert flux.shape == (cells, modules)
    assert balance.shape == (cells, compounds)
    assert np.isfinite(flux).all()
    assert np.isfinite(balance).all()
