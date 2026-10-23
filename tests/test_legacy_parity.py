import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scFEAann.core.loss import scfea_loss
from scFEAann.core.training import (
    compute_scfea_gradient,
    fit_scfea,
    predict_scfea,
)


def load_legacy_module():
    source_dir = Path("src").resolve()
    sys.path.insert(0, str(source_dir))

    magic = types.ModuleType("magic")
    magic.MAGIC = object
    sys.modules.setdefault("magic", magic)
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)

    spec = importlib.util.spec_from_file_location(
        "legacy_scfea", source_dir / "scFEA.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load legacy scFEA module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_legacy(
    legacy,
    X: torch.Tensor,
    gene_scale: torch.Tensor,
    module_scale: torch.Tensor,
    stoichiometry: torch.Tensor,
    *,
    n_modules: int,
    n_genes: int,
    n_comps: int,
):
    torch.manual_seed(16)
    network = legacy.FLUX(X, n_modules, f_in=n_genes, f_out=1)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.008)
    dataset = legacy.MyDataset(X, gene_scale, module_scale)
    loader = DataLoader(
        dataset, batch_size=X.shape[0], shuffle=False, num_workers=0
    )
    network.train()
    for _ in range(1):
        for batch_x, batch_gene_scale, batch_module_scale in loader:
            flux, balance = network(
                batch_x,
                n_modules,
                n_genes,
                n_comps,
                stoichiometry,
            )
            components = legacy.myLoss(
                flux,
                balance,
                lamb1=1,
                lamb2=1,
                lamb3=1,
                lamb4=1e-2,
                geneScale=batch_gene_scale,
                moduleScale=batch_module_scale,
            )
            optimizer.zero_grad()
            components[0].backward()
            optimizer.step()
    return network, optimizer


def make_core_inputs():
    cells, modules, genes, compounds = 4, 2, 3, 2
    expression = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            [3.0, 2.0, 1.0, 6.0, 4.0, 5.0],
            [4.0, 1.0, 2.0, 5.0, 3.0, 6.0],
        ]
    )
    gene_expression = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 2.0, 1.0],
            [4.0, 1.0, 2.0],
        ]
    )
    gene_scale = gene_expression.sum(dim=1) / gene_expression.sum().item()
    module_scale = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [1.0, 3.0], [3.0, 1.0]]
    )
    module_masks = torch.tensor(
        [[True, True, False], [False, False, True]]
    )
    stoichiometry = torch.tensor(
        [[1.0, -1.0], [0.0, 1.0]]
    )
    return {
        "cells": cells,
        "modules": modules,
        "genes": genes,
        "compounds": compounds,
        "expression": expression,
        "gene_expression": gene_expression,
        "gene_scale": gene_scale,
        "module_scale": module_scale,
        "module_masks": module_masks,
        "stoichiometry": stoichiometry,
    }


def test_flux_and_balance_match_legacy_core() -> None:
    torch.set_num_threads(1)
    legacy = load_legacy_module()
    values = make_core_inputs()
    X = values["expression"]
    legacy_net, _ = train_legacy(
        legacy,
        X,
        values["gene_scale"],
        values["module_scale"],
        values["stoichiometry"],
        n_modules=values["modules"],
        n_genes=values["genes"],
        n_comps=values["compounds"],
    )
    with torch.no_grad():
        legacy_flux, legacy_balance = legacy_net(
            X,
            values["modules"],
            values["genes"],
            values["compounds"],
            values["stoichiometry"],
        )

    fitted = fit_scfea(
        X,
        values["gene_scale"],
        values["module_scale"],
        values["stoichiometry"],
        n_modules=values["modules"],
        n_genes=values["genes"],
        n_comps=values["compounds"],
        epochs=1,
        device="cpu",
        verbose=False,
    )
    flux, balance = predict_scfea(
        fitted.net,
        X,
        values["stoichiometry"],
        n_modules=values["modules"],
        n_genes=values["genes"],
        n_comps=values["compounds"],
        device="cpu",
    )

    np.testing.assert_allclose(
        flux, legacy_flux.numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        balance, legacy_balance.numpy(), rtol=1e-6, atol=1e-6
    )


def legacy_gradient(
    legacy,
    network,
    optimizer,
    values,
) -> np.ndarray:
    X = values["expression"]
    gene_expression = values["gene_expression"]
    module_scale = values["module_scale"]
    gene_scale = values["gene_scale"]
    stoichiometry = values["stoichiometry"]
    n_modules = values["modules"]
    n_genes = values["genes"]
    n_comps = values["compounds"]
    gradient = np.zeros((values["cells"], n_genes, n_modules), dtype=np.float32)
    dataset = legacy.MyDataset(X, gene_scale, module_scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    module_genes = [{"g0", "g1"}, {"g2"}]

    for cell_index, (batch_x, batch_gene_scale, batch_module_scale) in enumerate(
        loader
    ):
        flux, balance = network(
            batch_x, n_modules, n_genes, n_comps, stoichiometry
        )
        loss = legacy.myLoss(
            flux,
            balance,
            lamb1=1,
            lamb2=1,
            lamb3=1,
            lamb4=1e-2,
            geneScale=batch_gene_scale,
            moduleScale=batch_module_scale,
        )[0]
        optimizer.zero_grad()
        loss.backward()
        cell_gradient = np.zeros((n_genes, n_modules), dtype=np.float32)
        parameters = list(network.named_parameters())
        for module in range(n_modules):
            first = parameters[module * 3][1].grad
            second = parameters[module * 3 + 1][1].grad
            parameter_gradient = torch.matmul(
                torch.t(first), torch.t(second)
            ).reshape(-1)
            selected = gene_expression[cell_index].numpy().astype(
                np.float32, copy=True
            )
            selected[
                [gene not in module_genes[module] for gene in ["g0", "g1", "g2"]]
            ] = 0
            with np.errstate(divide="ignore", invalid="ignore"):
                unit = selected / selected
            cell_gradient[:, module] = parameter_gradient.numpy() * unit
        gradient[cell_index] = cell_gradient
        optimizer.step()
    return gradient


def test_gradient_matches_legacy_core() -> None:
    torch.set_num_threads(1)
    legacy = load_legacy_module()
    values = make_core_inputs()
    legacy_net, legacy_optimizer = train_legacy(
        legacy,
        values["expression"],
        values["gene_scale"],
        values["module_scale"],
        values["stoichiometry"],
        n_modules=values["modules"],
        n_genes=values["genes"],
        n_comps=values["compounds"],
    )
    expected = legacy_gradient(
        legacy, legacy_net, legacy_optimizer, values
    )

    fitted = fit_scfea(
        values["expression"],
        values["gene_scale"],
        values["module_scale"],
        values["stoichiometry"],
        n_modules=values["modules"],
        n_genes=values["genes"],
        n_comps=values["compounds"],
        epochs=1,
        device="cpu",
        verbose=False,
    )
    actual = compute_scfea_gradient(
        fitted.net,
        fitted.optimizer,
        values["expression"],
        values["gene_scale"],
        values["module_scale"],
        values["gene_expression"],
        values["module_masks"],
        values["stoichiometry"],
        n_modules=values["modules"],
        n_genes=values["genes"],
        n_comps=values["compounds"],
        device="cpu",
        verbose=False,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
