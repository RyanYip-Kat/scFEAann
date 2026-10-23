"""Training, inference, and gradient loops for scFEA."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import MyDataset
from .flux import FLUX
from .loss import scfea_loss


LEARN_RATE = 0.008
LAMB_BA = 1.0
LAMB_NG = 1.0
LAMB_CELL = 1.0
LAMB_MOD = 1e-2
DEFAULT_SEED = 16


@dataclass
class FitResult:
    net: FLUX
    optimizer: torch.optim.Optimizer
    loss_history: dict[str, np.ndarray]
    elapsed_seconds: float

    @property
    def total_losses(self) -> np.ndarray:
        return self.loss_history["total"]


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_float_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, dtype=torch.float32)


def fit_scfea(
    expression: torch.Tensor,
    gene_scale: torch.Tensor,
    module_scale: torch.Tensor,
    stoichiometry: torch.Tensor,
    *,
    n_modules: int,
    n_genes: int,
    n_comps: int,
    epochs: int,
    device: str | torch.device = "auto",
    seed: int = DEFAULT_SEED,
    lamb_module: float = LAMB_MOD,
    safe_module_correlation: bool = False,
    verbose: bool = True,
) -> FitResult:
    """Train the scFEA network using the original loss and optimizer."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    selected_device = resolve_device(device)
    torch.manual_seed(seed)
    network = FLUX(
        expression, n_modules, f_in=n_genes, f_out=1
    ).to(selected_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARN_RATE)
    dataset = MyDataset(expression, gene_scale, module_scale)
    loader = DataLoader(
        dataset=dataset,
        batch_size=expression.shape[0],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    stoichiometry_device = stoichiometry.to(selected_device)

    history = {
        "total": [],
        "balance": [],
        "negative": [],
        "cell": [],
        "module": [],
    }
    progress = tqdm(range(epochs), disable=not verbose)
    started = time.perf_counter()
    network.train()
    for _ in progress:
        totals = np.zeros(5, dtype=np.float64)
        for batch_x, batch_gene_scale, batch_module_scale in loader:
            input_batch = _to_float_tensor(batch_x, selected_device)
            gene_scale_batch = _to_float_tensor(
                batch_gene_scale, selected_device
            )
            module_scale_batch = _to_float_tensor(
                batch_module_scale, selected_device
            )
            flux, balance = network(
                input_batch,
                n_modules,
                n_genes,
                n_comps,
                stoichiometry_device,
            )
            components = scfea_loss(
                flux,
                balance,
                lamb_balance=LAMB_BA,
                lamb_nonnegative=LAMB_NG,
                lamb_cell=LAMB_CELL,
                lamb_module=lamb_module,
                gene_scale=gene_scale_batch,
                module_scale=module_scale_batch,
                safe_module_correlation=safe_module_correlation,
            )
            optimizer.zero_grad()
            components[0].backward()
            optimizer.step()
            totals += np.asarray(
                [item.detach().cpu().item() for item in components],
                dtype=np.float64,
            )

        history["total"].append(totals[0])
        history["balance"].append(totals[1])
        history["negative"].append(totals[2])
        history["cell"].append(totals[3])
        history["module"].append(totals[4])

    elapsed = time.perf_counter() - started
    return FitResult(
        net=network,
        optimizer=optimizer,
        loss_history={
            key: np.asarray(values, dtype=np.float64)
            for key, values in history.items()
        },
        elapsed_seconds=elapsed,
    )


def predict_scfea(
    net: FLUX,
    expression: torch.Tensor,
    stoichiometry: torch.Tensor,
    *,
    n_modules: int,
    n_genes: int,
    n_comps: int,
    device: str | torch.device = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict flux and compound balance with the original batch size of one."""
    selected_device = resolve_device(device)
    stoichiometry_device = stoichiometry.to(selected_device)
    dataset = MyDataset(expression, torch.zeros(len(expression)), torch.zeros((len(expression), 1)))
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    flux = np.zeros((len(expression), n_modules), dtype=np.float32)
    balance = np.zeros((len(expression), n_comps), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for index, (batch_x, _, _) in enumerate(loader):
            output_flux, output_balance = net(
                _to_float_tensor(batch_x, selected_device),
                n_modules,
                n_genes,
                n_comps,
                stoichiometry_device,
            )
            flux[index, :] = output_flux.detach().cpu().numpy()
            balance[index, :] = output_balance.detach().cpu().numpy()
    return flux, balance


def compute_scfea_gradient(
    net: FLUX,
    optimizer: torch.optim.Optimizer,
    expression: torch.Tensor,
    gene_scale: torch.Tensor,
    module_scale: torch.Tensor,
    gene_expression: torch.Tensor,
    module_masks: torch.Tensor,
    stoichiometry: torch.Tensor,
    *,
    n_modules: int,
    n_genes: int,
    n_comps: int,
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> np.ndarray:
    """Compute the original per-cell gene-to-module gradient array."""
    selected_device = resolve_device(device)
    stoichiometry_device = stoichiometry.to(selected_device)
    dataset = MyDataset(expression, gene_scale, module_scale)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    gradient = np.zeros(
        (expression.shape[0], gene_expression.shape[1], n_modules),
        dtype=np.float32,
    )
    iterator = tqdm(loader, disable=not verbose)
    net.train()
    for cell_index, (batch_x, batch_gene_scale, batch_module_scale) in enumerate(
        iterator
    ):
        input_batch = _to_float_tensor(batch_x, selected_device)
        flux, balance = net(
            input_batch,
            n_modules,
            n_genes,
            n_comps,
            stoichiometry_device,
        )
        loss, _, _, _, _ = scfea_loss(
            flux,
            balance,
            lamb_balance=LAMB_BA,
            lamb_nonnegative=LAMB_NG,
            lamb_cell=LAMB_CELL,
            lamb_module=LAMB_MOD,
            gene_scale=_to_float_tensor(
                batch_gene_scale, selected_device
            ),
            module_scale=_to_float_tensor(
                batch_module_scale, selected_device
            ),
        )
        optimizer.zero_grad()
        loss.backward()

        parameters = list(net.named_parameters())
        if len(parameters) != n_modules * 3:
            raise RuntimeError(
                "gradient calculation expects exactly three parameters per module"
            )
        cell_gradient = np.zeros(
            (gene_expression.shape[1], n_modules), dtype=np.float32
        )
        for module in range(n_modules):
            first_weight = parameters[module * 3][1]
            second_weight = parameters[module * 3 + 1][1]
            if first_weight.grad is None or second_weight.grad is None:
                raise RuntimeError("module parameters did not produce gradients")
            parameter_gradient = torch.matmul(
                torch.t(first_weight.grad), torch.t(second_weight.grad)
            ).reshape(-1)
            selected = gene_expression[cell_index].to(selected_device).clone()
            module_mask = module_masks[module].to(
                device=selected_device, dtype=torch.bool
            )
            selected = selected * module_mask
            with np.errstate(divide="ignore", invalid="ignore"):
                unit_selection = (
                    selected / selected
                ).detach().cpu().numpy()
            cell_gradient[:, module] = (
                parameter_gradient.detach().cpu().numpy() * unit_selection
            )
        gradient[cell_index, :, :] = cell_gradient
        optimizer.step()
    return gradient
