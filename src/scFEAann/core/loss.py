"""The original scFEA loss function."""

from __future__ import annotations

import torch


def pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Correlation coefficient used by the original scFEA loss."""
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    numerator = x_centered.dot(y_centered)
    denominator = torch.norm(x_centered, 2) * torch.norm(y_centered, 2)
    return numerator / denominator


def scfea_loss(
    flux: torch.Tensor,
    balance: torch.Tensor,
    *,
    lamb_balance: float = 0.2,
    lamb_nonnegative: float = 0.2,
    lamb_cell: float = 0.2,
    lamb_module: float = 0.2,
    gene_scale: torch.Tensor | None = None,
    module_scale: torch.Tensor | None = None,
    safe_module_correlation: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Return total loss and its balance, sign, cell, and module components."""
    if gene_scale is None or module_scale is None:
        raise ValueError("gene_scale and module_scale are required")

    balance_total = torch.sum(torch.pow(balance, 2), dim=1)
    nonnegative_total = torch.sum(torch.abs(flux) - flux, dim=1)

    cell_difference = torch.pow(
        torch.sum(flux, dim=1) - gene_scale, 2
    )
    if torch.sum(cell_difference > 0) == flux.shape[0]:
        cell_total = torch.pow(cell_difference, 0.5)
    else:
        cell_total = cell_difference

    if lamb_module > 0:
        correlations = flux.new_empty((flux.shape[0],))
        for index in range(flux.shape[0]):
            correlation = pearsonr(
                flux[index, :], module_scale[index, :]
            )
            if safe_module_correlation and not torch.isfinite(
                correlation
            ).item():
                correlation = flux.new_ones(())
            correlations[index] = correlation
        module_total = flux.new_ones((flux.shape[0],)) - torch.abs(
            correlations
        )
    else:
        module_total = flux.new_zeros((flux.shape[0],))

    balance_loss = torch.sum(lamb_balance * balance_total)
    nonnegative_loss = torch.sum(lamb_nonnegative * nonnegative_total)
    cell_loss = torch.sum(lamb_cell * cell_total)
    module_loss = torch.sum(lamb_module * module_total)
    total = balance_loss + nonnegative_loss + cell_loss + module_loss
    return total, balance_loss, nonnegative_loss, cell_loss, module_loss
