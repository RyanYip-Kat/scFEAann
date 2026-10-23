"""The scFEA module-to-flux neural network."""

from __future__ import annotations

import torch
import torch.nn as nn


class FLUX(nn.Module):
    """Original scFEA module encoder.

    The ``matrix`` argument is retained for source compatibility; the original
    implementation does not use it when constructing the network.
    """

    def __init__(
        self,
        matrix: object,
        n_modules: int,
        f_in: int = 50,
        f_out: int = 1,
    ) -> None:
        super().__init__()
        self.inSize = f_in
        self.m_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.inSize, 8, bias=False),
                    nn.Tanhshrink(),
                    nn.Linear(8, f_out),
                    nn.Tanhshrink(),
                )
                for _ in range(n_modules)
            ]
        )

    def updateC(
        self,
        flux: torch.Tensor,
        n_comps: int,
        stoichiometry: torch.Tensor,
    ) -> torch.Tensor:
        balance = torch.zeros(
            (flux.shape[0], n_comps),
            device=flux.device,
            dtype=flux.dtype,
        )
        for compound in range(balance.shape[1]):
            balance[:, compound] = torch.sum(
                flux * stoichiometry[compound, :], dim=1
            )
        return balance

    def forward(
        self,
        expression: torch.Tensor,
        n_modules: int,
        n_genes: int,
        n_comps: int,
        stoichiometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flux: torch.Tensor | None = None
        for module in range(n_modules):
            start = module * n_genes
            block = expression[:, start : start + n_genes]
            module_flux = self.m_encoder[module](block)
            flux = (
                module_flux
                if flux is None
                else torch.cat((flux, module_flux), dim=1)
            )
        if flux is None:
            raise ValueError("n_modules must be positive")
        balance = self.updateC(flux, n_comps, stoichiometry)
        return flux, balance
