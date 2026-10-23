"""AnnData-independent scFEA algorithms."""

from .dataset import MyDataset
from .flux import FLUX
from .loss import scfea_loss
from .training import (
    FitResult,
    compute_scfea_gradient,
    fit_scfea,
    predict_scfea,
)

__all__ = [
    "FLUX",
    "FitResult",
    "MyDataset",
    "compute_scfea_gradient",
    "fit_scfea",
    "predict_scfea",
    "scfea_loss",
]
