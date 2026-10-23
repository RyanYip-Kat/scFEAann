"""AnnData API for the scFEA metabolic flux and gradient algorithms."""

from .api import (
    calculate_scfeaann_gradient,
    run_scfeaann,
    run_scfeaann_supermodules,
)
from .modules import get_module_annotation
from .results import (
    get_scfeaann_gradient,
    get_scfeaann_supermodule_balance,
    get_scfeaann_supermodule_flux,
)
from .supermodules import load_supermodules

__version__ = "0.1.0"


def export_scfeaann_report(*args, **kwargs):
    """Export SCI-style figures and tables; imports plotting dependencies lazily."""
    from .reporting.report import export_scfeaann_report as _export_report

    return _export_report(*args, **kwargs)


__all__ = [
    "calculate_scfeaann_gradient",
    "export_scfeaann_report",
    "get_module_annotation",
    "get_scfeaann_gradient",
    "get_scfeaann_supermodule_balance",
    "get_scfeaann_supermodule_flux",
    "load_supermodules",
    "run_scfeaann",
    "run_scfeaann_supermodules",
]
