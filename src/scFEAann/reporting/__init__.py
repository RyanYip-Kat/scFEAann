"""Scientific reporting utilities for scFEAann results."""

from .embedding import compute_flux_embedding, get_transcriptome_embedding
from .exports import (
    build_figure_source_data,
    write_cell_tables,
    write_figure_source_data,
    write_summary_tables,
)
from .figures import (
    export_figure,
    plot_figure1,
    plot_figure2,
    plot_figure3,
    plot_figure4_supermodules,
    plot_loss_history,
    plot_supp_supermodule_embedding,
)
from .report import export_scfeaann_report
from .qa import audit_layout
from .network import (
    FactorGraphData,
    SupermoduleNetworkData,
    build_factor_graph,
    build_supermodule_network,
    plot_factor_graph,
    plot_supermodule_network,
    write_network_tables,
)
from .supermodules import (
    SupermoduleReport,
    extract_supermodule_report,
    write_supermodule_tables,
)
from .tables import (
    ReportResults,
    dataset_consistency,
    extract_report_results,
    rank_feature_effects,
    summarize_features,
)

__all__ = [
    "ReportResults",
    "FactorGraphData",
    "SupermoduleNetworkData",
    "SupermoduleReport",
    "audit_layout",
    "build_figure_source_data",
    "compute_flux_embedding",
    "dataset_consistency",
    "extract_report_results",
    "export_figure",
    "export_scfeaann_report",
    "extract_supermodule_report",
    "build_factor_graph",
    "build_supermodule_network",
    "get_transcriptome_embedding",
    "plot_figure1",
    "plot_figure2",
    "plot_figure3",
    "plot_figure4_supermodules",
    "plot_factor_graph",
    "plot_loss_history",
    "plot_supp_supermodule_embedding",
    "plot_supermodule_network",
    "rank_feature_effects",
    "summarize_features",
    "write_cell_tables",
    "write_figure_source_data",
    "write_supermodule_tables",
    "write_network_tables",
    "write_summary_tables",
]
