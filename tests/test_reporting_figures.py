import numpy as np
from PIL import Image

from scFEAann.reporting.embedding import compute_flux_embedding
from scFEAann.reporting.exports import build_figure_source_data
from scFEAann.reporting.figures import (
    export_figure,
    plot_figure1,
    plot_figure2,
    plot_figure3,
    plot_loss_history,
)
from scFEAann.reporting.qa import audit_layout
from scFEAann.reporting.tables import extract_report_results


def make_source_data(reporting_adata):
    report = extract_report_results(reporting_adata, top_n=2)
    flux_coordinates, _ = compute_flux_embedding(
        reporting_adata.obsm["scFEA_flux"], method="pca"
    )
    source_data = build_figure_source_data(
        report,
        transcriptome_embedding=np.asarray(reporting_adata.obsm["X_umap"]),
        flux_embedding=flux_coordinates,
    )
    return report, source_data


def test_figure1_exports_vector_raster_and_grayscale(
    tmp_path, reporting_adata
) -> None:
    report, source_data = make_source_data(reporting_adata)

    paths = plot_figure1(report, source_data, tmp_path, dpi=150)

    suffixes = {path.name for path in paths}
    assert {
        "Fig1.pdf",
        "Fig1.svg",
        "Fig1.png",
        "Fig1_grayscale.png",
    } <= suffixes
    with Image.open(tmp_path / "Fig1.png") as image:
        assert image.width > image.height


def test_all_main_figures_and_supplement_export(
    tmp_path, reporting_adata
) -> None:
    report, source_data = make_source_data(reporting_adata)

    paths = [
        *plot_figure2(report, source_data, tmp_path, dpi=150),
        *plot_figure3(report, source_data, tmp_path, dpi=150),
        *plot_loss_history(report, source_data, tmp_path, dpi=150),
    ]

    names = {path.name for path in paths}
    assert "Fig2.pdf" in names
    assert "Fig3.pdf" in names
    assert "Supp_Fig_loss.pdf" in names


def test_export_figure_rejects_empty_format_list(tmp_path) -> None:
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots(figsize=(3.5, 2.5))
    try:
        export_figure(
            fig,
            tmp_path / "empty",
            size_inches=(3.5, 2.5),
            formats=[],
        )
    except ValueError as error:
        assert "formats" in str(error)
    finally:
        plt.close(fig)


def test_audit_layout_detects_text_outside_canvas() -> None:
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots(figsize=(3.0, 2.0))
    fig.text(1.2, 1.2, "outside")
    try:
        issues = audit_layout(fig)
    finally:
        plt.close(fig)

    assert any(
        severity == "WARN" and "clipped outside" in message
        for severity, message in issues
    )
