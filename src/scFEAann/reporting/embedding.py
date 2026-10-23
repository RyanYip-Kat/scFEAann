"""Embedding coordinates used by the scFEAann report."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _pca_coordinates(
    values: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, float]:
    components = min(n_components, values.shape[0] - 1, values.shape[1])
    if components < 2:
        raise ValueError("at least two samples and two features are required")
    model = PCA(n_components=components, random_state=16)
    coordinates = model.fit_transform(values)
    explained = float(np.sum(model.explained_variance_ratio_[:2]))
    return coordinates[:, :2], explained


def compute_flux_embedding(
    flux: np.ndarray,
    *,
    method: str = "umap",
    random_state: int = 16,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute a two-dimensional embedding from cell-by-module flux values.

    Features are z-scored before PCA to prevent high-scale modules from
    dominating. UMAP and t-SNE are computed on up to 30 leading PCs. If a
    request cannot be fulfilled, PCA coordinates are returned and the fallback
    is recorded.
    """
    flux = np.asarray(flux, dtype=float)
    if flux.ndim != 2:
        raise ValueError("flux must be two-dimensional")
    if not np.isfinite(flux).all():
        raise ValueError("flux must contain only finite values")
    standardized = StandardScaler().fit_transform(flux)
    method = method.lower()

    if method == "pca":
        coordinates, explained = _pca_coordinates(
            standardized, n_components=2
        )
        return coordinates, {
            "requested_method": "pca",
            "method": "pca",
            "standardized": True,
            "explained_variance_2d": explained,
            "fallback_reason": None,
        }

    pca_components = min(30, standardized.shape[1], standardized.shape[0] - 1)
    if pca_components < 2:
        raise ValueError("at least two samples and two features are required")
    pca_model = PCA(n_components=pca_components, random_state=random_state)
    pca_values = pca_model.fit_transform(standardized)
    fallback_reason = None
    try:
        if method == "umap":
            import umap

            coordinates = umap.UMAP(
                n_components=2,
                n_neighbors=min(30, max(2, flux.shape[0] - 1)),
                min_dist=0.3,
                n_jobs=1,
                random_state=random_state,
            ).fit_transform(pca_values)
            actual_method = "umap"
        elif method == "tsne":
            from sklearn.manifold import TSNE

            perplexity = min(
                30.0,
                max(2.0, (flux.shape[0] - 1) / 3.0),
            )
            coordinates = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            ).fit_transform(pca_values)
            actual_method = "tsne"
        else:
            raise ValueError("method must be 'umap', 'tsne', or 'pca'")
    except Exception as error:
        coordinates = pca_values[:, :2]
        actual_method = "pca"
        fallback_reason = f"{type(error).__name__}: {error}"

    return np.asarray(coordinates, dtype=float), {
        "requested_method": method,
        "method": actual_method,
        "standardized": True,
        "explained_variance_2d": float(
            np.sum(pca_model.explained_variance_ratio_[:2])
        ),
        "fallback_reason": fallback_reason,
    }


def get_transcriptome_embedding(adata) -> tuple[np.ndarray, dict[str, object]]:
    """Return an existing expression embedding or PCA fallback coordinates."""
    for key in ("X_umap", "X_tsne"):
        if key in adata.obsm:
            coordinates = np.asarray(adata.obsm[key], dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] < 2:
                raise ValueError(f"adata.obsm['{key}'] must have at least two columns")
            return coordinates[:, :2], {
                "method": key,
                "fallback_reason": None,
            }

    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] > 2000:
        variances = np.nanvar(matrix, axis=0)
        selected = np.argsort(variances)[-2000:]
        matrix = matrix[:, selected]
    scaled = StandardScaler().fit_transform(matrix)
    coordinates, explained = _pca_coordinates(scaled, n_components=2)
    return coordinates, {
        "method": "pca",
        "explained_variance_2d": explained,
        "fallback_reason": "no X_umap or X_tsne in obsm",
    }
