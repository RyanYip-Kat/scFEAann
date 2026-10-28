# scFEAann Scientific Reporting Design

## Goal

Add a reproducible reporting subsystem that converts an scFEAann result AnnData
into English SCI-style figures, statistical summary tables, figure source data,
and an auditable run manifest. The scFEA core and existing AnnData result schema
remain unchanged.

## Scientific scope

The report treats dataset and cluster as descriptive grouping variables, not as
training partitions. It does not attach cell-level p-values because thousands of
cells are not independent biological replicates. The main report emphasizes:

- cell composition across dataset and cluster;
- cross-dataset consistency of cluster-level metabolic profiles;
- relative module-flux and compound-balance patterns;
- distribution of the most group-variable features;
- agreement between transcriptome and flux representations.

## Public API

```python
export_scfeaann_report(
    adata,
    output_dir,
    *,
    dataset_key="dataset",
    cluster_key="cluster",
    top_n=30,
    embedding_method="umap",
    random_state=16,
    dpi=300,
)
```

The function writes files and returns a dictionary describing generated tables,
figures, source data, and warnings.

## Figure plan

### Figure 1: Cohort and metabolic structure

- a: stacked cell counts, x=dataset, fill=cluster, with group n labels.
- b: D1-versus-D2 cluster centroid scatter across modules, one color/marker per
  cluster, identity line, and Pearson r annotations.
- c: heatmap of top group-variable modules across nested dataset-cluster groups.
- d: distributions of the top four group-variable modules.

### Figure 2: Metabolic balance

- a: heatmap of top group-variable compounds across nested groups.
- b: distributions of the top four group-variable compounds.
- c: module-wise between-group effect ranking.
- d: compound-wise between-group effect ranking.

### Figure 3: Transcriptome and flux embeddings

- a: existing transcriptome embedding, colored by cluster.
- b: flux embedding, colored by cluster.
- c: existing transcriptome embedding, colored by dataset.
- d: flux embedding, colored by dataset.

### Supplementary figure

Loss-history curves for total, balance, negative, cell, and module terms.

## Table plan

Summary workbook:

```text
report_tables.xlsx
├── group_counts
├── dataset_consistency
├── module_group_summary
├── compound_group_summary
├── module_variance_ranking
├── compound_variance_ranking
└── loss_history
```

Separate compressed cell-level tables:

```text
cell_flux.csv.gz
cell_balance.csv.gz
cell_metadata.csv.gz
```

Figure source data:

```text
figure_source_data/
├── Fig1a_group_counts.csv
├── Fig1b_dataset_consistency.csv
├── Fig1c_module_heatmap.csv
├── Fig1d_top_modules.csv
├── Fig2a_balance_heatmap.csv
├── Fig2b_top_compounds.csv
├── Fig2c_module_effects.csv
├── Fig2d_compound_effects.csv
├── Fig3a_transcriptome_embedding.csv
├── Fig3b_flux_embedding.csv
└── Supp_loss_history.csv
```

Optional gradient summary:

```text
gradient_feature_summary.csv
gradient_top_by_group.csv
```

These are produced only when `obsm["scFEA_grad"]` exists. Full dense gradients
are never written as CSV because the original normalization makes most entries
NaN.

## Group statistics

For every feature and nested group, calculate:

- n cells;
- mean;
- SD;
- SEM;
- median;
- 25th and 75th percentiles;
- minimum and maximum.

Rank features by:

```text
between-group variance / (between-group variance + mean within-group variance)
```

The ratio is descriptive and is not a significance test.

## Embedding

The transcriptome embedding uses the first available keys in this order:

1. `obsm["X_umap"]`
2. `obsm["X_tsne"]`
3. PCA computed from the existing AnnData expression matrix

The flux embedding uses PCA on `obsm["scFEA_flux"]` followed by UMAP when
`embedding_method="umap"` and `umap-learn` is available. Otherwise it falls back
to the first two flux PCs. Coordinates are exported even when plotting fails.

## Style

- English labels only.
- Single-column width 3.5 inches; double-column width 7.2 inches.
- Axis labels 7 pt; tick labels 6.5 pt; panel labels 8 pt bold.
- Font preference: Arial, Helvetica, Liberation Sans, DejaVu Sans.
- PDF font type 42; SVG text kept as text.
- Colorblind-safe Okabe-Ito palette.
- Dataset encoded by both color and marker shape.
- Cluster colors remain fixed across all figures.
- Heatmaps use `RdBu_r` with a symmetric color scale centered at zero.

## Exports

Each figure is written as:

```text
Fig1.pdf
Fig1.svg
Fig1.png
Fig1_grayscale.png
```

PNG output uses 300 dpi. Source CSV files are written before figure rendering.
A `report_manifest.json` records input dimensions, grouping keys, feature counts,
software versions, figure dimensions, and generated files.

## Error handling

Raise actionable `ValueError` for:

- missing flux or balance results;
- missing grouping columns;
- fewer than two datasets or fewer than two clusters;
- fewer than two shared cluster labels across datasets;
- mismatched result-row counts;
- invalid top-feature counts;
- invalid output directory conflicts.

UMAP failure is non-fatal when PCA fallback coordinates are available, but the
fallback is recorded in the manifest.

## Verification

- Unit tests cover group statistics, feature ranking, dataset consistency, and
  embedding coordinate generation.
- Integration tests use a synthetic AnnData and export all three main figures,
  source tables, workbook, and manifest to a temporary directory.
- Figure tests assert dimensions and output formats.
- Programmatic layout checks run before final vector export.
- Rendered PNG previews are visually inspected for clipping, overlap, label
  alignment, color distinguishability, and consistency across panels.
