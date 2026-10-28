# Supermodule Reporting Design

## Goal

Extend the existing scientific report with supermodule-specific statistics,
figures, source data, and network-ready node tables when class-specific scFEA
results exist in AnnData.

## Main feature

`obsm["scFEA_supermodule_flux_mean"]` is the default supermodule feature for
figures and effect ranking. `flux_sum` is retained in tables and network node
exports but is not used for the main comparisons because classes differ in
module count.

## Figure 4

- a: 22 classes x dataset-cluster groups, per-class z-scored mean flux heatmap.
- b: distributions of the six most group-variable class features.
- c: 22 x 22 Spearman correlation matrix across cells.
- d: global-versus-class model agreement, horizontal Pearson/Spearman dot plot.

## Supplementary Figure 2

- a: PCA embedding of the 22 class features, colored by cluster.
- b: same PCA embedding, colored by dataset.
- c: representative global-versus-class cell-level scatter panels.
- d: supermodule effect ranking.

## Tables

Add worksheets:

- `supermodule_group_summary`
- `supermodule_effects`
- `supermodule_model_agreement`
- `supermodule_correlation`
- `supermodule_annotation`

Separate exports:

- `supermodule_node_features_by_cell.csv.gz`
- `supermodule_node_features_by_group.csv`
- `supermodule_annotation.csv`

## Integration

`export_scfeaann_report()` detects:

```python
"scFEA_supermodule_flux_mean" in adata.obsm
```

If present, it writes Figure 4 and Supplementary Figure 2 automatically and adds
their files to the manifest. Existing reports without supermodule output remain
unchanged.

## Statistical scope

- Use descriptive group statistics and effect ratios.
- Do not report cell-level p-values.
- Global aggregation uses the same class-to-module mapping as class models.
- Agreement metrics are Pearson r, Spearman rho, RMSE, and MAE.

## Verification

- Unit tests for group summaries, effects, agreement, and correlations.
- Figure export tests for all formats and source data.
- Integration tests verify automatic detection and workbook sheets.
- Visual QA runs on generated PNGs before completion.
