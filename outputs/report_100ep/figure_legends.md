# scFEAann figure legends

**Figure 1. Cohort composition and metabolic module structure.**
(a) Cell counts by dataset and cluster. (b) Pearson correlation between
dataset-level mean module-flux centroids within each cluster across
168 modules. (c) Heatmap of mean module flux for the top
group-variable modules. Values are z-scored across the six dataset-cluster
groups independently for each module; the color scale is centered at zero.
(d) Distributions of the four most group-variable modules. Box plots show the
median, interquartile range, and 1.5 x IQR whiskers; outliers are omitted from
the display, and all cell-level values are provided in the source-data tables.
Cell counts: D1 / corneal basal epithelium: 3954, D1 / corneal superficial epithelium: 6246, D1 / corneal suprabasal epithelial cells: 1800, D2 / corneal basal epithelium: 2876, D2 / corneal superficial epithelium: 1106, D2 / corneal suprabasal epithelial cells: 1510.

**Figure 2. Compound-balance structure and feature effects.**
(a) Heatmap of mean compound balance for the most group-variable compounds,
z-scored across nested groups independently for each compound. (b)
Distributions of the four most group-variable compounds, with the same box-plot
definition as Figure 1. (c, d) Module and compound effect rankings. The
effect ratio is between-group variance divided by the sum of between-group and
mean within-group variance. It is descriptive and is not a hypothesis test.

**Figure 3. Transcriptome and flux representations.**
(a, c) Transcriptome embedding obtained from `X_umap` and
colored by cluster and dataset, respectively. (b, d) Flux embedding computed
from standardized cell-by-module flux values using UMAP, colored by
cluster and dataset. Dataset colors are paired with circle/square markers.
Cluster colors are paired with circle/square/triangle markers.

**Supplementary Figure. Training loss.**
Total scFEA loss and its balance, non-negative, cell-variation, and
module-variation components across training epochs. The total-loss first and last values are 26650 and 2513.76, respectively.

**Statistics.**
No cell-level p-values are reported. Individual cells are not treated as
independent biological replicates. All group sizes and raw feature summaries
are provided in `report_tables.xlsx`. Exact module reactions, input/output
KEGG identifiers, supermodule identifiers, and member genes are listed in the
`module_annotation` sheet and `module_annotation.csv`.
