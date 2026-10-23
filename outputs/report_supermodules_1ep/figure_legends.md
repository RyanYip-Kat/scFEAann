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
Cell counts: D1 / corneal basal epithelium: 34, D1 / corneal superficial epithelium: 34, D1 / corneal suprabasal epithelial cells: 33, D2 / corneal basal epithelium: 33, D2 / corneal superficial epithelium: 33, D2 / corneal suprabasal epithelial cells: 33.

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
module-variation components across training epochs. The total-loss first and last values are 304.731 and 304.731, respectively.

**Statistics.**
No cell-level p-values are reported. Individual cells are not treated as
independent biological replicates. All group sizes and raw feature summaries
are provided in `report_tables.xlsx`. Exact module reactions, input/output
KEGG identifiers, supermodule identifiers, and member genes are listed in the
`module_annotation` sheet and `module_annotation.csv`.


**Figure 4. Supermodule-specific model results.**
(a) Mean flux for the 22 supermodule
classes, z-scored across the 6
dataset-cluster groups independently for each class. (b) Cell-level
distributions of the six most group-variable supermodules; box plots show the
median, interquartile range, and 1.5 x IQR whiskers, with outliers omitted for
display. (c) Spearman correlation between the 22 class-specific node features
across cells. (d) Pearson and Spearman agreement between each independently
trained class model and the same class aggregated from the global 168-module
model. RMSE and MAE for those comparisons are provided in
`supermodule_model_agreement`.

**Supplementary Figure 2. Supermodule feature embedding.**
(a, b) PCA of the 22-dimensional supermodule mean-flux profile, colored by
cluster and dataset. (c) Cell-level comparison between global aggregation and
class-specific flux for representative classes. (d) Ranking of classes by the
between-group variance ratio. All figures use `flux_mean`; `flux_sum` remains
available in the node-feature tables for network analysis.


**Figure 5. Supermodule network.**
Nodes are the 22 supermodule classes; edges connect classes sharing annotated
metabolites, and edge width represents the number of shared compounds. Node
size indicates the number of modules in each class. (a) Colors show
topology-derived spectral clusters. (b) Colors show the standardized mean-flux
difference for D2 versus D1.
Continuous standardized differences are shown to avoid arbitrary cutoff-based
categories.

**Supplementary Figure 3. Human metabolic factor graph.**
Bipartite graph connecting metabolic modules (green diamonds) to their input
and output metabolites (pink circles). Edges are derived from
`Human_M168_information.symbols.csv`; AnnData contributes no topology. Every
module node is labeled with its module ID and every compound node with its KEGG
ID. Blue annotations around the perimeter identify the 22 supermodule classes
and are connected to their module centroids by leader lines. Full compound
names, module annotations, and edges are available in
`factor_graph_nodes.csv`, `factor_graph_edges.csv`, and
`factor_graph_supermodule_labels.csv`.
