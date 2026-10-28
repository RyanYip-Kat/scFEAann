# Supermodule Network Visualization Design

## Goal

Recreate the scientific intent of the two supplied scFEA network references
using the current AnnData result:

- a supermodule-level network for class-specific flux results;
- a complete module-metabolite bipartite factor graph for structural context.

## Figure 5: Supermodule network

Nodes are the 22 supermodule classes. Edges are undirected class pairs that
share one or more annotated metabolites. Edge width represents the number of
shared compounds.

- a: node color represents topology-derived spectral clusters.
- b: node color represents continuous flux difference between a user-selected
  contrast.

Both panels share one deterministic layout. Node size encodes module count.
Source data include nodes, edges, layout coordinates, and contrast statistics.

## Supplementary Figure 3: Factor graph

Build a bipartite graph:

- module nodes from the human module annotation;
- compound nodes from module inputs and outputs;
- edges from module-compound participation.

Inputs and outputs are edge attributes. Visual styling follows the reference:

- compounds: pink ellipses;
- modules: green diamonds;
- gray edges.

All node, edge, and layout data are exported. Labels are limited to high-degree
nodes to preserve readability.

## API

```python
plot_supermodule_network(
    adata,
    output_dir,
    *,
    contrast_key="dataset",
    contrast=None,
    dpi=300,
)

plot_factor_graph(
    output_dir,
    *,
    dpi=300,
)
```

The report API also gains `network_contrast_key` and `network_contrast`.
If no explicit contrast is supplied, the first two sorted levels are used.

## Integration

`export_scfeaann_report()` automatically adds network figures and tables when
supermodule result keys exist.

Network files:

- `network_nodes.csv`
- `network_edges.csv`
- `network_layout.csv`
- `network_supermodule_summary.csv`
- `factor_graph_nodes.csv`
- `factor_graph_edges.csv`
- `factor_graph_layout.csv`

## Verification

- Graph construction tests for nodes, edges, directions, and layout reproducibility.
- Figure export tests for PDF, SVG, PNG, and grayscale PNG.
- Report integration and manifest tests.
- Programmatic and visual QA of network previews.
