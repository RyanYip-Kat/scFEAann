# scFEAann P0 Performance Optimization

## Goal

Reduce preprocessing peak memory, accelerate inference, and avoid building the
full module tensor when only selected supermodule classes are requested. Public
results must remain numerically compatible with the current implementation.

## Constraints

- Network architecture, loss formulas, optimizer, seed, and training batch stay
  unchanged.
- Existing public result keys stay unchanged.
- The new prediction batch size is configurable and recorded in metadata.
- Sparse inputs use a fast path only when MAGIC is disabled.
- Selected-class execution must not require the full 168-module X tensor.

## Preprocessing

For sparse non-MAGIC input:

1. Validate sparse data values without densifying the full matrix.
2. Compute the maximum and row sums directly from sparse storage.
3. Apply log2 to sparse values when required.
4. Select overlap genes and densify only the overlap subset.
5. Build module X directly in float32 without 168 DataFrame copies or concat.
6. Compute module-scale values from overlap expression and module masks.

MAGIC keeps the dense preprocessing path.

## Inference

Add:

```python
prediction_batch_size=1024
```

Inference moves contiguous cell chunks through the network and writes into
preallocated flux/balance arrays. Batch one remains available for strict
reference comparison.

## Supermodule execution

Add a prepared-data mode that contains:

- overlap gene expression;
- module masks;
- gene scale;
- module list;
- stoichiometry;
- annotation metadata.

Selected classes materialize only their own module block. The full X tensor is
built only when the global model is requested.

## Verification

- Sparse and dense paths compare exactly for X, gene scale, module scale.
- New preprocessing equals the former implementation under non-MAGIC input.
- Batched inference equals batch-one inference within a strict tolerance.
- Class-only supermodule execution equals execution from the full prepared
  tensor for the same class.
- Full test suite remains green.
