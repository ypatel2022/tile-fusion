# GNN linear-layer fusion benchmark

This CPU benchmark computes the linear part of a graph convolution:

```text
X [nodes × features] ── GEMM with W ──► H [nodes × hidden]
                                          │
normalized sparse adjacency A ── SpMM ──────┴──► Y [nodes × hidden]
```

The unfused implementation finishes all of `H = X W` before starting `Y = A H`.
Tile fusion produces a group of rows of H, then immediately consumes the rows
whose dependencies are ready. Reusing those intermediate values while they are
in cache can reduce memory traffic. The graph's sparsity pattern determines how
much work can be fused.

This follows the GEMM–SpMM experiment in *Loop Fusion in Matrix Multiplications
with Sparse Dependence*, especially sections 5.1, 5.2, and 5.2.5
([paper](https://doi.org/10.1145/3721145.3730427)). It uses the repository's existing
CPU kernels and variable tile-size inspector. It measures the two linear
operations, without activation, backward propagation, optimizer steps, or
training accuracy. Features and weights are seeded synthetic float32 values.

## Observed results

On an AMD EPYC 9655 with eight CPU threads and a 1 MiB inspector budget,
the 262,144-node local graph ran **1.85× faster** than optimized unfused AVX2
at width 64 (9.815 ms → 5.302 ms), and **1.52× faster** at width 128
(24.756 ms → 16.315 ms). Inspection took an additional 4.434 ms and 7.565 ms,
respectively. PubMed's gains were much smaller; some configurations lost.
All 2,016 timed correctness checks passed. See [all results](RESULTS.md) and
[the summary CSV](results.csv), including slowdowns and repeat variation.

## Implementations

| CSV name | Implementation |
|---|---|
| `unfused_mkl` | Full MKL GEMM followed by MKL sparse–dense multiplication |
| `unfused_avx2` | Full MKL GEMM followed by the repository's AVX2 SpMM |
| `tile_fused_avx2` | Repository inspector, tiled MKL GEMMs, and AVX2 SpMM |

Both baselines are optimized. The fused kernel uses OpenMP across tiles and
single-threaded MKL inside each tile; the baselines give the full thread count
to their GEMM. The same thread budget, matrices, and numerical operation apply
to all three. Hidden width must be a multiple of 32 because the existing AVX2
kernel processes 32 columns at a time.

This compares complete implementations: the fused path also changes GEMM tiling
and threading. In particular, a speedup with zero fused rows comes from those
execution changes, not reuse between GEMM and SpMM.

## Inputs

`prepare_inputs.py` needs only Python's standard library and the bundled PubMed
files. It writes full Matrix Market adjacency matrices with self-loops:

| Case | Nodes | Nonzeros | Purpose |
|---|---:|---:|---|
| `local_65536` | 65,536 | 1,114,112 | Synthetic ring, eight neighbors on each side; strong locality |
| `local_262144` | 262,144 | 4,456,448 | Larger version with a larger intermediate matrix |
| `shuffled_65536` | 65,536 | 1,114,112 | Identical topology with seeded random vertex relabeling |
| `pubmed` | 19,717 | 108,365 | Bundled PubMed citation graph, made undirected with self-loops |

The ordered synthetic cases deliberately provide a favorable setting for fusion.
They are not substitutes for real GNN datasets. PubMed uses its existing node
ordering and synthetic feature vectors, not its original 500 input features or
labels. All graphs use symmetric degree normalization `D^-1/2 A D^-1/2`.

## Build and run

Requires an AVX2/FMA-capable CPU, C++17 compiler, CMake, OpenMP, MKL, and METIS.
Configure MKL/METIS discovery as for the main project. From the repository root:

```bash
cmake -S tests/gnn_fusion -B build/gnn-fusion -DCMAKE_BUILD_TYPE=Release
cmake --build build/gnn-fusion --target gnn_fusion_benchmark -j2
python3 tests/gnn_fusion/prepare_inputs.py tests/gnn_fusion/inputs
python3 tests/gnn_fusion/run_suite.py build/gnn-fusion/gnn_fusion_benchmark \
    tests/gnn_fusion/inputs tests/gnn_fusion/results/run1
```

Run on CPUs allocated for the test. The suite defaults to 1 and 8 threads,
feature/hidden widths 64 and 128, and a 1 MiB inspector cache budget. Options
include `--threads`, `--widths`, and `--cache-bytes` (each accepts multiple values).
Output directories must be new, so previous results cannot be overwritten.

For one input or unequal feature/hidden widths:

```bash
build/gnn-fusion/gnn_fusion_benchmark graph.mtx 64 128 8 1048576
```

Small correctness regressions (one node, all rows fused, and no rows fused):

```bash
python3 tests/gnn_fusion/check_correctness.py build/gnn-fusion/gnn_fusion_benchmark
```

The all-fused case also exercises an empty trailing tile emitted by the existing
inspector. The executor now skips that tile before reading its first row index.

## Measurement and correctness

- Reuses `SWBench` and `Stats`, with seven timed trials and three warmups.
- The suite repeats each configuration in three separate processes, rotating
  implementation order. It reports the median of process medians and the range
  of per-process speedups; this range is not a confidence interval.
- Buffers persist across trials. Input loading, normalization, allocations,
  output clearing, warmup, and verification are outside execution timing.
- Every warmup and trial checks every output element against MKL, using
  `abs(error) <= 1e-4 + 1e-4 * abs(reference)` and rejecting nonfinite values.
  Up to 16 complete output rows of the MKL reference are independently checked
  using scalar double-precision arithmetic.
- The inspector is measured separately. Estimated break-even reuse count is
  `ceil(inspection_time / (unfused_time - fused_time))`, or `never` if fusion is
  slower. The graph/sparsity pattern must remain unchanged to reuse its schedule.
- `manifest.json`, all raw CSVs, `summary.csv`, and `RESULTS.md` are retained.
  A failed calculation or failed verification exits nonzero. The report includes
  every declared case, including losses, and the fraction of rows fused.

The inspector budget is an explicit parameter, not an automatic hardware cache
measurement. Comparing several budgets is a tuning experiment; report them all
rather than selecting only the fastest result.
