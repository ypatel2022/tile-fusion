# SpMM-SpMM CUDA Graph benchmark

`spmm_spmm_graph_benchmark` computes `H = A X; Y = A H` with the existing
tile-fusion kernels and compares three methods:

| Method | Execution |
|---|---|
| `direct` | Two kernel launches, with a CPU wait after each |
| `direct_same_stream` | Two kernel launches, with a CPU wait after the pair |
| `graph` | A two-node CUDA Graph, with a CPU wait after the pair |

The same-stream method separates graph replay's benefit from removing the wait
between kernels.

From the repository root, with an existing CUDA-enabled build:

```sh
cmake --build <build-dir> --target spmm_spmm_graph_benchmark
bash fusion/gpu_spmm_spmm_graph_benchmark.sh <build-dir>/gpu/spmm_spmm_graph_benchmark <log-folder>
python3 fusion/scripts/spmm_spmm_graph_plot.py <log-folder>
```

The shell script runs 32 features, 100 measured trials and three processes at
each of 0%, 25%, 50%, 75% and 100% fused rows. Method and ratio order rotate.
Its optional third argument changes the trial count; a fourth argument of
`tridiagonal` selects the original input family. Use a fresh log folder per sweep.

The plotter uses pandas, NumPy and Matplotlib, as the other GPU plotters do.
It writes `summary.csv` and latency/speedup figures in PNG and PDF, and prints
the summary. Speedups pair process medians; error bars show the observed process
range, not confidence intervals. Above 1x means graph execution is faster.

To run the executable directly:

```sh
<build-dir>/gpu/spmm_spmm_graph_benchmark [features=32] [runs=100] [order=0] [fused-percent]
```

Feature widths are 32, 64, 128 or 256; process order is 0, 1 or 2. An explicit
fused percentage requires 32 features (four rows per tile). It rewires tridiagonal
entries while preserving each row's nonzero count and weights; this also changes
memory locality. Omitting the percentage preserves the original tridiagonal input.
All sweeps use 64, 512, 4,096, 32,768, 262,144 and 1,048,576 rows. At 100% fusion,
all methods retain a second kernel launch that does no work.

CSV uses the existing `Stats` format: one row per method and matrix, with trials
in columns and times in seconds. Trial 0 is warmup and the plotter excludes it.
Graph construction happens before warmup. CUDA-event timings include host
submission/synchronization gaps and exclude setup, output copies and checks.
`Fused Ratio` records the inspector's actual fused-row fraction. Every trial is
checked against the CPU reference and for non-finite output; failures exit nonzero.
