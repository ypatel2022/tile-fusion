# SpMM-SpMM benchmarks

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
It writes `summary.csv` and latency/speedup figures in PNG and PDF. Speedups pair
process medians; error bars show the observed process range, not confidence
intervals. Above 1x means graph execution is faster.

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

## Structured CPU and GPU baselines

The structured suite uses float32, 32 features and the sizes above, with bands
of 3/5/9 diagonals or dense diagonal blocks of 4/16 rows. GPU methods are ordinary
unfused kernels, cuSPARSE CSR ALG2/ALG3 and tile fusion, each run directly and
with graphs. Both operations use the default stream and wait after the pair.
cuSPARSE preprocesses each operation with its own descriptor/workspace before timing.

The CPU target requires MKL and AVX2 and builds with CUDA disabled. It compares
ordinary parallel SpMM, MKL, unfused AVX2 and fixed-tile fused AVX2 at 1/4/32 threads.
With existing builds, run from the repository root:

```sh
bash fusion/gpu_spmm_spmm_graph_benchmark.sh <gpu-build>/gpu/spmm_spmm_graph_benchmark <gpu-logs> 100 structured
cmake --build <cpu-build> --target spmm_spmm_structured_cpu_benchmark
bash fusion/cpu_spmm_spmm_structured_benchmark.sh <cpu-build>/example/spmm_spmm_structured_cpu_benchmark <cpu-logs>
python3 fusion/scripts/spmm_spmm_graph_plot.py <gpu-logs> --cpu-log-folder <cpu-logs>
```

Use the executables' `--help` for arguments. Runners accept a trial count and
structured row filter. They use one warmup, 100 measured executions and three
processes by default; `run_status.csv` records exits and failures retain CSV/stderr.

Matrices share a deterministic signed dense input and use no permutation or rewiring.
Weights are diagonal 0.5, with band off-diagonals `0.25/half-bandwidth` and block
off-diagonals `0.5/(block-size-1)`. GPU tiles contain four rows; CPU inspectors use
`IterPerPartition=128`, `TileM=-1`, `TileN=32`. `Tile Eligible Ratio` describes
four-row GPU eligibility, separate from method counters and CPU tile schedules.

CPU timing uses the existing host timer. Setup, inspection, preprocessing,
transfers, clearing and checks are outside executor timing on both devices.
The plotter also writes `process_medians.csv`, `failed_results.csv` and, with a
CPU folder, `cpu_gpu_latency.csv` and a latency overview. Comparisons require
three correct paired processes; speedups are baseline time divided by candidate time.
Archived native CSVs remain supported.
