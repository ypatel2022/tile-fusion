# SpMM-SpMM CPU and CUDA Graph benchmarks

The benchmarks compute `H = A X; Y = A H` with float32 inputs. The structured
suite compares ordinary unfused kernels, cuSPARSE CSR ALG2/ALG3 and tile-fused
kernels, each with direct launches and CUDA Graph replay. Direct launches queue
both operations on the default stream and wait after the pair.
cuSPARSE direct and graph variants preprocess both SpMM operations outside timing,
using a separate sparse descriptor and workspace for each operation.

With existing CPU and CUDA-enabled builds, from the repository root:

```sh
cmake --build <gpu-build> --target spmm_spmm_graph_benchmark
bash fusion/gpu_spmm_spmm_graph_benchmark.sh <gpu-build>/gpu/spmm_spmm_graph_benchmark <gpu-logs> 100 structured
cmake --build <cpu-build> --target spmm_spmm_structured_cpu_benchmark
bash fusion/cpu_spmm_spmm_structured_benchmark.sh <cpu-build>/example/spmm_spmm_structured_cpu_benchmark <cpu-logs>
python3 fusion/scripts/spmm_spmm_graph_plot.py <gpu-logs> --cpu-log-folder <cpu-logs>
```

The CPU target requires MKL and AVX2 and can be built with CUDA disabled. It
compares ordinary parallel SpMM, MKL, unfused AVX2 and fixed-tile fused AVX2 at
1, 4 and 32 threads. Existing implementations are used unchanged; incorrect
outputs or crashes are reported without repairing the implementation.

Both structured suites use 32 features and sparse sizes 64, 512, 4,096, 32,768,
262,144 and 1,048,576. Each size has bands of 3/5/9 diagonals and dense diagonal
blocks of 4/16 rows. GPU producer tiles contain four rows; blocks are aligned
with them. CPU tiles use the existing inspector with `IterPerPartition=128`,
`TileM=-1`, `TileN=32`. Methods share each matrix and deterministic signed dense
input. No permutations or ratio-control rewiring are applied. Banded weights
are diagonal 0.5 and off-diagonal `0.25/half-bandwidth`; dense-block weights are
diagonal 0.5 and off-diagonal `0.5/(block-size-1)`.

Each runner uses three processes with rotating method order, one discarded
warmup and 100 measured executions. Optional runner arguments change the trial
count and restrict structured runs to one size. Use a fresh log folder. CSV and
stderr are retained per process; `run_status.csv` records exits. Runners continue
other processes after failures and return nonzero if any process fails.

Executable interfaces:

```sh
spmm_spmm_graph_benchmark --structured [runs=100] [order=0] [rows=all]
spmm_spmm_structured_cpu_benchmark [threads=1] [runs=100] [order=0] [rows=all]
```

Native `Stats` CSV stores trials in columns and times in seconds. Setup,
inspection, allocation, transfers, required clearing and verification are outside
executor timing. GPU CUDA-event times include submission/synchronization gaps;
CPU executor times use the existing host timer. `Tile Eligible Ratio` describes
the matrix under four-row GPU tiles; it is distinct from each method's actual
`Fused Ratio` and the CPU inspector's native schedule counters.

The plotter uses pandas, NumPy and Matplotlib. It writes `summary.csv`,
`process_medians.csv`, `failed_results.csv` and labelled PNG/PDF figures. Failed
or incomplete methods are excluded from plots and speedups; other valid methods
remain available. Speedups are medians of three paired process ratios; plotted
min/max ranges are observed ranges, not confidence intervals. The optional CPU
folder adds a latency overview and `cpu_gpu_latency.csv`; transfers and setup
are excluded, so these are executor comparisons. Record hardware with run logs.

## Original tile-fused suite

The existing executable CLI remains available:

```sh
spmm_spmm_graph_benchmark [features=32] [runs=100] [order=0] [fused-percent]
bash fusion/gpu_spmm_spmm_graph_benchmark.sh <executable> <log-folder> [runs=100] [suite=ratios|tridiagonal|structured] [rows=all]
```

The default `ratios` suite runs 0/25/50/75/100% fused rows. `tridiagonal` preserves
the original input. All three original methods use tile fusion: `direct` waits
after each kernel, `direct_same_stream` waits after the pair and `graph` replays
the pair. An explicit fused percentage requires 32 features and rewires
tridiagonal columns while preserving weights and nonzero counts; locality also
changes. Other feature widths are 64, 128 and 256. At 100% fusion the second
kernel remains a no-op. Archived native CSVs remain supported by the plotter.
