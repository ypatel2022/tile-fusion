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

From an existing CUDA-enabled build:

```sh
cmake --build <build-dir> --target spmm_spmm_graph_benchmark
<build-dir>/gpu/spmm_spmm_graph_benchmark 32 100 0 > trials.csv
```

Arguments are feature width (32, 64, 128 or 256), measured runs, and method-order
offset (0, 1 or 2). Defaults are `32 100 0`. Use different offsets to rotate the
method order across repeated runs.

The sweep uses tridiagonal sparse matrices with 64, 512, 4,096, 32,768, 262,144
and 1,048,576 rows. Each method runs one warmup before the measured runs; exclude
CSV rows marked `warmup=1`. Graph construction happens before warmup.

An optional fourth argument requests 0, 25, 50, 75 or 100 percent fused rows:

```sh
<build-dir>/gpu/spmm_spmm_graph_benchmark 32 100 0 75 > trials-75.csv
```

This mode requires 32 features (four rows per tile). It rewires tridiagonal
entries within or across tiles, preserving each row's nonzero count and weights.
The inspector's fused row count must match the requested percentage. At 100%,
all methods retain a second kernel launch that does no work. Omitting the
argument preserves the original matrices; `target_fused_percent` is then -1.

`gpu_us` uses CUDA events and includes gaps from host submission and synchronization.
It excludes setup, output copies and correctness checks. `analysis_us` records
preparation time; `fused_ratio` is the fraction of output rows computed by the first kernel.
Every run is checked against the CPU reference and for non-finite output.
Verification failures make the benchmark exit nonzero.
