# SpMM-SpMM CUDA Graph benchmark

`spmm_spmm_graph_benchmark` computes `H = A X; Y = A H` using the existing
tile-fusion kernels and inspector. It compares three execution paths:

| Method | Submission | Synchronization |
|---|---|---|
| `direct` | Original two kernel launches | CPU waits after each kernel |
| `direct_same_stream` | Same two kernels on the default stream | CPU waits after the pair |
| `graph` | Existing two-node CUDA Graph | CPU waits after the pair |

The same-stream control isolates the benefit of graph replay from removing the
original implementation's wait between kernels. Stream ordering already preserves
the dependency between the direct launches. The implementation classes are unchanged.

Build the target in an existing CUDA-enabled build, then run:

```sh
cmake --build <build-dir> --target spmm_spmm_graph_benchmark
<build-dir>/gpu/spmm_spmm_graph_benchmark 32 100 0 > trials.csv
```

Arguments are dense feature width (32, 64, 128 or 256; default 32), measured runs
(default 100), and method-order offset (0, 1 or 2; default 0). Each invocation uses
sparse square matrices of size 64, 512, 4,096, 32,768, 262,144 and 1,048,576.
The matrices are tridiagonal with coefficients 0.25, 0.5, 0.25; dense inputs are
deterministic and contain both positive and negative values. Blocks use 128 threads.

The benchmark reuses `SWBench`, `Stats`, the CUDA event timer and the CPU reference.
Each method runs once for warmup, then 100 times by default. Graph construction and
instantiation happen during analysis, before any execution; the first execution
is a warmup replay, not capture. The CSV retains that row as `warmup=1`; exclude it
from timing summaries. Every execution, including warmup, is checked against the
CPU reference and for non-finite output. Failed checks make the program exit nonzero.

`gpu_us` measures the GPU timeline around the pair, including gaps caused by host
submission and synchronization. Allocation, inspection, graph construction, output
copies and correctness checks are outside this interval. `analysis_us` records
preparation separately. `fused_ratio` is the fraction of output rows the inspector
assigns to the first kernel; it is not a measurement of memory traffic.

For repeated measurements, run offsets 0, 1 and 2 in separate processes. Method
order also rotates between matrix sizes. These are repeated evaluations of fixed
matrices and buffers, not end-to-end application timings.

## Results

Measured on an NVIDIA H100 80 GB, CUDA 12.9, driver 580.159.03. Dense width was 32.
Three processes each used one warmup and 100 measured runs per method and matrix,
with order offsets 0, 1 and 2. All 5,454 output checks passed with zero reported error.

Times below are the median of the three process medians, in microseconds per pair.
Speedups are medians of paired process ratios (`direct time / graph time`), so
they need not equal the ratio of the displayed times. Above 1 means graph is faster.
The ranges in the last column are observed process minimum/maximum, not confidence
intervals. A million-row sparse matrix here has about three million nonzeros;
the dense feature matrix is still only 32 columns wide.

| Sparse rows/columns | Nonzeros | Fused rows | Direct (us) | Same-stream direct (us) | Graph (us) | Graph vs direct | Graph vs same stream (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 190 | 53.1% | 17.472 | 12.832 | 14.272 | 1.222x | 0.899x (0.660–1.106) |
| 512 | 1,534 | 50.4% | 17.904 | 13.184 | 11.952 | 1.498x | 1.103x (1.093–1.108) |
| 4,096 | 12,286 | 50.0% | 19.104 | 14.528 | 12.688 | 1.515x | 1.145x (1.142–1.154) |
| 32,768 | 98,302 | 50.0% | 28.448 | 23.584 | 21.824 | 1.308x | 1.085x (1.073–1.089) |
| 262,144 | 786,430 | 50.0% | 162.592 | 156.560 | 155.040 | 1.050x | 1.011x (1.008–1.012) |
| 1,048,576 | 3,145,726 | 50.0% | 577.072 | 570.512 | 566.832 | 1.018x | 1.006x (1.006–1.006) |

The largest gain over the original implementation includes removing its middle
host wait. Against the same-stream control, graph replay improves latency by about
1.10–1.15x at 512–4,096 rows, falling below 1% at the largest size. The 64-row case
varies substantially between processes with this single-warmup procedure, so it
does not establish a reliable graph benefit. These results apply to this ordered
tridiagonal workload; all three methods already use tile fusion. They do not
measure fusion versus unfused execution, DRAM traffic or full application speedup.
