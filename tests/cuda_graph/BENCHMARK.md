# Performance test for the 30-GEMM DAG

`gemm_dag_benchmark.cu` follows the repository's GPU benchmark conventions:
it derives from `swiftware::benchmark::SWBench`, uses `Stats` and `Timer` for
per-trial CSV output, defaults to seven trials, and verifies results outside
timing. It shares the kernel, inputs, graph, and CPU reference in `gemm_dag.cuh`
with the correctness example.

## What is compared

| CSV implementation name | Execution |
| --- | --- |
| `GPU_Sequential_Launches` | 30 ordinary kernel launches in topological order on one stream. |
| `GPU_Serial_Graph` | Capture that same sequence once, then replay it as a graph. |
| `GPU_DAG_Graph` | Replay the original graph with its 48 explicit dependencies. |

The serial graph comparison keeps kernel ordering the same and shows the benefit
of graph replay. Comparing the DAG graph with the serial graph shows the effect
of permitting independent GEMMs to overlap. The GPU may still serialize branches;
a speedup is not guaranteed. All three modes compute the same 30 matrix products.
This benchmarks the teaching kernel; it is not a comparison with optimized GEMM.

## Run on Fir

From the repository root:

```bash
sbatch tests/cuda_graph/perf.sbatch
```

The batch script requests one H100, one CPU, 2 GB RAM, and ten minutes, then builds
and runs the benchmark. Output files are:

- `tests/cuda_graph/perf-<job-id>.out`: build log, device/version information,
  correctness failures, and a short timing/speedup summary.
- `tests/cuda_graph/perf-<job-id>.csv`: one header and three implementation rows,
  with individual trial times, correctness flags, and median results.

Optional positional arguments are **matrix size, replays per trial, number of
trials, warmup replays per trial**, in that order:

```bash
sbatch tests/cuda_graph/perf.sbatch 64 100 7 10
sbatch tests/cuda_graph/perf.sbatch 256 100 7 10
```

Defaults are `64 100 7 10`. Sizes from 1 to 1024 are accepted, including sizes
not divisible by 16. All counts must be positive. CPU reference work scales as
30 times the cube of the matrix size, so larger sizes can need more job time.

For an existing GPU allocation, build from the repository root:

```bash
module load StdEnv/2023 cuda/12.6 cmake/3.27.7
cmake -S tests/cuda_graph -B tests/cuda_graph/build -DCMAKE_BUILD_TYPE=Release
cmake --build tests/cuda_graph/build --parallel 1
srun tests/cuda_graph/build/gemm_dag_benchmark 64 100 7 10 > tests/cuda_graph/perf-manual.csv
```

The default architecture is `sm_90`; CMake accepts
`-DCMAKE_CUDA_ARCHITECTURES=...` for a different GPU. No sparse matrices, MKL,
METIS, PAPI, or external downloads are required for this standalone build.

## How timing works

1. Prepare fixed inputs and compute the CPU reference once. Initialize the CUDA
   kernel before collecting timings.
2. For graph modes, measure graph construction/capture, instantiation, and upload
   as a separate, one-time preparation cost.
3. Before each trial, fill output buffers with NaNs and run the warmup replays.
   Finish that work before starting the measurement.
4. Record CUDA events around a batch of complete DAG evaluations on the same
   stream. Also use a host `steady_clock` around submission and completion.
   Synchronize once at the end of the batch, not between GEMMs or replays.
5. Divide both elapsed times by the number of replays. Copy back and verify all
   30 outputs against the CPU reference after each trial. A mismatch exits nonzero.

Each replay overwrites all outputs using the same A and B. Replays use one stream,
so output buffers are not reused concurrently across replays. Buffers stay
allocated for the whole benchmark. Allocation, CPU/GPU copies, validation,
printing, warmup, and graph preparation are excluded from replay times.

CUDA-event time is elapsed time on the GPU timeline, including gaps while the
CPU submits work; it is not the sum of isolated kernel runtimes. Host wall time
includes submission, completion waiting, and the event calls, amortized over the
batch. These are steady-state batch averages, not single-request latency. Inputs
and buffers are reused, so caches are warm. Modes run in the table's order; repeat
jobs to assess noise from clocks or other activity before drawing conclusions.

## Reading the CSV

All CSV times are **seconds per complete 30-GEMM evaluation**, except the graph
preparation column, which is total one-time seconds. The log displays replay
times in microseconds and preparation in milliseconds.

The repo's `Stats` format names executor regions by position:

- `TrialX Subregion0 Executor`: CUDA-event seconds per DAG for trial X.
- `TrialX Subregion1 Executor`: host wall seconds per DAG for trial X.
- `CorrectX` and `ErrorX`: CPU comparison status and maximum absolute error.
- `Trial0 Subregion0 Graph preparation seconds`: one-time graph cost; zero for
  ordinary launches.
- `Median GPU Seconds Per DAG0` and `Median Wall Seconds Per DAG0`: median of
  the trial batch averages.
- `GPU Speedup vs Sequential0` and `Wall Speedup vs Sequential0`: corresponding
  ordinary-launch median divided by the implementation's median. Above 1 is faster.

The trailing `0` on summary fields and trailing empty CSV column come from the
existing `Stats` serializer. GPU name, CUDA runtime/driver versions, matrix size,
GEMM count, and replay/warmup counts are included for reproducibility.

For API details, see NVIDIA's [CUDA event timing reference](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)
and [CUDA Graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
