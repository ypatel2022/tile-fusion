# CUDA tile-fusion replay benchmark

Compares `SpMMSpMMGraph`'s four execution modes on the same matrices, device
buffers, and kernels. Each evaluation computes `H = B X`, then `Y = A H`.

| Comparison | What it measures |
|---|---|
| `unfused / unfused_graph` | Graph replay for the two unfused kernels |
| `fused / fused_graph` | Graph replay for the fused schedule |
| `unfused / fused` | Fusion with ordinary launches |
| `unfused / fused_graph` | Fusion and graph replay together |

Ratios above one indicate a speedup. These baselines use this implementation's
kernels; they are not cuSPARSE comparisons.

## Build and run

Set the CUDA architecture for the target GPU:

```bash
cmake -S tests/cuda_tile_fusion -B build/cuda-tile-fusion-benchmark \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/cuda-tile-fusion-benchmark --target spmm_spmm_graph_benchmark -j2
build/cuda-tile-fusion-benchmark/spmm_spmm_graph_benchmark 4096 64 50 > trials.csv
```

```text
spmm_spmm_graph_benchmark rows width locality_percent [replays=200] [trials=7] [warmup=20] [order_offset=0]
```

`rows` must be a multiple of 32 in [64, 262144], `width` is in [1, 256],
and `locality_percent` is 0, 50, or 100. `order_offset` is in [0, 3].
CSV goes to stdout; device information and errors go to stderr.

## Workload

A and B are square float32 CSR matrices with eight distinct entries per row.
B has a cyclic banded pattern. Tile-local A rows reference one 32-row producer
tile; other A rows reference two adjacent tiles. The 50% case alternates local
and nonlocal rows within each tile. Sparse values have mixed signs and an
absolute row sum of one; dense inputs are deterministic and signed.

The percentage controls tile eligibility, not general graph randomness. These
structured matrices are intended to expose size and scheduling effects. They
do not represent full GNN inference or training. The fused kernel still writes
H to global memory; local consumers reuse its shared-memory copy.

## Timing and correctness

The benchmark reuses `SWBench` and `Stats`. An outer loop interleaves the four
modes across seven trials, rotating their order each trial. Each CSV row is one
mode/trial (`Number of Trials` is therefore one); `trial0` records its round.
Use different order offsets for independent process repetitions.

- CUDA events on a nonblocking stream bracket a batch of complete evaluations.
  GPU timeline time can include gaps caused by host submission.
- A steady clock records host submission through completion, plus submission
  time alone. Submission may itself block when the runtime queue fills.
- Inputs stay resident and unchanged across replays. No synchronization occurs
  between evaluations. These are steady-state repeated-input measurements.
- Warmup, output poisoning, allocation, copies, CPU references, and verification
  are outside replay timing. Every timed trial checks the full output against
  two calls to the existing double-precision sequential SpMM routine.
- The inspector's fused/deferred row counts must match the requested percentage.
  Any mismatch or incorrect output exits nonzero.

The shared plan's construction time includes sparse copies, inspection,
workspace allocation, and both graph instantiations. It is recorded separately
and repeated in each CSV row, not paid again per trial. It excludes first-launch
upload, which is absorbed by warmup. This is not a per-mode graph setup cost.

Key CSV fields (all times are seconds per complete evaluation except setup):

| Field | Meaning |
|---|---|
| `Implementation Name` | Execution mode |
| `Trial0 Subregion0 Executor` | GPU timeline seconds per evaluation |
| `wall_seconds0`, `submit_seconds0` | Host timing per evaluation |
| `Trial0 Subregion0 PlanSetup` | One-time shared plan construction seconds |
| `rows0`, `width0`, `locality_percent0` | Workload parameters |
| `trial0`, `order0`, `order_offset0` | Trial round and ordering |
| `replays0`, `warmup0`, `trials0` | Repetition counts |
| `fused_rows0`, `deferred_rows0` | Inspected partition sizes |
| `Correct0`, `max_abs_error0` | Verification result and maximum absolute error |

For size sweeps, retain all trials and use medians within each independent
process, then summarize paired mode ratios across processes. Report variation
and slowdowns alongside speedups. Allocation and setup costs matter separately
for applications that rebuild plans frequently.
