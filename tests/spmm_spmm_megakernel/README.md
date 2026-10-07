# One-launch SpMM pairs

Compute `H = A X; Y = A H` in one CUDA kernel. The experiment compares a
cooperative grid barrier, completion events with global H, and completion events
with a shared H tile. Existing tile-fused, unfused and cuSPARSE pairs remain
baselines. All methods use float32 and 32 features.

[DESIGN.md](DESIGN.md) explains dependencies and visibility.
[EVALUATION.md](EVALUATION.md) fixes the oracle, inputs, sweep and timing.
[RESULTS.md](RESULTS.md) records the measured gains and losses.

## Build and run

Initialize the repository submodules and provide its existing MKL/METIS build
dependencies, CUDA and CMake. From the repository root:

```sh
git submodule update --init --recursive
cmake -S tests/spmm_spmm_megakernel -B build/spmm-megakernel \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DSPMM_MEGAKERNEL_SOURCE="$PWD/tests/spmm_spmm_megakernel/megakernel.cu"
cmake --build build/spmm-megakernel --target spmm_spmm_megakernel_benchmark
```

Run GPU commands on a GPU-equipped node. Correctness includes fresh allocations,
changed values at the same addresses and poisoned outputs:

```sh
binary=build/spmm-megakernel/spmm_spmm_megakernel_benchmark
for method in megakernel_barrier megakernel_events_global megakernel_events_shared; do
  "$binary" --correctness "$method"
  for tool in memcheck racecheck synccheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 1 \
      "$binary" --correctness "$method" --sanitizer-cases
  done
done
python3 tests/spmm_spmm_megakernel/run_suite.py "$binary" results/megakernel
python3 tests/spmm_spmm_megakernel/report.py results/megakernel --output results/summary
```

The runner records three processes per condition and retains errors. Its output
directory must be new. `summary.csv` contains every method and all three timing
phases; process ranges are observed minima/maxima. Generated data stay local.
Use `--help` for individual cases and filters. Omitting `SPMM_MEGAKERNEL_SOURCE`
builds the baseline-only evaluation.

## Profile a pair

`--trace` warms the selected buffer ring, captures one invocation through the
CUDA profiler API, and checks its output. Use graph node tracing to count kernels:

```sh
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
  "$binary" --trace 1048576 banded 3 16 megakernel_events_shared rotating
```

Profiler durations are separate from the sweep's submission/completion timings.
Cache counters sum every kernel in the invocation and weight L2 hits by sector
counts. The current profiler capture covers one pair; it does not measure
whole-application performance.

For matching memory/resource counters with Nsight Compute:

```sh
ncu --config-file off --replay-mode application --app-replay-mode strict \
  --profile-from-start off --cache-control none --clock-control none \
  --section LaunchStats --section Occupancy \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum,lts__t_sectors_lookup_hit.sum \
  --export pair \
  "$binary" --trace 1048576 banded 3 16 megakernel_events_shared rotating
ncu --import pair.ncu-rep --csv --page raw --print-units base > pair-counters.csv
```

Retain raw counters when replay passes disagree; invalid hit percentages remain
missing. Library pairs may launch more than two kernels.
