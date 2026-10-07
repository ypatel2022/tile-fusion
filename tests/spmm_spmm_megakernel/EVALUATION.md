# Fixed SpMM–SpMM evaluation

Validate the baseline-only build and commit this evaluation before writing a
megakernel. Record that commit and SHA-256 hashes of these evaluation files in
every run. Later changes belong to the implementation. A genuine evaluation
error invalidates affected results and must be reported before its correction.

## Computation and correctness

Compute `H = A X; Y = A H` for square zero-based CSR A and row-major X with
32 features. Y is the required output; H is intermediate workspace. Store and
compute in float32, with FMA permitted. No fast math, TF32, lower precision,
precomputed answers, A-squared reassociation or omitted terms. Structure-specific
schedules may depend on CSR indices, never input values, case names or seeds.

The independent CPU oracle copies the float32 CSR values and X, uses separate
double-precision H/Y arrays and scalar CSR loops for both products. Every checked
Y element must be finite and within `1e-6` absolute error. Check materialized H
as well; implementations may keep intermediate H private or avoid global stores
when no consumer needs them. This does not weaken the Y criterion.

Correctness fixtures use 1, 3, 17, 65, 127 and 257 rows; bands of 3/5/9 diagonals;
blocks of 4/16 rows with a truncated final block; and identity, zero-nonzero,
empty-row and unsorted/duplicate-index cases. Test tile sizes 1/4/16. At least
three deterministic signed input variants test arbitrary valid values. Test fresh
allocations, replaced X values at the same addresses, and replaced sparse values
at the same addresses without rebuilding structural schedules. Signed sparse
replacements have bounded absolute row sums. GPU functions receive CSR and values,
not fixture identifiers. Poison outputs with NaNs before each checked invocation;
keep oracle buffers separate. Correctness and timing call the same launch path.
Run memcheck, racecheck and synccheck on the same families and tiles at 1/17/65
rows (`--sanitizer-cases`), with all three variants and mutation stages. Retain failures and errors;
a failed method does not become a valid result because a previous output matched.

## Methods

Keep existing kernels and inspectors at parent commit
`2bc71e7ca5768344df4c88eb8e4b563c2e217b4f` unchanged. This parent contains CUDA
Graph implementation PR #2 and benchmark PR #3; ancestry and the graph wrapper
were verified against the fork before the branch was created.

Compare the original waited tile-fused pair, tile-fused same-stream pair and
its graph; ordinary unfused direct/graph pairs; and cuSPARSE ALG2/ALG3
direct/graph pairs. Preserve the benchmark's guarded no-op deferred launch when
all rows are eligible. Library preprocessing and graph preparation are setup.

Reserve three custom methods before implementation: `megakernel_barrier`,
`megakernel_events_global` and `megakernel_events_shared`. The first is a
cooperative one-launch global-phase control. The other two use completion events;
the latter additionally reuses producer H in shared memory. Use the same FP32 row
arithmetic. A candidate must perform the full computation with one kernel launch,
including any required reset/helper work. Verify actual launches with a trace;
one graph API call does not imply one GPU kernel launch.

## Workloads and cache conditions

Benchmark 64, 512, 4,096, 32,768, 131,072, 262,144 and 1,048,576 rows, the five
structures above and GPU tile sizes 1/4/16. Benchmark A values retain the existing
positive structured weights: diagonal 0.5, band off-diagonals
`0.25 / half-bandwidth`, block off-diagonals `0.5 / (block-size - 1)`. No rewiring
or permutation. Dense signed inputs vary deterministically by process and buffer.
Use the same current input for every method in a condition. The input generator
uses a fixed SplitMix64 hash, maps it to signed multiples of 1/1024, and selects
the variant as `process_order * 1000003 + buffer_index`. Its code is frozen.

Baseline validation with CUDA 12.9 found cuSPARSE rejects the 1/3-row duplicate fixtures:
they store 2/10 entries, exceeding 1/9 matrix positions. Retain these explicit
library failures and the unchanged fixtures. All custom methods must still pass
them. Performance matrices have unique indices and stay within that limit.

Repeated execution reuses one X/H/Y buffer set. Rotating execution is defined for
sizes at least 4,096, with distinct dense buffers and shared sparse structure.
Its X pool alone must exceed twice the device-reported L2 capacity:

```text
slots = max(2, floor(2 * L2_bytes / X_bytes) + 1)
```

Do not cap this count. Report allocated CSR, X/H/Y, schedules, events and library
workspace bytes separately, along with per-call and whole-pool footprints. Around
131,072 rows CSR+X/H/Y is near or above H100's 50 MiB L2; smaller and larger sizes
cover both sides. Allocation size alone does not establish cache residence.

## Measurement

Use three independent processes with rotating method order. Warm the entire
buffer ring once. Each of seven measured batches executes `max(100, 2*slots)`
complete pairs in round-robin buffer order. Check every ring output after each
batch outside timing. Correctness checks each invocation independently; source
review and launch tracing establish that repeated calls always execute the work.

Steady-state CUDA events and a synchronized steady clock bracket identical
submission-through-completion work. Synchronize the whole device at boundaries;
include required resets, helper kernels and work on other streams. Exclude oracle,
verification, readback, setup and initial copies from both compute timers. Do not
modify the repository Timer. Store high-precision batch seconds per invocation
alongside native Stats CSV, including batch size, slot count and precise errors.
Wait for the start event before submission. Record the stop event after whole-device synchronization so the event interval
includes completion of all streams. Both measurements include the start-event
wait and host wakeup/stop-event submission gap; they describe submission/completion time. Use
profiler kernel durations separately when examining GPU compute alone.

Report seven resident end-to-end measurements including X upload, computation,
required reset and Y download. Also report reusable setup components (allocation,
inspection, schedule uploads, library preprocessing and per-slot graphs) and cold
end-to-end time from pre-generated host CSR/X through allocation, copies,
preparation, first execution and Y readback. Input generation and oracle creation
are excluded and identified. State when changes of structure, shape, pointers or
epoch capacity require setup again; show amortization without hiding setup costs.
Allocation time includes initial CSR/X uploads; separate columns measure device
malloc and initial copies. Preparation combines inspection/schedule upload,
library preparation (including priming), and graph capture/instantiate/upload for
the whole ring, with component columns. These are reusable costs, excluded from
steady timings. Report allocated H and globally written H separately. Graph/runtime internal
device allocation sizes are unavailable from these APIs and remain unreported.

Summarize each process with its median; aggregate latencies with median process
medians. Speedups are medians of paired process ratios. Show observed min/max,
not confidence intervals. Preserve all slowdowns, errors and incomplete methods.
Do not rank a method with fewer than three correct processes.

## Profiling and reporting

Keep profiler timings separate. Use Nsight Systems with graph node tracing to
count compute/reset/helper kernels and memory operations inside an isolated
invocation. Use Nsight Compute on matched repeated/rotating cases to measure
DRAM reads/writes, weighted L2 hits, registers, shared memory and occupancy.
Investigate unusually large gains using these counters and event/wall agreement;
whole-pair traffic is not H-only traffic, and counters do not prove a sole cause.

Retain executable/source hashes, evaluation commit, implementation commits, raw
trials, correctness and sanitizer logs, profiler reports, hardware/software,
commands and full summaries. Keep Fir-specific scripts outside the repository.

Sources: [Event Tensor](https://arxiv.org/pdf/2604.13327),
[Wafer field guide](https://www.wafer.ai/blog/reward-hacks-field-guide),
[CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html),
[CUDA execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html).
