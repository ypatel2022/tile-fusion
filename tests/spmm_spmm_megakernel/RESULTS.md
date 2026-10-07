# One-launch SpMM–SpMM results

The cooperative phase-barrier control wins more often than the completion-event
methods in this sweep. Avoiding unused H stores reduces traffic in all-local
tiles, but the byte-mask version does not improve the examples below.

All timed methods run on the GPU. The CPU supplies an independent double oracle.
Measured on H100 80 GB HBM3: 132 SMs, 50 MiB L2, driver 580.159.03, CUDA 12.9.1
(nvcc 12.9.86), cuSPARSE 12.5.10.65, GCC 12.3. Profilers: Nsight Systems 2025.1.3
and Nsight Compute 2025.2.1. Release, sm_90, FP32, FMA enabled, 32 features.
Benchmark runs use no explicit GPU clock lock. Nsight Compute uses
`cache-control none` and `clock-control none`.

| Revision | Commit |
|---|---|
| Frozen evaluation | `9a3da64a78f547d28675c2eddb5b9a5bd43be921` |
| Initial kernels, full global H stores | `4156ba9429347393101680123f0965226bc4c96e` |
| Shared kernel with selective H stores | `b10f0f15d58282ab39afc5457d4232d3c604a2eb` |

Each version has 540 processes, 97,200 raw measurement rows and 6,480 complete
method/condition/phase summaries, with no performance failures. All six evaluation
files retain their frozen hashes. Maximum checked absolute error is 3.0113e-7
against the fixed 1e-6 criterion. Fresh allocations, NaN sentinels, same-address
X/A-value changes and memcheck/racecheck/synccheck passed for the custom methods.
Baseline validation retains 72 cuSPARSE constructor rejections for tiny duplicate
fixtures, documented in [EVALUATION.md](EVALUATION.md). Partial tiles passed
correctness, sanitizer and launch checks; their latency was not measured.

## Execution

Seven sizes, bands 3/5/9, blocks 4/16, and row tiles 1/4/16 use three processes
and seven batches per method. Repeated execution uses one buffer set. Rotation
uses distinct X/H/Y sets with X alone exceeding twice actual L2. All twelve
methods use the same inputs for a condition. The full rules are in
[EVALUATION.md](EVALUATION.md).

The `graph` baseline captures the original two-kernel pair in this evaluator.
The production `FusedSpMMSpMMSeqReduceRowBalanceGraph` wrapper remains available
unchanged on this branch.

Times below are CUDA-event submission-through-completion measurements, including
host gaps. Synchronized wall measurements bracket the same required work and
were independently cross-checked. Latencies are medians of process medians;
speedups pair process orders. Ranges are observed min/max, not confidence intervals.
The vendor comparison selects the faster ALG2/ALG3 graph per process: tuning.

Final-version conditions faster than each reference; parentheses count conditions
faster in all three processes:

| One-launch method | Fused graph | Fused direct, no middle wait | Unfused graph | Tuned vendor graph |
|---|---:|---:|---:|---:|
| Cooperative barrier | 136/180 (134) | 138/180 (133) | 128/180 (123) | 106/180 (106) |
| Events, global H | 32/180 (32) | 22/180 (22) | 47/180 (47) | 47/180 (47) |
| Events, shared H + mask | 23/180 (23) | 16/180 (16) | 44/180 (44) | 40/180 (40) |

Representative final latencies in microseconds; every row uses rotating inputs.

| Rows / pattern / tile | Fused direct | Fused graph | Unfused graph | Barrier | Global events | Shared + mask |
|---|---:|---:|---:|---:|---:|---:|
| 4,096 / band-3 / 4 | 7.933 | 44.154 | 41.742 | 7.574 | 11.303 | 11.954 |
| 1,048,576 / band-3 / 4 | 556.916 | 554.837 | 508.774 | 387.913 | 1,266.021 | 1,302.951 |
| 1,048,576 / block-4 / 4 | 367.969 | 366.068 | 459.357 | 404.446 | 403.436 | 434.237 |
| 1,048,576 / block-16 / 16 | 746.026 | 745.076 | 896.485 | 863.243 | 749.572 | 852.779 |
| 1,048,576 / block-16 / 1 | 1,617.473 | 1,614.648 | 1,450.719 | 1,394.740 | 10,095.537 | 10,305.499 |

At 1,048,576 band-3 rows/tile 4, the barrier is 1.430x faster than the fused graph
(process range 1.430–1.432x). Fine events lose despite fewer DRAM reads. At 4,096
rotating rows, graph replay is unusually slow; large graph-relative gains should
be read alongside the direct baseline. Matched Nsight tracing measured longer
submission/dispatch gaps, but reduced the graph result from about 44 to 14 us.
The remaining discrepancy is unresolved; profiling changes execution behavior.

## H-store change

The initial shared variant still wrote every H row globally. The mask marks every
H row referenced by a deferred Y row, including home-tile references, because
another producer can run its callback. All H values are recomputed; unused global
stores are omitted. Full H allocation remains. Mask bytes count in schedule
storage; logical H-store bytes are separate from measured DRAM traffic.

Separate before/after runs, shared method only. Counters cover one profiled
invocation each; no counter variability estimate is available.

| Matrix / tile / regime | Initial us | Masked us | Initial DRAM write MiB | Masked DRAM write MiB |
|---|---:|---:|---:|---:|
| 131,072 block-16 / 16 / repeated | 112.239 | 112.560 | 32.09 | 17.06 |
| 1,048,576 block-4 / 4 / rotating | 413.076 | 434.237 | 255.32 | 129.15 |
| 1,048,576 block-16 / 16 / rotating | 848.558 | 852.779 | 256.63 | 129.48 |

Million-row all-local writes roughly halve; runtime does not improve here. The
mask test adds work, while row arithmetic and scheduling remain. Their separate
costs were not isolated. Band-3/tile 4 omits only four H rows (512 bytes) and adds
an N-byte mask, giving negligible large-case store savings.

CSR + X/H/Y for band-3 uses about 1.61, 51.5 and 412 MiB at 4,096, 131,072 and
1,048,576 rows. Schedules/events/library workspace add to these estimates. The
all-local 131,072/block-16/tile-16 case has a logical active footprint near
49.25 MiB after omitting global H stores; nevertheless measured reads remain
about 33.17 MiB in both repeated and rotating profiles. Size alone does not prove
cache residence. Graph/runtime internal allocation sizes are unavailable.

## Scheduling and resources

All custom captures contain one kernel and zero copies/resets; epochs avoid a
per-call reset. Existing fused/unfused pairs contain two kernels and cuSPARSE
ALG2 graphs contain six. [DESIGN.md](DESIGN.md) gives the visibility/progress proof.

Initial million-row band-3/tile-4 rotating profiles read about 312 MiB with the
barrier versus 169 MiB with events. Explicit shared staging adds little reduction
over global events. Event methods generate 120–134 million L2 sectors versus
33 million for the barrier. These counters include scheduling/atomic traffic;
they do not isolate H or establish a sole bottleneck.

At block-16/tile-1, both designs have about 49.7% achieved occupancy (50% theoretical).
The events grid has 1,048,576 blocks and 16,777,216 producer notifications
(source-derived); the barrier has 4,224 resident blocks. Events reduce reads
from about 520 to 352 MiB but produce roughly 15x as many L2 sectors and run
about 7x slower than the barrier. Finer scheduling carries substantial work.

Candidates use 32 registers/thread, with zero compiled stack frame/spill bytes
reported by ptxas. Shared H needs 128/512/2,048 bytes for tiles 1/4/16. Tile 4/16
profiles have 100% theoretical occupancy. Driver-reserved shared storage and
rounding are additional. Configured 1,024-byte stack capacity is separate from
compiled use; local-memory traffic counters were not collected. Inconsistent
replay counters produce L2 hit ratios above 100% in one small profile per version;
valid percentages remain unset and raw counts are retained.

## Setup and transfers

Resident end-to-end includes X upload, computation and Y readback. Cold time
means a fresh plan through first output in an initialized CUDA process; input
creation, the oracle and process/context startup are excluded.

Final rotating band-3, 1,048,576 rows/tile 4; wall milliseconds:

| Method | Resident call | Cold first output | Setup |
|---|---:|---:|---:|
| Fused graph | 15.904 | 373.127 | 364.920 |
| Unfused graph | 15.583 | 27.210 | 18.529 |
| Barrier | 15.628 | 25.946 | 17.646 |
| Shared + mask | 16.711 | 68.173 | 59.008 |

Transfers narrow execution gains. The fused graph's setup here includes about
347 ms of original inspection and 0.40 ms of graph preparation. Captured graphs
use one executable per buffer slot; allocation, inspection and capture/instantiate/
upload are reported separately. These are the benchmark adapter's preparation
costs. Faster first-plan results also reflect a different inspector, not just GPU
execution. Full component timings and extra-setup break-even estimates are in
`summary.csv`; estimates exclude transfers and other application work.

Plans can be reused for value changes at fixed addresses. Shape, CSR topology,
tile size or pointer changes require new preparation; epochs must not wrap.
The full sweep retains slowdowns and all methods. Raw measurements, native Stats,
source/binary hashes, commands, sanitizer and profiler archives remain outside Git.
[README.md](README.md) gives portable reproduction commands.

Untested next steps: use a schedule-level no-store policy for all-local tiles to
avoid mask reads, or a shared-only load path for local Y. This sweep establishes
neither improvement. Scope: synthetic square CSR SpMM pairs with 32 FP32 features.
