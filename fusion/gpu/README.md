# CUDA tile fusion with graph replay

`cuda_tile_fusion` executes `H = B X`, then `Y = A H`, where A and B are CSR
sparse matrices and X/Y are row-major dense float32 matrices. The reusable
`SpMMSpMMGraph` owns the sparse device storage, intermediate matrix, schedule,
and instantiated CUDA Graphs. Its public API is in
[`Cuda_SpMM_SpMM.h`](../include/sparse-fusion/Cuda_SpMM_SpMM.h).

## Execution

The host inspector partitions B's rows into fixed contiguous tiles. An A row
can execute locally when all its column indices identify rows of H in one
producer tile. Other A rows are deferred. Empty A rows produce zeros.

```text
fused producer/consumer kernel ──► deferred-consumer SpMM kernel
       H = B X + local Y                   remaining Y = A H
```

Each thread block owns one producer-row tile and 32 dense columns. It computes
H into shared memory, also writes H to global memory for deferred consumers,
synchronizes its threads, and computes its local Y rows from shared memory.
Every H and Y element has one writer. There are no inter-block spin waits or
floating-point atomics.

The graph edge ensures the deferred kernel sees completed H values. When every
consumer is local, the graph contains only the fused kernel. CUDA Graphs reuse
the launch/dependency setup; the fused kernel provides the shared-memory reuse.
Shared memory does not survive between graph nodes.

Four modes expose the same workload for controlled comparisons:

| Mode | Execution |
|---|---|
| `Unfused` | Two ordinary SpMM launches on the supplied stream |
| `UnfusedGraph` | Those two SpMMs connected by a CUDA Graph dependency |
| `Fused` | Fused kernel and, when needed, deferred kernel on the stream |
| `FusedGraph` | The fused schedule as an instantiated CUDA Graph; default |

## API

```cpp
#include "sparse-fusion/Cuda_SpMM_SpMM.h"

// Host CSR views; A.columns == B.rows. X/Y are separate device buffers.
swiftware::cuda::CsrView a{m, k, a_nnz, a_offsets, a_indices, a_values};
swiftware::cuda::CsrView b{k, n, b_nnz, b_offsets, b_indices, b_values};
swiftware::cuda::SpMMSpMMGraph plan(a, b, feature_columns, d_x, d_y);

plan.launch(stream);
// Update d_x on the same stream, then replay the plan with the new values.
plan.launch(stream);
// Check cudaStreamSynchronize(stream) before reading Y or destroying the plan.
```

Construction validates CSR indices, copies both sparse matrices, inspects their
dependencies, allocates workspace, and instantiates the graphs. These are setup
costs. Launch is asynchronous and performs no allocation or host synchronization.

A/B storage may be released after construction. Changing their structure or
values requires a new plan. X and Y have fixed device addresses, must not overlap,
and must remain alive until all submitted work completes. Dense values may
change between ordered launches. The plan overwrites every output element;
callers do not need to clear Y between replays.

Use one stream, or events that serialize uses of a plan's shared workspace.
Concurrent uses of one plan are unsupported. Create, launch, and destroy it on
the same CUDA device, and synchronize before destruction. Invalid arguments
throw `std::invalid_argument`; CUDA API failures throw `std::runtime_error`.
Asynchronous execution errors are reported by the caller's synchronization.

Initial scope: positive matrix dimensions, 32-bit CSR indices, float32 values,
and row tiles of 1–128 (default 32). Rectangular matrices, empty sparse rows,
unsorted/duplicate column indices, and partial feature tiles are supported.
This is the first SpMM–SpMM implementation layer; GEMM–SpMM and scheduling an
arbitrary operator DAG are separate extensions. Performance has not yet been
characterized against cuSPARSE or across representative workloads.

## Build and validate

Build independently of the CPU demos' MKL/METIS dependencies, setting the CUDA
architecture for the target GPU:

```bash
cmake -S fusion/gpu -B build/cuda-tile-fusion \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/cuda-tile-fusion -j2
ctest --test-dir build/cuda-tile-fusion --output-on-failure
```

It is also available through the main `fusion` build with `ENABLE_CUDA=ON`.
Link `swiftware::cuda_tile_fusion` from CMake. The standalone build excludes
legacy GPU demos; the main build retains them by default. Set
`TILE_FUSION_BUILD_LEGACY_CUDA_DEMOS=OFF` to build only the new GPU targets.
Caller-supplied CUDA architectures are preserved.

One correctness executable covers all-local, mixed, no-local, and zero-nonzero
schedules, rectangular/tail dimensions, and replay after input changes. All
four modes are checked against double-precision CPU calculations, with poisoned
outputs to catch missing writes. It runs on a nondefault stream and rejects
malformed CSR. On H100 with CUDA 12.6, all 32 output comparisons passed;
Compute Sanitizer memcheck, racecheck, and synccheck reported zero errors.

## Relation to the papers

- [Tile Fusion, ICS 2025](https://doi.org/10.1145/3721145.3730427), §3.1 maps
  wavefronts to GPU kernels and tiles to thread blocks; §5.4 evaluates GPU
  SpMM–SpMM fusion. This implementation adds explicit CUDA Graph replay to that
  execution pattern. Its initial inspector uses fixed producer-row tiles.
- [FusedMM](https://arxiv.org/abs/2011.06391) fuses edge-message computation and
  aggregation (SDDMM–SpMM). It is useful background on intermediate reuse and
  output ownership, but describes a different operation from this first path.
- [CUDA Graphs](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#cuda-graphs)
  documents graph dependency and replay semantics.
