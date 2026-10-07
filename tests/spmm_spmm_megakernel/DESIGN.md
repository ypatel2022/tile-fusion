# One launch for `Y = A (A X)`

The original fused kernel computes H and any Y rows whose H dependencies fit
inside its row tile. A second kernel finishes the other Y rows. This experiment
keeps those baselines and adds three one-launch methods.

For an eight-row tridiagonal A with four rows per tile:

| Producer block | Computes H | Y rows it can finish locally | Deferred Y row |
| --- | --- | --- | --- |
| 0 | H0–H3 | Y0–Y2 | Y3 needs H2, H3, H4 |
| 1 | H4–H7 | Y5–Y7 | Y4 needs H3, H4, H5 |

Each deferred group counts its distinct producer tiles. Both groups above need
two producers. A producer computes H, finishes its local Y rows, then notifies
every dependent group. The last notification computes that group's deferred Y.
It already has all the H it needs; it never waits for another block.

`megakernel_events_global` reads H from global memory. The shared variant keeps
its own producer H tile in shared memory and uses it wherever a consumer needs
those rows. H from other blocks goes through global memory; shared memory is
private to its owning block. The initial versions wrote every H row globally.

The shared variant now marks every H row referenced by any deferred Y row and
stores only those rows globally. This includes references within the consumer's
home tile: another producer can execute its callback. Local Y reads its own
shared tile; every possible global H read by a deferred callback has a current
producer store. Every H row is still computed. The topology-only byte mask is
uploaded during setup and omitted when all H rows need stores. The benchmark
retains the full H allocation, counts mask bytes in schedule storage, and reports
global H writes separately.

The schedule follows the original inspector's tile-membership test. That small
test is repeated here because its class header defines GPU kernels and cannot be
included in a second CUDA translation unit without duplicate symbols. Local and
deferred row lists partition Y; producer dependencies are deduplicated.

## Visibility and progress

All producer threads write H and reach a block barrier. The leader publishes
completion with a device-scope acquire/release compare-and-swap. Notifications
form a chain on the same event word, so the final notifier observes every
producer's H writes. A block barrier passes that visibility to the consumer
threads. Cross-block H reads use ordinary global loads. Consumer callbacks keep
the producer's shared tile intact and synchronize before reusing the trigger.

One 64-bit event word holds a launch epoch and remaining producer count. Each
launch advances the epoch, including warmups. The first notification initializes
the count for that epoch; the transition to zero owns the consumer. There is no
per-call memset or reset kernel. Ordered default-stream invocations cannot
overlap; epoch exhaustion requires a new plan. Values may change at the same
addresses; changes to shape, topology, tile size or pointers require new setup.

Ordinary blocks never wait for an unscheduled producer. Their computation and
notification work is finite, so filling the GPU with waiting blocks cannot
deadlock this schedule. Atomic retries resolve contention rather than waiting
for a dependency. Each H row and each Y row has one owner.

`megakernel_barrier` is the coarse control: compute all H, synchronize the
cooperative grid, then compute all Y. Its resident grid is bounded by the compiled
kernel's occupancy and the SM count. It uses CUDA's cooperative launch, rather
than an ordinary-grid spin barrier. Registers, shared memory and occupancy are
measured from the actual kernels; they can limit performance even with one launch.

The fixed reference, sweep and timing rules are in [EVALUATION.md](EVALUATION.md).
Launch traces must include reset/helper work before a method is reported as one
launch. Profiler durations stay separate from submission/completion timings.

Sources: [Event Tensor](https://arxiv.org/pdf/2604.13327),
[CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html),
[cooperative groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html).
