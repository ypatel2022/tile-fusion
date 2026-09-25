# GNN linear-layer benchmark results

Measured on 2026-09-24, AMD EPYC 9655 CPU, GCC 12.3 / MKL 2023.2,
Release (`-O3 -march=native`), OpenMP close binding to cores. Slurm job
61385580 completed successfully. These are observations from this run.
The suite covered 32 configurations and 96 processes; all 2,016 timed output
checks and 864 warmup checks passed. Scalar reference checks passed in all
processes. The separate one-node/all-fused/no-fusion regressions passed, and
AddressSanitizer confirmed that skipping the inspector's empty trailing tile
fixes the prior out-of-bounds row-index read.

Operation: H = X W, Y = normalized(A) H; float32 CPU kernels. Features and weights are deterministic synthetic data, including for PubMed. This measures the linear operations, not full GNN training.

3 independent processes per case; each has 3 warmups and 7 timed trials per implementation. Implementation order rotates between processes. Times below are medians of process medians; speedup > 1 favors fusion. Every warmup and trial checks the full output against MKL, whose reference also receives a scalar double-precision check on 16 sampled rows.

| Graph | Width | Threads | Cache KiB | MKL ms | Unfused AVX2 ms | Fused ms | Speedup vs AVX2 | Repeat range | Inspection ms | Break-even reuses | Fused rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_65536 | 64 | 1 | 1024 | 10.053 | 8.290 | 8.041 | 1.03x | 1.02–1.06x | 1.140 | 5 | 98.4% |
| local_65536 | 64 | 1 | 4096 | 10.001 | 8.308 | 8.310 | 1.00x | 0.99–1.00x | 0.798 | never | 99.6% |
| local_65536 | 64 | 8 | 1024 | 1.872 | 1.647 | 1.232 | 1.34x | 1.26–1.36x | 1.116 | 3 | 98.4% |
| local_65536 | 64 | 8 | 4096 | 1.749 | 1.546 | 1.212 | 1.28x | 1.23–1.28x | 0.821 | 3 | 99.6% |
| local_65536 | 128 | 1 | 1024 | 32.394 | 28.640 | 27.093 | 1.06x | 1.04–1.08x | 1.421 | 1 | 96.7% |
| local_65536 | 128 | 1 | 4096 | 32.154 | 27.767 | 27.013 | 1.03x | 0.97–1.09x | 0.908 | 2 | 99.2% |
| local_65536 | 128 | 8 | 1024 | 5.064 | 4.642 | 3.671 | 1.26x | 1.25–1.33x | 1.501 | 2 | 96.7% |
| local_65536 | 128 | 8 | 4096 | 5.174 | 4.746 | 3.706 | 1.28x | 1.23–1.35x | 0.916 | 1 | 99.2% |
| local_262144 | 64 | 1 | 1024 | 49.158 | 45.220 | 36.428 | 1.24x | 1.18–1.25x | 4.314 | 1 | 98.4% |
| local_262144 | 64 | 1 | 4096 | 48.555 | 43.006 | 35.554 | 1.21x | 1.21–1.26x | 3.056 | 1 | 99.6% |
| local_262144 | 64 | 8 | 1024 | 9.695 | 9.815 | 5.302 | 1.85x | 1.77–1.92x | 4.434 | 1 | 98.4% |
| local_262144 | 64 | 8 | 4096 | 9.736 | 9.593 | 5.479 | 1.75x | 1.74–1.77x | 3.182 | 1 | 99.6% |
| local_262144 | 128 | 1 | 1024 | 136.777 | 128.736 | 117.702 | 1.09x | 1.09–1.13x | 7.300 | 1 | 96.7% |
| local_262144 | 128 | 1 | 4096 | 136.288 | 126.255 | 109.759 | 1.15x | 1.14–1.16x | 3.742 | 1 | 99.2% |
| local_262144 | 128 | 8 | 1024 | 24.926 | 24.756 | 16.315 | 1.52x | 1.49–1.52x | 7.565 | 1 | 96.7% |
| local_262144 | 128 | 8 | 4096 | 23.677 | 23.385 | 16.187 | 1.44x | 1.42–1.45x | 3.780 | 1 | 99.2% |
| shuffled_65536 | 64 | 1 | 1024 | 13.547 | 10.927 | 11.237 | 0.97x | 0.96–0.99x | 141.233 | never | 0.0% |
| shuffled_65536 | 64 | 1 | 4096 | 14.381 | 11.046 | 12.133 | 0.91x | 0.91–0.99x | 166.624 | never | 0.0% |
| shuffled_65536 | 64 | 8 | 1024 | 3.894 | 3.268 | 3.022 | 1.08x | 1.04–1.09x | 153.378 | 624 | 0.0% |
| shuffled_65536 | 64 | 8 | 4096 | 3.833 | 3.263 | 3.280 | 0.99x | 0.97–1.02x | 176.262 | never | 0.0% |
| shuffled_65536 | 128 | 1 | 1024 | 51.845 | 38.090 | 37.585 | 1.01x | 1.01–1.02x | 132.829 | 264 | 0.0% |
| shuffled_65536 | 128 | 1 | 4096 | 52.271 | 38.451 | 37.785 | 1.02x | 1.01–1.03x | 154.671 | 233 | 0.0% |
| shuffled_65536 | 128 | 8 | 1024 | 10.818 | 9.000 | 7.924 | 1.14x | 1.11–1.15x | 141.111 | 132 | 0.0% |
| shuffled_65536 | 128 | 8 | 4096 | 10.693 | 8.991 | 8.055 | 1.12x | 1.11–1.12x | 163.182 | 175 | 0.0% |
| pubmed | 64 | 1 | 1024 | 2.254 | 1.935 | 1.980 | 0.98x | 0.97–0.98x | 12.847 | never | 2.4% |
| pubmed | 64 | 1 | 4096 | 2.241 | 1.924 | 1.972 | 0.98x | 0.96–0.98x | 7.467 | never | 12.1% |
| pubmed | 64 | 8 | 1024 | 0.861 | 0.785 | 0.767 | 1.02x | 1.02–1.04x | 13.372 | 746 | 2.4% |
| pubmed | 64 | 8 | 4096 | 0.829 | 0.785 | 0.926 | 0.85x | 0.81–0.86x | 7.700 | never | 12.1% |
| pubmed | 128 | 1 | 1024 | 7.582 | 7.059 | 7.277 | 0.97x | 0.96–0.97x | 13.224 | never | 1.2% |
| pubmed | 128 | 1 | 4096 | 7.642 | 7.064 | 7.202 | 0.98x | 0.97–0.98x | 10.633 | never | 5.8% |
| pubmed | 128 | 8 | 1024 | 1.864 | 1.769 | 1.708 | 1.04x | 1.03–1.06x | 14.152 | 235 | 1.2% |
| pubmed | 128 | 8 | 4096 | 1.941 | 1.765 | 1.688 | 1.05x | 1.03–1.05x | 11.145 | 145 | 5.8% |

Inspection is excluded from execution. Break-even is ceil(inspection / (unfused AVX2 execution − fused execution)); it estimates how many uses of the same graph repay scheduling, assuming these execution times persist. Input loading, allocation, normalization, buffer clearing, warmup, and verification are outside execution timing. Buffers are reused. Cache sizes are explicit inspector budgets, not measured cache occupancy.

The ordered synthetic graph is deliberately favorable to locality. The shuffled case has identical topology with relabeled vertices. PubMed uses its bundled node ordering, undirected edges, and self-loops. All declared cases, including slowdowns, are included.

This compares complete implementations, including differences in GEMM tiling and threading. A speedup with zero fused rows is not evidence of inter-operation data reuse.
