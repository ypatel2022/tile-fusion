#!/bin/bash
#SBATCH --job-name=gemm-dag-perf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h100:1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=tests/cuda_graph/perf-%j.out

set -euo pipefail

# Submit from the repository root. Optional arguments: n replays trials warmup.
# sbatch tests/cuda_graph/perf.sbatch 64 100 7 10
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 cuda/12.6 cmake/3.27.7

cmake -S tests/cuda_graph -B "${SLURM_TMPDIR}/gemm-dag-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${SLURM_TMPDIR}/gemm-dag-build" --target gemm_dag_benchmark --parallel 1
srun "${SLURM_TMPDIR}/gemm-dag-build/gemm_dag_benchmark" "$@" \
    > "tests/cuda_graph/perf-${SLURM_JOB_ID}.csv"
