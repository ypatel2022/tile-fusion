#!/bin/bash
#SBATCH --job-name=gemm-dag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h100:1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=tests/cuda_graph/gemm-dag-%j.out

set -euo pipefail

# Submit from the repository root:
# sbatch tests/cuda_graph/run.sbatch
cd "${SLURM_SUBMIT_DIR}/tests/cuda_graph"
module load StdEnv/2023 cuda/12.6

# Build in this job's temporary directory so concurrent jobs have separate binaries.
nvcc -std=c++17 -O2 -arch=sm_90 gemm_dag.cu -o "${SLURM_TMPDIR}/gemm_dag"

# Run once on the allocated GPU; the program checks all 30 GEMM outputs.
srun "${SLURM_TMPDIR}/gemm_dag"
