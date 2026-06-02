#!/bin/bash
#SBATCH --job-name=amazon-smoke
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/smoke results/smoke plots/smoke

# On Ares, load/activate your Python environment before running the job.
# Example:
# module load Python/3.11.5-GCCcore-13.2.0
# source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/00_smoke_test.py --n-samples 500 --batch-size 16
