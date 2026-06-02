#!/bin/bash
#SBATCH --job-name=amazon-reduction
#SBATCH --array=0-15
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/reduction_%A_%a.out
#SBATCH --error=logs/reduction_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# On Ares, load/activate your Python environment before running the job.
# Example:
# module load Python/3.11.5-GCCcore-13.2.0
# source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

METHODS=(pca umap pacmap fitsne)
N_VALUES=(5000 10000 25000 50000)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % ${#METHODS[@]}))
N_INDEX=$((SLURM_ARRAY_TASK_ID / ${#METHODS[@]}))

METHOD="${METHODS[$METHOD_INDEX]}"
N_SAMPLES="${N_VALUES[$N_INDEX]}"

echo "Running method=${METHOD}, n_samples=${N_SAMPLES}"

python scripts/02_run_benchmark.py \
  --method "$METHOD" \
  --n-samples "$N_SAMPLES"
