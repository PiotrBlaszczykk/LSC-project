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

PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.3-gcccore-13.3.0}"
module load "$PYTHON_MODULE"

VENV_PATH="${VENV_PATH:-$HOME/venvs/lsc-amazon}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing venv: $VENV_PATH"
  echo "Create it first, for example: python3 -m venv $VENV_PATH"
  exit 1
fi

source "$VENV_PATH/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SLURM_SUBMIT_DIR/.matplotlib}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

python --version

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
