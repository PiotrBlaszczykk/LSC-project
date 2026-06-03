#!/bin/bash
#SBATCH --job-name=amazon-embed
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

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

python scripts/01_prepare_embeddings.py --n-samples 50000 --batch-size 64
