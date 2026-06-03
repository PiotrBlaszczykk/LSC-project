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
mkdir -p "$MPLCONFIGDIR"

python --version

python scripts/00_smoke_test.py --n-samples 500 --batch-size 16
