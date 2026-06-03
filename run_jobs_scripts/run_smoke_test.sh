#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs data/smoke results/smoke plots/smoke

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/logs/smoke_adhoc_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Project root: $PROJECT_ROOT"
echo "Log file: $LOG_FILE"

PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.3-gcccore-13.3.0}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/lsc-amazon}"
N_SAMPLES="${N_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CPU_THREADS="${CPU_THREADS:-4}"

if command -v module >/dev/null 2>&1; then
  module load "$PYTHON_MODULE"
else
  echo "Warning: module command not found; using current Python from PATH."
fi

if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing venv: $VENV_PATH"
  echo "Create it first, for example:"
  echo "  module load $PYTHON_MODULE"
  echo "  python3 -m venv $VENV_PATH"
  exit 1
fi

source "$VENV_PATH/bin/activate"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.matplotlib}"

if [ -z "${HF_HOME:-}" ]; then
  if [ -z "${SCRATCH:-}" ]; then
    echo "SCRATCH is not set. Set HF_HOME manually or run this on Ares."
    exit 1
  fi
  export HF_HOME="$SCRATCH/hf_cache"
fi

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

echo "Python module: $PYTHON_MODULE"
echo "Venv: $VENV_PATH"
echo "Python: $(python --version)"
echo "HF_HOME: $HF_HOME"
echo "N_SAMPLES: $N_SAMPLES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "CPU_THREADS: $CPU_THREADS"

python - <<'PY'
import torch

print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())

import datasets  # noqa: F401
import numpy  # noqa: F401
import pandas  # noqa: F401
import sklearn  # noqa: F401
import sentence_transformers  # noqa: F401
import umap  # noqa: F401
import pacmap  # noqa: F401
import openTSNE  # noqa: F401

print("Imports OK")
PY

python scripts/00_smoke_test.py \
  --n-samples "$N_SAMPLES" \
  --batch-size "$BATCH_SIZE"

echo
echo "Smoke test finished."
echo "Results:"
echo "  $PROJECT_ROOT/results/smoke/results_all.csv"
echo "  $PROJECT_ROOT/plots/smoke/time_by_method.png"
echo "Log:"
echo "  $LOG_FILE"
