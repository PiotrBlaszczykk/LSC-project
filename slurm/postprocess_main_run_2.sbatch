#!/bin/bash
#SBATCH --job-name=amazon-post-r2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/main_run_2/postprocess_%j.out
#SBATCH --error=logs/main_run_2/postprocess_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/main_run_2 results/main_run_2 plots/main_run_2/coordinates

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

RESULTS_DIR="${RESULTS_DIR:-results/main_run_2}"
PLOTS_DIR="${PLOTS_DIR:-plots/main_run_2}"
CREATE_CLUSTER_PLOTS="${CREATE_CLUSTER_PLOTS:-1}"

python scripts/03_merge_results.py \
  --results-dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/results_all.csv"

python scripts/04_plot_results.py \
  --input "$RESULTS_DIR/results_all.csv" \
  --output "$PLOTS_DIR/time_by_method.png"

python scripts/04_plot_results.py \
  --input "$RESULTS_DIR/results_all.csv" \
  --yscale log \
  --output "$PLOTS_DIR/time_by_method_log.png"

METHODS=(pca umap pacmap fitsne)
N_VALUES=(5000 10000 25000 50000)

for METHOD in "${METHODS[@]}"; do
  for N_SAMPLES in "${N_VALUES[@]}"; do
    python scripts/05_plot_coordinates.py \
      --method "$METHOD" \
      --n-samples "$N_SAMPLES" \
      --results-dir "$RESULTS_DIR" \
      --output "$PLOTS_DIR/coordinates/embedding_${METHOD}_${N_SAMPLES}.png"
  done
done

if [ "$CREATE_CLUSTER_PLOTS" = "1" ]; then
  for METHOD in "${METHODS[@]}"; do
    python scripts/05_plot_coordinates.py \
      --method "$METHOD" \
      --n-samples 50000 \
      --results-dir "$RESULTS_DIR" \
      --color-by cluster \
      --output "$PLOTS_DIR/coordinates/embedding_${METHOD}_50000_clusters.png"
  done
fi

echo "Postprocess finished."
echo "Runtime plot: $PLOTS_DIR/time_by_method.png"
echo "Runtime log plot: $PLOTS_DIR/time_by_method_log.png"
echo "2D plots: $PLOTS_DIR/coordinates/"
