#!/bin/bash
#SBATCH --job-name=amazon-post
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/postprocess_%j.out
#SBATCH --error=logs/postprocess_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results plots/coordinates

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

python scripts/03_merge_results.py
python scripts/04_plot_results.py
python scripts/04_plot_results.py \
  --yscale log \
  --output plots/time_by_method_log.png

METHODS=(pca umap pacmap fitsne)
N_VALUES=(5000 10000 25000 50000)

for METHOD in "${METHODS[@]}"; do
  for N_SAMPLES in "${N_VALUES[@]}"; do
    python scripts/05_plot_coordinates.py \
      --method "$METHOD" \
      --n-samples "$N_SAMPLES" \
      --output "plots/coordinates/embedding_${METHOD}_${N_SAMPLES}.png"
  done
done

CREATE_CLUSTER_PLOTS="${CREATE_CLUSTER_PLOTS:-1}"
if [ "$CREATE_CLUSTER_PLOTS" = "1" ]; then
  python scripts/05_plot_coordinates.py \
    --method umap \
    --n-samples 50000 \
    --color-by cluster \
    --output plots/coordinates/embedding_umap_50000_clusters.png

  python scripts/05_plot_coordinates.py \
    --method pacmap \
    --n-samples 50000 \
    --color-by cluster \
    --output plots/coordinates/embedding_pacmap_50000_clusters.png
fi

echo "Postprocess finished."
echo "Runtime plot: plots/time_by_method.png"
echo "2D plots: plots/coordinates/"
