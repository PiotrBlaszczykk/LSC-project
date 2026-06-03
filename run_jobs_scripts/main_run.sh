#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs data results plots

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/logs/main_run_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.3-gcccore-13.3.0}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/lsc-amazon}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
CREATE_CLUSTER_PLOTS="${CREATE_CLUSTER_PLOTS:-1}"

echo "Project root: $PROJECT_ROOT"
echo "Log file: $LOG_FILE"
echo "Python module: $PYTHON_MODULE"
echo "Venv: $VENV_PATH"
echo "Skip prepare: $SKIP_PREPARE"
echo "Create cluster plots: $CREATE_CLUSTER_PLOTS"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Missing sbatch. Run this script on Ares login node."
  exit 1
fi

if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing venv: $VENV_PATH"
  echo "Create it first and install requirements."
  exit 1
fi

if [ -z "${SCRATCH:-}" ]; then
  echo "SCRATCH is not set. Run this on Ares or export SCRATCH manually."
  exit 1
fi

if [ "$SKIP_PREPARE" = "1" ] && [ ! -f data/embeddings_50000.npy ]; then
  echo "SKIP_PREPARE=1, but data/embeddings_50000.npy is missing."
  exit 1
fi

SBATCH_EXPORT="ALL,VENV_PATH=$VENV_PATH,PYTHON_MODULE=$PYTHON_MODULE,CREATE_CLUSTER_PLOTS=$CREATE_CLUSTER_PLOTS"

prepare_job_id=""
if [ "$SKIP_PREPARE" = "1" ]; then
  echo "Skipping embedding preparation; reusing data/embeddings_50000.npy"
else
  echo "Submitting embedding preparation job..."
  prepare_submit="$(sbatch --parsable --export="$SBATCH_EXPORT" slurm/prepare_embeddings.sbatch)"
  prepare_job_id="${prepare_submit%%;*}"
  echo "Prepare job id: $prepare_job_id"
fi

echo "Submitting reduction array..."
if [ -n "$prepare_job_id" ]; then
  reduction_submit="$(
    sbatch \
      --parsable \
      --dependency="afterok:$prepare_job_id" \
      --export="$SBATCH_EXPORT" \
      slurm/reduction_array.sbatch
  )"
else
  reduction_submit="$(sbatch --parsable --export="$SBATCH_EXPORT" slurm/reduction_array.sbatch)"
fi
reduction_job_id="${reduction_submit%%;*}"
echo "Reduction array job id: $reduction_job_id"

echo "Submitting postprocess job..."
postprocess_submit="$(
  sbatch \
    --parsable \
    --dependency="afterok:$reduction_job_id" \
    --export="$SBATCH_EXPORT" \
    slurm/postprocess.sbatch
)"
postprocess_job_id="${postprocess_submit%%;*}"
echo "Postprocess job id: $postprocess_job_id"

echo
echo "Main experiment submitted."
if [ -n "$prepare_job_id" ]; then
  echo "  prepare:     $prepare_job_id"
fi
echo "  reductions:  $reduction_job_id"
echo "  postprocess: $postprocess_job_id"
echo
echo "Check status:"
echo "  squeue -u \"\$USER\""
echo "  sacct -j $postprocess_job_id --format=JobID,State,ExitCode,Elapsed,NodeList -X"
echo
echo "Expected final outputs:"
echo "  results/results_all.csv"
echo "  results/coordinates/coords_<method>_<n_samples>.csv"
echo "  plots/time_by_method.png"
echo "  plots/coordinates/embedding_<method>_<n_samples>.png"
echo "  plots/coordinates/embedding_umap_50000_clusters.png"
echo "  plots/coordinates/embedding_pacmap_50000_clusters.png"
