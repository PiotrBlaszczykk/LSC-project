#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs/main_run_1 logs/main_run_2 logs/smoke
mkdir -p results/main_run_1 results/main_run_2 plots/main_run_1 plots/main_run_2/coordinates

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/logs/main_run_2/main_run_2_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

archive_main_run_1_artifacts() {
  shopt -s nullglob

  local result_files=(results/benchmark_*.csv)
  if [ "${#result_files[@]}" -gt 0 ]; then
    mv -n "${result_files[@]}" results/main_run_1/
  fi
  if [ -f results/results_all.csv ]; then
    mv -n results/results_all.csv results/main_run_1/
  fi
  if [ -d results/coordinates ] && [ ! -e results/main_run_1/coordinates ]; then
    mv results/coordinates results/main_run_1/coordinates
  fi

  local plot_files=(plots/time_by_method*.png)
  if [ "${#plot_files[@]}" -gt 0 ]; then
    mv -n "${plot_files[@]}" plots/main_run_1/
  fi
  if [ -d plots/coordinates ] && [ ! -e plots/main_run_1/coordinates ]; then
    mv plots/coordinates plots/main_run_1/coordinates
  fi

  local main_log_files=(logs/main_run_*.log logs/prepare_*.* logs/reduction_*.* logs/postprocess_*)
  if [ "${#main_log_files[@]}" -gt 0 ]; then
    mv -n "${main_log_files[@]}" logs/main_run_1/
  fi

  local smoke_log_files=(logs/smoke_adhoc_*.log)
  if [ "${#smoke_log_files[@]}" -gt 0 ]; then
    mv -n "${smoke_log_files[@]}" logs/smoke/
  fi

  shopt -u nullglob
}

PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.3-gcccore-13.3.0}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/lsc-amazon}"
DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results/main_run_2}"
PLOTS_DIR="${PLOTS_DIR:-plots/main_run_2}"
TIMING_MODE="${TIMING_MODE:-fit-only}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-3}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1000}"
RANDOM_STATE="${RANDOM_STATE:-none}"
CREATE_CLUSTER_PLOTS="${CREATE_CLUSTER_PLOTS:-1}"

echo "Project root: $PROJECT_ROOT"
echo "Log file: $LOG_FILE"
echo "Python module: $PYTHON_MODULE"
echo "Venv: $VENV_PATH"
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Plots dir: $PLOTS_DIR"
echo "Timing mode: $TIMING_MODE"
echo "Benchmark repeats: $BENCHMARK_REPEATS"
echo "Warm-up samples: $WARMUP_SAMPLES"
echo "Random state: $RANDOM_STATE"
echo "Create cluster plots: $CREATE_CLUSTER_PLOTS"

echo "Archiving existing root-level main_run_1 artifacts, if any..."
archive_main_run_1_artifacts

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

if [ ! -f "$DATA_DIR/embeddings_50000.npy" ]; then
  echo "Missing $DATA_DIR/embeddings_50000.npy."
  echo "Run main_run.sh first or prepare embeddings manually before main_run_2."
  exit 1
fi

if [ ! -f "$DATA_DIR/reviews_50000.csv" ]; then
  echo "Missing $DATA_DIR/reviews_50000.csv."
  echo "Coordinates can still be computed, but rating/product info would be missing."
  echo "Run main_run.sh first or prepare reviews manually before main_run_2."
  exit 1
fi

SBATCH_EXPORT="ALL,VENV_PATH=$VENV_PATH,PYTHON_MODULE=$PYTHON_MODULE,DATA_DIR=$DATA_DIR,RESULTS_DIR=$RESULTS_DIR,PLOTS_DIR=$PLOTS_DIR,TIMING_MODE=$TIMING_MODE,BENCHMARK_REPEATS=$BENCHMARK_REPEATS,WARMUP_SAMPLES=$WARMUP_SAMPLES,RANDOM_STATE=$RANDOM_STATE,CREATE_CLUSTER_PLOTS=$CREATE_CLUSTER_PLOTS"

echo "Submitting reduction-only benchmark array..."
reduction_submit="$(
  sbatch \
    --parsable \
    --export="$SBATCH_EXPORT" \
    slurm/reduction_array_main_run_2.sbatch
)"
reduction_job_id="${reduction_submit%%;*}"
echo "Reduction array job id: $reduction_job_id"

echo "Submitting postprocess job..."
postprocess_submit="$(
  sbatch \
    --parsable \
    --dependency="afterok:$reduction_job_id" \
    --export="$SBATCH_EXPORT" \
    slurm/postprocess_main_run_2.sbatch
)"
postprocess_job_id="${postprocess_submit%%;*}"
echo "Postprocess job id: $postprocess_job_id"

echo
echo "Main run 2 submitted."
echo "  reductions:  $reduction_job_id"
echo "  postprocess: $postprocess_job_id"
echo
echo "Check status:"
echo "  squeue -u \"\$USER\""
echo "  sacct -j $reduction_job_id,$postprocess_job_id --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS"
echo
echo "Expected final outputs:"
echo "  $RESULTS_DIR/results_all.csv"
echo "  $RESULTS_DIR/benchmark_<method>_<n_samples>.csv"
echo "  $RESULTS_DIR/timings_<method>_<n_samples>.csv"
echo "  $RESULTS_DIR/coordinates/coords_<method>_<n_samples>.csv"
echo "  $PLOTS_DIR/time_by_method.png"
echo "  $PLOTS_DIR/time_by_method_log.png"
echo "  $PLOTS_DIR/coordinates/embedding_<method>_<n_samples>.png"
echo "  $PLOTS_DIR/coordinates/embedding_<method>_50000_clusters.png"
