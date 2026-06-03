# Amazon Reviews HPC LSC benchmark

Simple Large Scale Computing project for AGH LSC classes.

The project benchmarks dimensionality reduction methods for text embeddings
computed from Amazon review texts. It uses only reviews from:

- `McAuley-Lab/Amazon-Reviews-2023`
- `raw_review_Electronics`

No product metadata is used.

## Pipeline

1. Load review texts.
2. Compute SentenceTransformer embeddings.
3. Run PCA, UMAP, PaCMAP and FIt-SNE for several sample sizes.
4. Save one runtime CSV file and one 2D coordinates CSV file per benchmark task.
5. Merge runtime results and plot runtime curves.
6. Plot selected 2D projections for visualization.

The expensive part is executed once on Ares. The generated
`results/<run_name>/coordinates/coords_<method>_<n_samples>.csv` files are the
reusable visualization dataset: they contain the final 2D points and can be
used later to redesign plots, recolor points, join reviews, or run simple
cluster analysis without recomputing embeddings or dimensionality reductions.

Sample sizes:

- 5000
- 10000
- 25000
- 50000

Methods:

- PCA
- UMAP
- PaCMAP
- FIt-SNE

## Local run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a tiny end-to-end smoke test first:

```bash
python scripts/00_smoke_test.py
```

On Ares, an ad-hoc shell wrapper is also available:

```bash
bash smoke_test.sh
```

Optional overrides:

```bash
N_SAMPLES=200 BATCH_SIZE=8 CPU_THREADS=2 bash smoke_test.sh
```

Prepare the largest embedding sample:

```bash
python scripts/01_prepare_embeddings.py --n-samples 50000
```

Run one benchmark manually:

```bash
python scripts/02_run_benchmark.py --method pca --n-samples 5000
```

Merge results:

```bash
python scripts/03_merge_results.py
```

Create the plot:

```bash
python scripts/04_plot_results.py
```

Create a log-scale runtime plot, useful when PCA is much faster than the other
methods:

```bash
python scripts/04_plot_results.py --yscale log --output plots/time_by_method_log.png
```

Plot a 2D embedding projection:

```bash
python scripts/05_plot_coordinates.py --method umap --n-samples 50000
```

Plot the same projection with simple HDBSCAN cluster coloring:

```bash
python scripts/05_plot_coordinates.py --method umap --n-samples 50000 --color-by cluster --output plots/coordinates/embedding_umap_50000_clusters.png
```

## SLURM run

Run a tiny smoke test on Ares before the full benchmark:

```bash
sbatch slurm/smoke_test.sbatch
```

The original full experiment launcher is:

```bash
bash run_jobs_scripts/main_run.sh
```

It submits the full SLURM chain and is treated as `main_run_1` in the report:

1. `slurm/prepare_embeddings.sbatch`
2. `slurm/reduction_array.sbatch`
3. `slurm/postprocess.sbatch`

The reduction job starts only after embeddings are ready, and postprocessing
starts only after the full reduction array succeeds.

If embeddings already exist and you only want to rerun reductions:

```bash
SKIP_PREPARE=1 bash run_jobs_scripts/main_run.sh
```

### Main run 2: cleaner reduction-only benchmark

After `data/embeddings_50000.npy` and `data/reviews_50000.csv` already exist,
run the cleaner benchmark:

```bash
bash run_jobs_scripts/main_run_2.sh
```

This run reuses the same embeddings and focuses on timing the dimensionality
reduction step itself. It avoids timing data loading, output saving, imports and
reducer construction. The default settings are:

- `TIMING_MODE=fit-only`
- `BENCHMARK_REPEATS=3`
- `WARMUP_SAMPLES=1000`
- `RANDOM_STATE=none`

At startup, the launcher also moves any existing root-level outputs from the
first full run into `results/main_run_1`, `plots/main_run_1` and
`logs/main_run_1`. It uses `mv -n`, so existing files are not overwritten.

In this mode each SLURM array task:

1. loads the embeddings before the timer,
2. runs a small warm-up reduction to trigger imports, JIT compilation and
   library setup,
3. creates the reducer outside the timer,
4. measures only `fit` / `fit_transform`,
5. repeats the measurement three times,
6. stores the median in `time_seconds` and all repeat timings in a separate CSV.

`RANDOM_STATE=none` is intentional for this performance run: it lets UMAP use
multiple workers instead of forcing `n_jobs=1`. If you want deterministic
coordinates instead, override it:

```bash
RANDOM_STATE=42 bash run_jobs_scripts/main_run_2.sh
```

Manual equivalent:

```bash
prepare_job=$(sbatch --parsable slurm/prepare_embeddings.sbatch)
reduction_job=$(sbatch --parsable --dependency=afterok:$prepare_job slurm/reduction_array.sbatch)
sbatch --dependency=afterok:$reduction_job slurm/postprocess.sbatch
```

The array runs 16 tasks:

- 4 sample sizes
- 4 reduction methods

With the plain script defaults, each task writes one runtime file and one
coordinates file. In the organized outputs these are stored under
`results/<run_name>/benchmark_<method>_<n_samples>.csv` and
`results/<run_name>/coordinates/coords_<method>_<n_samples>.csv`.
After all jobs finish, postprocessing can be rerun manually, for example:

```bash
python scripts/03_merge_results.py --results-dir results/main_run_2 --output results/main_run_2/results_all.csv
python scripts/04_plot_results.py --input results/main_run_2/results_all.csv --output plots/main_run_2/time_by_method.png
python scripts/05_plot_coordinates.py --results-dir results/main_run_2 --method umap --n-samples 50000 --output plots/main_run_2/coordinates/embedding_umap_50000.png
```

The launchers already submit postprocessing automatically.

## Outputs

Smoke test outputs:

- `data/smoke/reviews_500.csv`
- `data/smoke/embeddings_500.npy`
- `results/smoke/benchmark_<method>_500.csv`
- `results/smoke/coordinates/coords_<method>_500.csv`
- `results/smoke/results_all.csv`
- `plots/smoke/time_by_method.png`
- `plots/smoke/coordinates/embedding_<method>_500.png`
- `logs/smoke_<job_id>.out`
- `logs/smoke_<job_id>.err`

Main run 1 outputs:

- `data/reviews_50000.csv`
- `data/embeddings_50000.npy`
- `logs/main_run_1/prepare_<job_id>.out`
- `logs/main_run_1/prepare_<job_id>.err`
- `results/main_run_1/benchmark_<method>_<n_samples>.csv`
- `results/main_run_1/coordinates/coords_<method>_<n_samples>.csv`
- `results/main_run_1/results_all.csv`
- `plots/main_run_1/time_by_method.png`
- `plots/main_run_1/time_by_method_log.png`
- `plots/main_run_1/coordinates/embedding_<method>_<n_samples>.png`
- `logs/main_run_1/reduction_<array_job_id>_<task_id>.out`
- `logs/main_run_1/reduction_<array_job_id>_<task_id>.err`
- `logs/main_run_1/postprocess_<job_id>.out`
- `logs/main_run_1/postprocess_<job_id>.err`

Main run 2 outputs:

- `logs/main_run_2/main_run_2_<timestamp>.log`
- `logs/main_run_2/reduction_<array_job_id>_<task_id>.out`
- `logs/main_run_2/reduction_<array_job_id>_<task_id>.err`
- `logs/main_run_2/postprocess_<job_id>.out`
- `logs/main_run_2/postprocess_<job_id>.err`
- `results/main_run_2/benchmark_<method>_<n_samples>.csv`
- `results/main_run_2/timings_<method>_<n_samples>.csv`
- `results/main_run_2/coordinates/coords_<method>_<n_samples>.csv`
- `results/main_run_2/results_all.csv`
- `plots/main_run_2/time_by_method.png`
- `plots/main_run_2/time_by_method_log.png`
- `plots/main_run_2/coordinates/embedding_<method>_<n_samples>.png`
- `plots/main_run_2/coordinates/embedding_<method>_50000_clusters.png`

Hugging Face cache is written to `$SCRATCH/hf_cache` by the SLURM scripts, so
model and dataset cache files do not fill the smaller `$HOME` quota.

Most important reusable files:

- `results/main_run_1/results_all.csv` - first merged runtime benchmark table.
- `results/main_run_2/results_all.csv` - cleaner reduction-only runtime table.
- `results/<run_name>/coordinates/coords_<method>_<n_samples>.csv` - final 2D
  coordinates for visualization and analysis.
- `plots/<run_name>/time_by_method.png` - linear runtime scaling plot.
- `plots/<run_name>/time_by_method_log.png` - log-scale runtime plot that makes
  very fast PCA visible.
- `plots/<run_name>/coordinates/*.png` - 2D review maps colored by rating or
  clusters.

## Ares workflow

The simplest workflow is to push changes to GitHub and pull them on Ares:

First-time environment setup:

```bash
cd ~
mkdir -p ~/venvs
module avail Python
# Good Ares choice from the current module list:
module load python/3.12.3-gcccore-13.3.0
python3 -m venv ~/venvs/lsc-amazon
source ~/venvs/lsc-amazon/bin/activate
python -m pip install --upgrade pip setuptools wheel
cd ~/LSC-project
python -m pip install -r requirements.txt
python -c "import datasets, pandas, numpy, sklearn, sentence_transformers, umap, pacmap, openTSNE; print('imports OK')"
```

If installation fails with `Disk quota exceeded`, remove the partial venv and
pip cache before retrying. The requirements file pins CPU-only PyTorch to avoid
downloading large CUDA/NVIDIA wheels.

```bash
deactivate 2>/dev/null || true
rm -rf ~/venvs/lsc-amazon ~/.cache/pip
module load python/3.12.3-gcccore-13.3.0
python3 -m venv ~/venvs/lsc-amazon
source ~/venvs/lsc-amazon/bin/activate
python -m pip install --upgrade pip setuptools wheel
cd ~/LSC-project
python -m pip install --no-cache-dir -r requirements.txt
```

If Python 3.12 causes package build issues, recreate the venv with:

```bash
module load python/3.11.5-gcccore-13.2.0
python3 -m venv ~/venvs/lsc-amazon
```

Regular run after pulling changes:

```bash
cd ~/LSC-project
git pull
source ~/venvs/lsc-amazon/bin/activate
python -m pip install -r requirements.txt
bash smoke_test.sh
bash run_jobs_scripts/main_run.sh
bash run_jobs_scripts/main_run_2.sh
```

Do not run SLURM scripts with `sh`. Use `sbatch`, otherwise SLURM variables and
log naming will not work correctly.

Check jobs:

```bash
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,NodeList -X
```

After the SLURM array finishes:

```bash
source ~/venvs/lsc-amazon/bin/activate
python scripts/03_merge_results.py
python scripts/04_plot_results.py
```

The full launcher already submits postprocessing. The manual commands above are
useful if you change plot styling locally or want to regenerate figures from
existing CSV files.

Download final lightweight outputs from your laptop terminal, not from Ares:

```bash
scp -r plgblaszczykk@login01.ares.cyfronet.pl:~/LSC-project/results .
scp -r plgblaszczykk@login01.ares.cyfronet.pl:~/LSC-project/plots .
scp -r plgblaszczykk@login01.ares.cyfronet.pl:~/LSC-project/logs .
```

## CPU-hours estimate

The benchmark array uses 16 tasks. With the current SLURM settings:

- `--cpus-per-task=8`
- `--time=02:00:00`
- `--array=0-15`

The worst-case allocation is:

```text
16 tasks * 8 CPU * 2 h = 256 CPU-hours
```

Preparing embeddings is run once before the array. If it is submitted as a
separate 8 CPU / 2 h job, reserve another 16 CPU-hours.

This is the minimum cost for one clean run. For the grant, a larger allocation
is reasonable because the project will need setup jobs, failed/debug runs,
reruns after changing parameters and a small reserve for extending the
experiment.

Recommended grant estimate:

```text
about 3000 CPU-hours
```

Suggested breakdown:

```text
10 benchmark array runs  * 256 CPU-hours = 2560 CPU-hours
embedding/setup/debug jobs                    300 CPU-hours
extra safety margin                           140 CPU-hours
------------------------------------------------------------
total                                        3000 CPU-hours
```

The actual final benchmark can still use the simple 16-task SLURM array. The
larger number is a practical allocation request, not a requirement to consume
all CPU-hours.

For `main_run_2`, the array wall-time limit is raised to 3 hours because each
task performs warm-up plus three timed repeats:

```text
16 tasks * 8 CPU * 3 h = 384 CPU-hours worst-case allocation
```

In practice it should be much lower, but this bound is useful when estimating
grant usage.
