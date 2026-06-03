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
`results/coordinates/coords_<method>_<n_samples>.csv` files are the reusable
visualization dataset: they contain the final 2D points and can be used later
to redesign plots, recolor points, join reviews, or run simple cluster analysis
without recomputing embeddings or dimensionality reductions.

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

The recommended full experiment launcher is:

```bash
bash run_jobs_scripts/main_run.sh
```

It submits the full SLURM chain:

1. `slurm/prepare_embeddings.sbatch`
2. `slurm/reduction_array.sbatch`
3. `slurm/postprocess.sbatch`

The reduction job starts only after embeddings are ready, and postprocessing
starts only after the full reduction array succeeds.

If embeddings already exist and you only want to rerun reductions:

```bash
SKIP_PREPARE=1 bash run_jobs_scripts/main_run.sh
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

Each task writes one file to `results/benchmark_<method>_<n_samples>.csv`.
It also writes reduced 2D coordinates to
`results/coordinates/coords_<method>_<n_samples>.csv`.
After all jobs finish, run:

```bash
python scripts/03_merge_results.py
python scripts/04_plot_results.py
python scripts/05_plot_coordinates.py --method umap --n-samples 50000
```

The final plot is saved to `plots/time_by_method.png`.

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

Full benchmark outputs:

- `data/reviews_50000.csv`
- `data/embeddings_50000.npy`
- `logs/prepare_<job_id>.out`
- `logs/prepare_<job_id>.err`
- `results/benchmark_<method>_<n_samples>.csv`
- `results/coordinates/coords_<method>_<n_samples>.csv`
- `results/results_all.csv`
- `plots/time_by_method.png`
- `plots/time_by_method_log.png`
- `plots/coordinates/embedding_<method>_<n_samples>.png`
- `logs/reduction_<array_job_id>_<task_id>.out`
- `logs/reduction_<array_job_id>_<task_id>.err`
- `logs/postprocess_<job_id>.out`
- `logs/postprocess_<job_id>.err`

Hugging Face cache is written to `$SCRATCH/hf_cache` by the SLURM scripts, so
model and dataset cache files do not fill the smaller `$HOME` quota.

Most important reusable files:

- `results/results_all.csv` - merged runtime benchmark table.
- `results/coordinates/coords_<method>_<n_samples>.csv` - final 2D coordinates
  for visualization and analysis.
- `plots/time_by_method.png` - linear runtime scaling plot.
- `plots/time_by_method_log.png` - log-scale runtime plot that makes very fast
  PCA visible.
- `plots/coordinates/*.png` - 2D review maps colored by rating or clusters.

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
