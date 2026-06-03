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
4. Save one CSV file per benchmark task.
5. Merge results and plot runtime curves.

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

## SLURM run

Run a tiny smoke test on Ares before the full benchmark:

```bash
sbatch slurm/smoke_test.sbatch
```

Prepare embeddings once as a SLURM job, then submit the benchmark array:

```bash
sbatch slurm/prepare_embeddings.sbatch
sbatch slurm/reduction_array.sbatch
```

The array runs 16 tasks:

- 4 sample sizes
- 4 reduction methods

Each task writes one file to `results/benchmark_<method>_<n_samples>.csv`.
After all jobs finish, run:

```bash
python scripts/03_merge_results.py
python scripts/04_plot_results.py
```

The final plot is saved to `plots/time_by_method.png`.

## Outputs

Smoke test outputs:

- `data/smoke/reviews_500.csv`
- `data/smoke/embeddings_500.npy`
- `results/smoke/benchmark_<method>_500.csv`
- `results/smoke/results_all.csv`
- `plots/smoke/time_by_method.png`
- `logs/smoke_<job_id>.out`
- `logs/smoke_<job_id>.err`

Full benchmark outputs:

- `data/reviews_50000.csv`
- `data/embeddings_50000.npy`
- `logs/prepare_<job_id>.out`
- `logs/prepare_<job_id>.err`
- `results/benchmark_<method>_<n_samples>.csv`
- `results/results_all.csv`
- `plots/time_by_method.png`
- `logs/reduction_<array_job_id>_<task_id>.out`
- `logs/reduction_<array_job_id>_<task_id>.err`

Hugging Face cache is written to `$SCRATCH/hf_cache` by the SLURM scripts, so
model and dataset cache files do not fill the smaller `$HOME` quota.

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
sbatch slurm/smoke_test.sbatch
sbatch slurm/prepare_embeddings.sbatch
sbatch slurm/reduction_array.sbatch
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
