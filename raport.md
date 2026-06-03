# Experiment Report: Amazon Reviews Text Embeddings and Dimensionality Reduction

## Project Goal

The goal of this experiment was to build a small but complete pipeline for
analyzing customer reviews from a large text dataset. The project connects two
contexts:

- **Large Scale Computing**: running the computational pipeline on the Ares HPC
  cluster with SLURM and measuring dimensionality reduction runtime.
- **Large-scale data visualization**: producing reusable 2D coordinates for
  review embeddings, so the results can later be visualized, recolored, joined
  with review text, or analyzed without recomputing expensive reductions.

Main pipeline:

```text
Amazon Reviews
-> review text
-> SentenceTransformer embeddings, 384D
-> PCA / UMAP / PaCMAP / FIt-SNE
-> 2D coordinates + runtime measurement
-> CSV files + plots
```

## Dataset

Dataset:

- `McAuley-Lab/Amazon-Reviews-2023`
- subset: `raw_review_Electronics`

Only review data was used. Product metadata from `raw_meta_Electronics` was not
joined into the benchmark. The retained fields are:

- `text` - review text,
- `rating` - review rating,
- `parent_asin` - product identifier, kept for reference,
- `point_id` - review index within the sampled dataset.

Full sample artifacts:

- `data/reviews_50000.csv`
- `data/embeddings_50000.npy`

From the embedding preparation log:

```text
Embedding shape: (50000, 384)
Encoding time_seconds: 172.85
```

This means that 50,000 reviews were encoded into 384-dimensional text
embeddings using `sentence-transformers/all-MiniLM-L6-v2`.

## Runtime Environment

The experiment was executed on the Ares HPC cluster using SLURM.

Environment:

- Python `3.12.3`,
- virtual environment: `~/venvs/lsc-amazon`,
- CPU-only PyTorch: `torch==2.5.1+cpu`,
- Hugging Face cache stored under `$SCRATCH/hf_cache` to avoid filling the
  smaller `$HOME` quota.

Main launcher:

```bash
bash run_jobs_scripts/main_run.sh
```

The launcher submits a SLURM job chain:

```text
slurm/prepare_embeddings.sbatch
-> slurm/reduction_array.sbatch
-> slurm/postprocess.sbatch
```

SLURM dependencies (`afterok`) ensure that:

- dimensionality reductions start only after embeddings are prepared,
- postprocessing starts only after the full reduction array completes.

## Experiment Scope

Sample sizes:

```text
5000, 10000, 25000, 50000
```

Dimensionality reduction methods:

```text
PCA, UMAP, PaCMAP, FIt-SNE
```

Number of reduction jobs:

```text
4 sample sizes * 4 methods = 16 SLURM array tasks
```

Each reduction task produced:

- one runtime CSV file,
- one 2D coordinate CSV file.

## Output Artifacts

Most important `main_run_1` files:

- `results/main_run_1/results_all.csv` - merged runtime benchmark table.
- `results/main_run_1/benchmark_<method>_<n_samples>.csv` - individual runtime
  measurements.
- `results/main_run_1/coordinates/coords_<method>_<n_samples>.csv` - reusable 2D
  coordinates for review embeddings.
- `plots/main_run_1/time_by_method.png` - runtime scaling plot.
- `plots/main_run_1/time_by_method_log.png` - log-scale runtime plot.
- `plots/main_run_1/coordinates/*.png` - 2D review maps.

Artifact completeness after the full run:

```text
benchmark CSV files: 16
coordinate CSV files: 16
plots/main_run_1/coordinates PNG files: 20
```

The logs did not contain hard failures such as `ERROR`, `Traceback`, `FAILED`,
`Killed`, or `OutOfMemory`.

## Runtime Results

Table from `results/main_run_1/results_all.csv`:

| Method | 5000 | 10000 | 25000 | 50000 |
|---|---:|---:|---:|---:|
| PCA | 0.0277 s | 0.0284 s | 0.0404 s | 0.0595 s |
| UMAP | 39.4494 s | 54.7491 s | 54.7298 s | 90.4903 s |
| PaCMAP | 3.5480 s | 24.2231 s | 11.0863 s | 21.7893 s |
| FIt-SNE | 42.9014 s | 49.1715 s | 67.5071 s | 87.8828 s |

Linear runtime plot:

![Runtime plot](plots/main_run_1/time_by_method.png)

## How Runtime Was Measured

Runtime was measured in `scripts/02_run_benchmark.py` using:

```python
embeddings = load_embeddings(...)
start = time.perf_counter()
coordinates = reduce_embeddings(method, embeddings, random_state)
elapsed = time.perf_counter() - start
```

The reported `time_seconds` therefore does **not** include:

- Python process startup,
- SLURM scheduling overhead,
- loading embeddings from `.npy`,
- saving coordinates to CSV,
- saving benchmark CSV files.

It does include:

- reducer construction,
- the actual reduction call (`fit_transform` or equivalent),
- and, for UMAP / PaCMAP / FIt-SNE, library imports performed inside the
  timed reduction call.

Therefore the benchmark should be interpreted as:

> single-run wall-clock time of the reduction step in a fresh Python job.

It is not a fully isolated algorithmic microbenchmark. This matters especially
for methods with one-time setup, import, JIT, FAISS, or heuristic overheads.

This limitation is exactly why `main_run_2` was prepared. The second run keeps
the same data and methods, but measures a cleaner reduction-only timing:

- embeddings are loaded before the timer,
- imports and reducer construction are excluded from the timer,
- a small warm-up run is executed before measurement,
- each `(method, n_samples)` pair is repeated three times,
- `time_seconds` stores the median runtime,
- all repeat timings are saved separately.

Expected `main_run_2` outputs:

- `results/main_run_2/results_all.csv`,
- `results/main_run_2/timings_<method>_<n_samples>.csv`,
- `results/main_run_2/coordinates/coords_<method>_<n_samples>.csv`,
- `plots/main_run_2/time_by_method.png`,
- `plots/main_run_2/time_by_method_log.png`,
- `plots/main_run_2/coordinates/embedding_<method>_50000_clusters.png`.

## Runtime Interpretation

The most visible result is that **PCA is several orders of magnitude faster**
than the other methods. On the linear plot PCA appears almost equal to zero, but
it is not zero. Its runtime is on the order of hundredths of a second, while
UMAP and FIt-SNE take tens of seconds.

For presentation, a log-scale runtime plot is recommended:

```bash
python scripts/04_plot_results.py --yscale log --output plots/main_run_1/time_by_method_log.png
```

PCA is fast because reducing `384D -> 2D` relies on highly optimized linear
algebra. UMAP, PaCMAP and FIt-SNE are more complex: they build neighborhood
relations or graphs and perform iterative optimization.

### Interesting Observation: PaCMAP

PaCMAP was much faster than UMAP and FIt-SNE for 50k reviews:

```text
PaCMAP  50000: 21.7893 s
UMAP    50000: 90.4903 s
FIt-SNE 50000: 87.8828 s
```

However, PaCMAP was not monotonic:

```text
10000: 24.2231 s
25000: 11.0863 s
50000: 21.7893 s
```

This is counterintuitive and should not be overinterpreted. Possible reasons:

- single-run measurement,
- method-specific heuristics,
- initialization overhead,
- import/JIT/cache effects,
- different cluster nodes or transient system load,
- the fact that setup/import time is not separated from fit time.

For a more rigorous benchmark, each `(method, n_samples)` pair should be run
multiple times and summarized using median and variance. Imports and setup time
should also be separated from the actual fitting time.

## 2D Coordinates: Main Visualization Output

The `coords_*.csv` files are the most important output for the visualization
project. They contain the expensive dimensionality reduction result and can be
used later without running HPC jobs again.

Example schema:

```text
method,n_samples,point_id,x,y,rating,parent_asin
umap,50000,123,1.27,-3.88,5.0,B08...
```

Column meaning:

- `method` - reduction method,
- `n_samples` - sample size,
- `point_id` - review index in the sampled dataset,
- `x`, `y` - 2D coordinates after dimensionality reduction,
- `rating` - review rating,
- `parent_asin` - product identifier.

These files can be used to:

- redesign plots without recomputing reductions,
- recolor points by rating, cluster, sentiment, or any joined metadata,
- join points back to review text using `point_id`,
- build interactive Plotly, Dash, Tableau, or Orange visualizations,
- analyze review segments.

## 2D Visualizations

The most visually useful methods for 50,000 reviews were PaCMAP and UMAP.

PaCMAP 50k, colored by rating:

![PaCMAP 50k](plots/main_run_1/coordinates/embedding_pacmap_50000.png)

UMAP 50k, colored by rating:

![UMAP 50k](plots/main_run_1/coordinates/embedding_umap_50000.png)

PaCMAP 50k, colored by HDBSCAN clusters:

![PaCMAP clusters](plots/main_run_1/coordinates/embedding_pacmap_50000_clusters.png)

UMAP 50k, colored by HDBSCAN clusters:

![UMAP clusters](plots/main_run_1/coordinates/embedding_umap_50000_clusters.png)

## Visual Interpretation

The 2D maps show visible local structures in the review embedding space. PaCMAP
appears especially readable, mainly in the cluster-colored version. UMAP also
forms visible local groups, although the projection is more compact.

Rating-based coloring is less visually contrasted because the dataset is
strongly dominated by 5-star reviews.

Rating distribution for the 50k sample:

| Rating | Number of reviews |
|---:|---:|
| 1.0 | 4144 |
| 2.0 | 2116 |
| 3.0 | 3412 |
| 4.0 | 7381 |
| 5.0 | 32947 |

This explains why rating-colored plots are mostly yellow. This is a dataset
property, not a plotting error. For presenting opinion segments, cluster-colored
plots are more readable.

## Relationship to the Visualization Project

This experiment can be treated as the computational backend for the project:

> Customer opinion analysis and segmentation of products/opinions.

It satisfies the relevant requirements:

- review data preparation,
- work on a large Amazon Reviews dataset,
- basic text preprocessing by filtering empty reviews,
- text embedding preparation,
- comparison of PCA, UMAP, PaCMAP and FIt-SNE,
- runtime benchmark plots,
- 2D opinion maps,
- reusable data for cluster and segment analysis.

The most important bridge between LSC and visualization is:

```text
results/<run_name>/coordinates/coords_<method>_<n_samples>.csv
```

These files are a ready-to-use visualization dataset.

## CPU-Hours

Maximum allocation based on current SLURM limits:

```text
prepare embeddings: 1 job * 8 CPU * 2 h = 16 CPU-hours
reduction array:   16 jobs * 8 CPU * 2 h = 256 CPU-hours
postprocess:        1 job * 4 CPU * 0.5 h = 2 CPU-hours
--------------------------------------------------------
maximum total: 274 CPU-hours
```

The actual run was much shorter. From `sacct`:

- prepare embeddings: `00:03:14` on 8 CPUs,
- reduction array tasks: from a few seconds to about `00:01:36`,
- postprocess: `00:00:37` on 4 CPUs.

The actual CPU-hour usage of this run was roughly around 2 CPU-hours, but for
resource planning it is safer to use the SLURM allocation limit of about 274
CPU-hours per full run. A grant request around 3000 CPU-hours gives enough room
for debugging, reruns and additional variants.

For `main_run_2`, the reduction array wall-time limit is increased to 3 hours
because each task performs warm-up plus three timed repeats:

```text
reduction-only array: 16 jobs * 8 CPU * 3 h = 384 CPU-hours
postprocess:           1 job * 4 CPU * 1 h = 4 CPU-hours
----------------------------------------------------------
maximum total: 388 CPU-hours
```

This is a worst-case allocation bound. The expected actual usage should be much
lower, but this run is intentionally more rigorous than `main_run_1`.

## Limitations

1. **Main run 1 is a single-run benchmark**
   Each method/sample-size pair was run once. Results show practical runtime for
   one run, but do not estimate variance.

2. **Main run 1 timing includes some method setup/import overhead**
   Since imports for UMAP, PaCMAP and FIt-SNE happen inside the timed function,
   the runtime is not a perfectly isolated fit-only measurement.

3. **UMAP with `random_state=42`**  
   Logs show that setting `random_state` limits UMAP parallelism. The result is
   more deterministic, but may not show maximum possible multi-CPU throughput.

4. **No product metadata**  
   By design, only review data was used. `parent_asin` is available, but product
   names, brands and categories are not included.

5. **Dominance of 5-star reviews**  
   Rating-based interpretation requires caution because most reviews have
   rating 5.0.

## Main Conclusions

1. The pipeline works end-to-end on Ares: embedding preparation, reductions,
   CSV export, result merging and plot generation.

2. PCA is extremely fast, but it is not necessarily the most useful method for
   visualizing review structure.

3. UMAP and FIt-SNE are the most expensive methods at 50k reviews, taking around
   90 seconds for the reduction step.

4. PaCMAP appears to be the best compromise in this run: readable 2D maps and
   much lower runtime than UMAP/FIt-SNE at 50k reviews.

5. The `coords_*.csv` files are the main output for visualization. They allow
   new plots and analyses without recomputing reductions on the cluster.

6. Recommended figures for report/presentation:

   - `plots/main_run_1/time_by_method.png`,
   - `plots/main_run_1/time_by_method_log.png`,
   - `plots/main_run_1/coordinates/embedding_pacmap_50000.png`,
   - `plots/main_run_1/coordinates/embedding_pacmap_50000_clusters.png`,
   - optionally `plots/main_run_1/coordinates/embedding_umap_50000.png`.
