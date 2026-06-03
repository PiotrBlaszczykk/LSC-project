# Report v2: Clean Benchmark of Dimensionality Reduction Methods

## Project Goal

The goal of this experiment was to compare dimensionality reduction methods for
text review embeddings and produce reusable 2D coordinates for later
visualization and qualitative analysis.

The pipeline was:

```text
Amazon Reviews 2023, raw_review_Electronics
-> review text
-> SentenceTransformer embeddings, 384D
-> PCA / UMAP / PaCMAP / FIt-SNE
-> 2D coordinates
-> runtime benchmark + plots + cluster inspection
```

The experiment focuses on review/opinion segmentation, not product metadata
analysis. Product IDs are still preserved through `parent_asin`, so the 2D
points can later be joined with product metadata if needed.

## Data

Dataset:

- `McAuley-Lab/Amazon-Reviews-2023`
- subset: `raw_review_Electronics`

Fields used:

- `text` - input review text,
- `rating` - review score, used for coloring and interpretation,
- `parent_asin` - product ID, retained as an optional metadata join key.

The embedding stage produced:

```text
data/reviews_50000.csv
data/embeddings_50000.npy
```

The embedding matrix has shape:

```text
50000 reviews x 384 embedding dimensions
```

Only the 384-dimensional text embeddings are used as input to dimensionality
reduction.

## Main Run 2 Methodology

`main_run_2` was prepared to measure the dimensionality reduction step more
cleanly than the first run.

Compared with `main_run_1`, the second run:

- reuses the already prepared embeddings,
- loads data before the timer,
- excludes CSV saving from the timer,
- excludes reducer construction from the timer,
- performs a warm-up run on 1,000 samples,
- runs each `(method, n_samples)` pair three times,
- stores the median runtime as `time_seconds`,
- stores all repeat timings in `timings_<method>_<n_samples>.csv`.

The default benchmark settings were:

```text
timing_mode: fit-only
repeats: 3
warmup_samples: 1000
random_state: none
```

`random_state=none` was used intentionally for the performance benchmark. In
particular, it allows UMAP to use multiple workers instead of forcing
single-threaded deterministic execution.

The generated outputs are:

- `results/main_run_2/results_all.csv`,
- `results/main_run_2/benchmark_<method>_<n_samples>.csv`,
- `results/main_run_2/timings_<method>_<n_samples>.csv`,
- `results/main_run_2/coordinates/coords_<method>_<n_samples>.csv`,
- `plots/main_run_2/time_by_method.png`,
- `plots/main_run_2/time_by_method_log.png`,
- `plots/main_run_2/coordinates/*.png`.

Artifact completeness:

```text
benchmark CSV files: 16
timings CSV files: 16
coordinate CSV files: 16
coordinate plots: 20
```

No hard errors such as `ERROR`, `Traceback`, `FAILED`, `Killed`, or
`OutOfMemory` were found in the `main_run_2` logs.

## Runtime Results

Median reduction-only runtime from `results/main_run_2/results_all.csv`:

| Method | 5,000 | 10,000 | 25,000 | 50,000 |
|---|---:|---:|---:|---:|
| PCA | 0.0179 s | 0.0280 s | 0.0332 s | 0.0526 s |
| UMAP | 2.2794 s | 4.4885 s | 5.4189 s | 19.4880 s |
| PaCMAP | 2.1473 s | 3.7812 s | 7.6758 s | 20.5185 s |
| FIt-SNE | 40.1747 s | 43.4965 s | 62.0709 s | 78.3921 s |

Linear runtime plot:

![Runtime plot](plots/main_run_2/time_by_method.png)

Log-scale runtime plot:

![Runtime log plot](plots/main_run_2/time_by_method_log.png)

## Scaling Analysis

The second run shows much more regular scaling than the first run. The most
important improvement is that PaCMAP and UMAP now grow in a mostly monotonic and
approximately linear way.

Approximate scaling from 5k to 50k reviews:

| Method | 50k / 5k runtime ratio | Approx. exponent | Linear fit R2 | Slope |
|---|---:|---:|---:|---:|
| PCA | 2.94x | 0.47 | 0.961 | 0.0007 s / 1k reviews |
| UMAP | 8.55x | 0.93 | 0.914 | 0.3702 s / 1k reviews |
| PaCMAP | 9.56x | 0.98 | 0.977 | 0.4070 s / 1k reviews |
| FIt-SNE | 1.95x | 0.29 | 0.978 | 0.8687 s / 1k reviews |

The exponent is estimated from:

```text
time(50000) / time(5000) = 10^exponent
```

This is only a practical estimate for this sample range, not a theoretical
complexity proof.

### PCA

PCA is by far the fastest method. It reduces a fixed-width matrix:

```text
n_reviews x 384 dimensions -> n_reviews x 2 dimensions
```

Because the original embedding dimension is fixed at 384 and the target
dimension is fixed at 2, PCA mainly relies on optimized linear algebra. The
runtime is so small that it is almost invisible on the linear plot.

The log-scale plot is necessary to show PCA properly.

### UMAP

UMAP became much faster in `main_run_2` than in `main_run_1`:

```text
main_run_1 UMAP 50k: 90.4903 s
main_run_2 UMAP 50k: 19.4880 s
```

The main reason is that `main_run_2` uses `random_state=none`, which lets UMAP
use multiple workers. In `main_run_1`, `random_state=42` caused UMAP to override
parallel execution and behave like a single-threaded run.

UMAP runtime is now much closer to PaCMAP. It grows from 2.2794 s at 5k reviews
to 19.4880 s at 50k reviews. This is close to linear scaling in this range.

### PaCMAP

PaCMAP is the cleanest scaling result in this run:

```text
5k:   2.1473 s
10k:  3.7812 s
25k:  7.6758 s
50k: 20.5185 s
```

The 50k / 5k ratio is 9.56x for a 10x increase in sample size, so the observed
scaling is almost linear. This fixes the strange non-monotonic behavior seen in
the first run, where setup effects and single-run noise distorted the result.

PaCMAP is also visually useful: the 50k projection produces readable local
structures and interpretable clusters.

### FIt-SNE

FIt-SNE remains the most expensive method in absolute time:

```text
5k:  40.1747 s
50k: 78.3921 s
```

Its curve is almost straight on the plot, but it starts from a high baseline.
This suggests a large fixed cost from initialization and iterative
optimization. The method uses FFT-accelerated optimization, so within this
sample range the runtime increase is not as steep as a naive t-SNE
implementation would suggest.

For this project, FIt-SNE is useful as a stronger nonlinear baseline, but it is
less attractive for quick iterative visualization work.

## Repeat-Level Observations

The separate `timings_*.csv` files show why median runtime is a better summary
than a single run.

For UMAP, the first full-size repeat was often much slower:

```text
UMAP 5k:   16.5244, 2.2473, 2.2794
UMAP 10k:  18.8872, 4.4872, 4.4885
UMAP 25k:  20.4992, 5.4189, 5.3958
UMAP 50k:  31.0393, 19.4880, 18.4053
```

PCA also had first-repeat overhead at small sizes:

```text
PCA 5k:  0.1470, 0.0175, 0.0179
PCA 10k: 0.5952, 0.0265, 0.0280
```

This means the 1,000-sample warm-up removed import/JIT setup, but not every
full-size memory allocation and threadpool effect. Using the median of three
repeats makes the reported runtime more robust.

## Comparison with Main Run 1

`main_run_1` measured a more practical fresh-job runtime. It included more
method setup effects and used deterministic `random_state=42`.

`main_run_2` is a cleaner reduction-only benchmark.

Selected comparison:

| Method | n_samples | main_run_1 | main_run_2 | Difference |
|---|---:|---:|---:|---:|
| UMAP | 50,000 | 90.4903 s | 19.4880 s | 4.64x faster |
| PaCMAP | 10,000 | 24.2231 s | 3.7812 s | 6.41x faster |
| PaCMAP | 25,000 | 11.0863 s | 7.6758 s | 1.44x faster |
| FIt-SNE | 50,000 | 87.8828 s | 78.3921 s | 1.12x faster |
| PCA | 50,000 | 0.0595 s | 0.0526 s | similar |

The first run was still useful because it represented a simple HPC workflow.
The second run is better for reporting algorithm runtime.

## 2D Coordinates as Reusable Visualization Data

Each coordinates file contains one point per review:

```text
method,n_samples,point_id,x,y,rating,parent_asin
```

The most important files for visualization are:

```text
results/main_run_2/coordinates/coords_pacmap_50000.csv
results/main_run_2/coordinates/coords_umap_50000.csv
```

These files can be reused without rerunning the cluster jobs. They can be
recolored by rating, cluster, sentiment, product category, or metadata joined by
`parent_asin`.

## PaCMAP Cluster Inspection

The PaCMAP 50k projection was also inspected qualitatively. HDBSCAN was applied
to a 20,000-point sample of the 2D PaCMAP coordinates using the same plotting
logic as the generated cluster figure.

Cluster IDs are arbitrary labels. `-1` means noise, i.e. points that HDBSCAN did
not assign to any dense cluster.

PaCMAP 50k, cluster-colored:

![PaCMAP clusters](plots/main_run_2/coordinates/embedding_pacmap_50000_clusters.png)

### Cluster 11: Echo / Alexa smart displays

This cluster contains reviews about Echo devices, Alexa, smart displays, wake
words and home audio.

| point_id | rating | short excerpt |
|---:|---:|---|
| 48931 | 4.0 | "Echo Shows ... wake word ... Alexa" |
| 12400 | 5.0 | "echo show 8 ... kitchen size" |
| 376 | 5.0 | "Alexas ... better sound quality" |

### Cluster 50: laptop bags and backpacks

This cluster contains reviews about bags, pockets, laptop sleeves, carry-on use
and storage space.

| point_id | rating | short excerpt |
|---:|---:|---|
| 4068 | 5.0 | "loads of pockets for all the computer wires" |
| 23879 | 5.0 | "MacBook Pro ... water bottle ... chargers" |
| 17235 | 5.0 | "water bottle area ... lap top sleeve" |

### Cluster 48: external drives and storage enclosures

This cluster contains reviews about drives, USB storage, enclosures and laptop
backup or disk accessories.

| point_id | rating | short excerpt |
|---:|---:|---|
| 28103 | 5.0 | "periodic backups of my Mac laptop" |
| 30091 | 2.0 | "drive housing ... portable drive" |
| 7379 | 5.0 | "place to put an mSATA drive" |

### Cluster 13: charging cables and power adapters

Cluster 13 is an additional detailed example. It contains reviews about iPhone
charging cords, USB-C cables, power adapters, cable durability and charging
compatibility.

Selected full review texts from four different `parent_asin` values:

| point_id | parent_asin | rating | full review text |
|---:|---|---:|---|
| 15922 | `B07MPSFJSZ` | 5.0 | I wanted a longer, stronger iPhone charging cord and this one is perfect. |
| 3751 | `B0BZ15FM42` | 1.0 | Worked great for 3 days. Can someone please explain why none of these charging cables last for Iphone I have to replace constantly. Will be returning this one. I was very hopeful when I first received. Now it is like all others have to turn twist move around to get cord to work stupid. |
| 25124 | `B0BNC1QBH6` | 4.0 | Bought for my daughter. Took a bit to connect, but now connects with ease. Lasts for days. Seem durable. Only annoyed with the USB-C charging cable. I'm so tired of having 13 different charging cables! Can't we all just pick one and stick with it!!! |
| 5174 | `B07RN46PWL` | 5.0 | My dog chewed the cord off my first one so I'm on my second one now. I didn't hesitate to get another one based on my good experience with the first one. I could buy 4 or 5 of these power adapters for the price of the oem replacement. I'll buy a third one if my dog ruins number two! |

This cluster is a useful example because the reviews are not identical, but
they clearly share the same practical topic: charging accessories. Some reviews
are positive and some are negative, yet they still land in the same semantic
neighborhood because the text discusses similar objects and usage problems.

These examples show a meaningful semantic correlation between text content and
2D location. The reduction methods were not given product categories. They only
received text embeddings. Nevertheless, reviews about similar product types and
usage contexts appear in the same local neighborhoods.

This is useful for visualization: after joining metadata by `parent_asin`, it
should be possible to check whether product categories align with visible 2D
regions.

## Main Conclusions

1. `main_run_2` is the better benchmark for reduction runtime.

   It measures fit-only runtime, uses warm-up, repeats each task three times and
   reports the median.

2. PCA is extremely fast.

   It is useful as a linear baseline, but too compressed to be the most
   interesting visualization method.

3. UMAP and PaCMAP are now similar in runtime at 50k reviews.

   UMAP took 19.4880 s and PaCMAP took 20.5185 s. This is much more believable
   than the first run, where UMAP was slowed down by deterministic
   single-threaded execution.

4. PaCMAP has the cleanest observed scaling.

   Its 50k / 5k ratio is 9.56x for a 10x increase in sample size, which is close
   to linear in this experiment.

5. FIt-SNE is the slowest method, but still scales regularly.

   It has a large fixed cost and remains expensive even for 5k samples.

6. The 2D maps are semantically interpretable.

   PaCMAP clusters contain recognizable groups such as Echo/Alexa devices,
   laptop bags, external drive accessories and charging cables.

7. The project successfully supports both goals:

   - LSC: HPC execution, SLURM arrays and runtime benchmarking.
   - Visualization: reusable 2D coordinates, rating coloring, cluster maps and
     qualitative cluster interpretation.
