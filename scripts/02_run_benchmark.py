from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one dimensionality reduction benchmark.")
    parser.add_argument("--method", choices=config.METHODS, required=True)
    parser.add_argument("--n-samples", type=int, choices=config.N_VALUES, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def load_embeddings(data_dir: Path, n_samples: int) -> np.ndarray:
    exact_path = data_dir / f"embeddings_{n_samples}.npy"
    max_path = data_dir / f"embeddings_{config.MAX_SAMPLES}.npy"

    if exact_path.exists():
        embeddings = np.load(exact_path)
    elif max_path.exists():
        embeddings = np.load(max_path)[:n_samples]
    else:
        raise FileNotFoundError(
            f"Missing embeddings file. Run scripts/01_prepare_embeddings.py first. "
            f"Expected {exact_path} or {max_path}."
        )

    if len(embeddings) < n_samples:
        raise ValueError(f"Only {len(embeddings)} embeddings available, expected {n_samples}")

    return embeddings[:n_samples].astype(np.float32, copy=False)


def n_jobs() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def reduce_embeddings(method: str, embeddings: np.ndarray) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings)

    if method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings)

    if method == "pacmap":
        import pacmap

        reducer = pacmap.PaCMAP(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings)

    if method == "fitsne":
        from openTSNE import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=500,
            initialization="pca",
            negative_gradient_method="fft",
            n_jobs=n_jobs(),
            random_state=42,
        )
        return np.asarray(reducer.fit(embeddings))

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(args.data_dir, args.n_samples)
    print(f"Running {args.method} for n_samples={args.n_samples}, dim={embeddings.shape[1]}")

    start = time.perf_counter()
    reduce_embeddings(args.method, embeddings)
    elapsed = time.perf_counter() - start

    row = {
        "method": args.method,
        "n_samples": args.n_samples,
        "time_seconds": round(elapsed, 4),
    }

    output_path = args.results_dir / f"benchmark_{args.method}_{args.n_samples}.csv"
    pd.DataFrame([row]).to_csv(output_path, index=False)

    print(f"Saved result: {output_path}")
    print(row)


if __name__ == "__main__":
    main()
