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
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")

    return args


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


def load_review_info(data_dir: Path, n_samples: int) -> pd.DataFrame:
    exact_path = data_dir / f"reviews_{n_samples}.csv"
    max_path = data_dir / f"reviews_{config.MAX_SAMPLES}.csv"

    if exact_path.exists():
        reviews = pd.read_csv(exact_path)
    elif max_path.exists():
        reviews = pd.read_csv(max_path).head(n_samples)
    else:
        reviews = pd.DataFrame(index=range(n_samples))

    return reviews.head(n_samples).reset_index(drop=True)


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


def save_coordinates(
    method: str,
    n_samples: int,
    coordinates: np.ndarray,
    data_dir: Path,
    results_dir: Path,
) -> Path:
    reviews = load_review_info(data_dir, n_samples)
    coordinates_dir = results_dir / "coordinates"
    coordinates_dir.mkdir(parents=True, exist_ok=True)

    coords = pd.DataFrame(
        {
            "method": method,
            "n_samples": n_samples,
            "point_id": np.arange(len(coordinates)),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )

    for col in ["rating", "parent_asin"]:
        if col in reviews.columns:
            coords[col] = reviews[col].values

    output_path = coordinates_dir / f"coords_{method}_{n_samples}.csv"
    coords.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(args.data_dir, args.n_samples)
    print(f"Running {args.method} for n_samples={args.n_samples}, dim={embeddings.shape[1]}")

    start = time.perf_counter()
    coordinates = reduce_embeddings(args.method, embeddings)
    elapsed = time.perf_counter() - start

    row = {
        "method": args.method,
        "n_samples": args.n_samples,
        "time_seconds": round(elapsed, 4),
    }

    output_path = args.results_dir / f"benchmark_{args.method}_{args.n_samples}.csv"
    pd.DataFrame([row]).to_csv(output_path, index=False)

    coordinates_path = save_coordinates(
        args.method,
        args.n_samples,
        coordinates,
        args.data_dir,
        args.results_dir,
    )

    print(f"Saved result: {output_path}")
    print(f"Saved coordinates: {coordinates_path}")
    print(row)


if __name__ == "__main__":
    main()
