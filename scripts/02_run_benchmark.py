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
    parser.add_argument(
        "--timing-mode",
        choices=["end-to-end", "fit-only"],
        default="end-to-end",
        help=(
            "end-to-end keeps the original timing style. fit-only creates the reducer "
            "before the timer and measures only fit/fit_transform."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument(
        "--random-state",
        default="42",
        help="Integer seed or 'none'. Using 'none' lets UMAP use multiple workers.",
    )
    args = parser.parse_args()

    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_samples < 0:
        parser.error("--warmup-samples cannot be negative")

    return args


def parse_random_state(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("--random-state must be an integer or 'none'") from exc


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


def make_reducer(method: str, random_state: int | None):
    if method == "pca":
        return PCA(n_components=2, random_state=random_state)

    if method == "umap":
        import umap

        if random_state is None:
            return umap.UMAP(n_components=2, n_jobs=n_jobs())
        return umap.UMAP(n_components=2, random_state=random_state)

    if method == "pacmap":
        import pacmap

        return pacmap.PaCMAP(n_components=2, random_state=random_state)

    if method == "fitsne":
        from openTSNE import TSNE

        return TSNE(
            n_components=2,
            perplexity=30,
            n_iter=500,
            initialization="pca",
            negative_gradient_method="fft",
            n_jobs=n_jobs(),
            random_state=random_state,
        )

    raise ValueError(f"Unknown method: {method}")


def fit_reducer(method: str, reducer, embeddings: np.ndarray) -> np.ndarray:
    if method == "fitsne":
        return np.asarray(reducer.fit(embeddings))

    return reducer.fit_transform(embeddings)


def reduce_embeddings(
    method: str,
    embeddings: np.ndarray,
    random_state: int | None,
) -> np.ndarray:
    reducer = make_reducer(method, random_state)
    return fit_reducer(method, reducer, embeddings)


def run_warmup(
    method: str,
    embeddings: np.ndarray,
    warmup_samples: int,
    random_state: int | None,
) -> None:
    if warmup_samples <= 0:
        return

    sample_size = min(warmup_samples, len(embeddings))
    sample = embeddings[:sample_size]
    print(f"Warm-up {method}: n_samples={sample_size}, dim={sample.shape[1]}")
    reducer = make_reducer(method, random_state)
    fit_reducer(method, reducer, sample)


def run_timed_repeats(
    method: str,
    embeddings: np.ndarray,
    timing_mode: str,
    repeats: int,
    warmup_samples: int,
    random_state: int | None,
) -> tuple[np.ndarray, list[float]]:
    if timing_mode == "fit-only":
        run_warmup(method, embeddings, warmup_samples, random_state)

    coordinates = None
    times: list[float] = []

    for repeat in range(1, repeats + 1):
        if timing_mode == "fit-only":
            reducer = make_reducer(method, random_state)
            start = time.perf_counter()
            coordinates = fit_reducer(method, reducer, embeddings)
        else:
            start = time.perf_counter()
            coordinates = reduce_embeddings(method, embeddings, random_state)

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Repeat {repeat}/{repeats}: {elapsed:.4f}s")

    if coordinates is None:
        raise RuntimeError("No coordinates were produced")

    return coordinates, times


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
    random_state = parse_random_state(args.random_state)

    embeddings = load_embeddings(args.data_dir, args.n_samples)
    print(
        f"Running {args.method} for n_samples={args.n_samples}, dim={embeddings.shape[1]}, "
        f"timing_mode={args.timing_mode}, repeats={args.repeats}, "
        f"warmup_samples={args.warmup_samples}, random_state={args.random_state}"
    )

    coordinates, times = run_timed_repeats(
        args.method,
        embeddings,
        args.timing_mode,
        args.repeats,
        args.warmup_samples,
        random_state,
    )
    elapsed = float(np.median(times))

    row = {
        "method": args.method,
        "n_samples": args.n_samples,
        "time_seconds": round(elapsed, 4),
        "time_seconds_mean": round(float(np.mean(times)), 4),
        "time_seconds_std": round(float(np.std(times, ddof=1)) if len(times) > 1 else 0.0, 4),
        "time_seconds_min": round(float(np.min(times)), 4),
        "time_seconds_max": round(float(np.max(times)), 4),
        "repeats": args.repeats,
        "timing_mode": args.timing_mode,
        "warmup_samples": args.warmup_samples,
        "random_state": args.random_state,
    }

    output_path = args.results_dir / f"benchmark_{args.method}_{args.n_samples}.csv"
    pd.DataFrame([row]).to_csv(output_path, index=False)

    timings_path = args.results_dir / f"timings_{args.method}_{args.n_samples}.csv"
    pd.DataFrame(
        [
            {
                "method": args.method,
                "n_samples": args.n_samples,
                "repeat": repeat,
                "time_seconds": round(value, 4),
                "timing_mode": args.timing_mode,
                "warmup_samples": args.warmup_samples,
                "random_state": args.random_state,
            }
            for repeat, value in enumerate(times, start=1)
        ]
    ).to_csv(timings_path, index=False)

    coordinates_path = save_coordinates(
        args.method,
        args.n_samples,
        coordinates,
        args.data_dir,
        args.results_dir,
    )

    print(f"Saved result: {output_path}")
    print(f"Saved repeat timings: {timings_path}")
    print(f"Saved coordinates: {coordinates_path}")
    print(row)


if __name__ == "__main__":
    main()
