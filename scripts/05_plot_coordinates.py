from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reduced 2D review embeddings.")
    parser.add_argument("--method", choices=config.METHODS, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-by", choices=["rating", "cluster", "none"], default="rating")
    parser.add_argument("--min-cluster-size", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coordinates_path = (
        args.results_dir
        / "coordinates"
        / f"coords_{args.method}_{args.n_samples}.csv"
    )

    if not coordinates_path.exists():
        raise FileNotFoundError(f"Missing coordinates file: {coordinates_path}")

    coords = pd.read_csv(coordinates_path)
    if len(coords) > args.max_points:
        coords = coords.sample(args.max_points, random_state=args.seed)

    label = config.METHOD_LABELS.get(args.method, args.method)
    fig, ax = plt.subplots(figsize=(8, 6))

    if args.color_by == "cluster":
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
        labels = clusterer.fit_predict(coords[["x", "y"]])
        scatter = ax.scatter(
            coords["x"],
            coords["y"],
            c=labels,
            cmap="tab20",
            s=8,
            alpha=0.7,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label="Cluster (-1 = noise)")
    elif args.color_by == "rating" and "rating" in coords.columns and coords["rating"].notna().any():
        scatter = ax.scatter(
            coords["x"],
            coords["y"],
            c=coords["rating"],
            cmap="viridis",
            s=8,
            alpha=0.65,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label="Rating")
    else:
        ax.scatter(coords["x"], coords["y"], s=8, alpha=0.65, linewidths=0)

    ax.set_title(f"{label} projection of review embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)

    output_path = args.output
    if output_path is None:
        output_path = (
            ROOT
            / "plots"
            / "coordinates"
            / f"embedding_{args.method}_{args.n_samples}.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    print(f"Saved coordinates plot: {output_path}")


if __name__ == "__main__":
    main()
