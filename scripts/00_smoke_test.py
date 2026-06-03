from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end smoke test.")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()

    data_dir = ROOT / "data" / "smoke"
    results_dir = ROOT / "results" / "smoke"
    plot_path = ROOT / "plots" / "smoke" / "time_by_method.png"
    coordinates_plots_dir = ROOT / "plots" / "smoke" / "coordinates"
    merged_path = results_dir / "results_all.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    coordinates_plots_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/01_prepare_embeddings.py",
            "--n-samples",
            str(args.n_samples),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(data_dir),
        ]
    )

    for method in ["pca", "umap", "pacmap", "fitsne"]:
        run(
            [
                sys.executable,
                "scripts/02_run_benchmark.py",
                "--method",
                method,
                "--n-samples",
                str(args.n_samples),
                "--data-dir",
                str(data_dir),
                "--results-dir",
                str(results_dir),
            ]
        )

    run(
        [
            sys.executable,
            "scripts/03_merge_results.py",
            "--results-dir",
            str(results_dir),
            "--output",
            str(merged_path),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/04_plot_results.py",
            "--input",
            str(merged_path),
            "--output",
            str(plot_path),
        ]
    )

    for method in ["pca", "umap", "pacmap", "fitsne"]:
        run(
            [
                sys.executable,
                "scripts/05_plot_coordinates.py",
                "--method",
                method,
                "--n-samples",
                str(args.n_samples),
                "--results-dir",
                str(results_dir),
                "--output",
                str(coordinates_plots_dir / f"embedding_{method}_{args.n_samples}.png"),
            ]
        )

    print("\nSmoke test finished.")
    print(f"Data: {data_dir}")
    print(f"Results: {results_dir}")
    print(f"Plot: {plot_path}")
    print(f"Coordinates plots: {coordinates_plots_dir}")


if __name__ == "__main__":
    main()
