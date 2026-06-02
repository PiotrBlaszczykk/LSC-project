from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge benchmark CSV files.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "results_all.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.results_dir.glob("benchmark_*.csv"))

    if not paths:
        raise FileNotFoundError(f"No benchmark_*.csv files found in {args.results_dir}")

    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    results = results.sort_values(["method", "n_samples"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(f"Merged {len(paths)} files into {args.output}")
    print(results)


if __name__ == "__main__":
    main()
