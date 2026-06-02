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
    parser = argparse.ArgumentParser(description="Plot benchmark runtime by sample size.")
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "results_all.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "plots" / "time_by_method.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.input)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in config.METHODS:
        data = results[results["method"] == method].sort_values("n_samples")
        if data.empty:
            continue
        label = config.METHOD_LABELS.get(method, method)
        ax.plot(data["n_samples"], data["time_seconds"], marker="o", label=label)

    ax.set_title("Dimensionality reduction runtime")
    ax.set_xlabel("Number of reviews")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
