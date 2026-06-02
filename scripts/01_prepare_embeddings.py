from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Amazon review texts and save SentenceTransformer embeddings."
    )
    parser.add_argument("--n-samples", type=int, default=config.MAX_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default=None, help="Optional device, for example cpu or cuda.")
    return parser.parse_args()


def load_review_sample(n_samples: int) -> pd.DataFrame:
    dataset = load_dataset(
        config.DATASET_NAME,
        config.SUBSET,
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    for row in dataset["full"]:
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        rows.append(
            {
                "text": text,
                "rating": row.get("rating"),
                "parent_asin": row.get("parent_asin"),
            }
        )

        if len(rows) >= n_samples:
            break

        if len(rows) % 10_000 == 0:
            print(f"Loaded {len(rows)} reviews")

    if len(rows) < n_samples:
        raise RuntimeError(f"Loaded only {len(rows)} non-empty reviews, expected {n_samples}")

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_samples} reviews from {config.SUBSET}")
    reviews = load_review_sample(args.n_samples)

    reviews_path = args.output_dir / f"reviews_{args.n_samples}.csv"
    reviews.to_csv(reviews_path, index=False)
    print(f"Saved review sample: {reviews_path}")

    print(f"Encoding texts with {args.model}")
    model = SentenceTransformer(args.model, device=args.device)

    start = time.perf_counter()
    embeddings = model.encode(
        reviews["text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - start

    embeddings_path = args.output_dir / f"embeddings_{args.n_samples}.npy"
    np.save(embeddings_path, embeddings)

    print(f"Saved embeddings: {embeddings_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Encoding time_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
