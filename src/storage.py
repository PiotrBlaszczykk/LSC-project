from pathlib import Path
from typing import Callable

import pandas as pd

# Stage outputs live in plotly-project/data/ as Parquet, so each notebook can
# read the previous stage instead of recomputing loading/cleaning/embeddings.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def stage_exists(name: str) -> bool:
    return _path(name).exists()


def save_stage(df: pd.DataFrame, name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = _path(name)
    df.to_parquet(path, index=False)
    return path


def load_stage(name: str) -> pd.DataFrame:
    return pd.read_parquet(_path(name))


def cached(name: str, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    # Return the cached stage if present, otherwise build it once and persist.
    if stage_exists(name):
        return load_stage(name)
    df = build()
    save_stage(df, name)
    return df
