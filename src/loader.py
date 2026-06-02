import pandas as pd
from datasets import load_dataset


def load_reviews(dataset_name: str, subset: str, n: int | None = None) -> pd.DataFrame:
    ds = load_dataset(dataset_name, subset, streaming=True, trust_remote_code=True)
    records = list(ds["full"].take(n)) if n else list(ds["full"])
    return pd.DataFrame(records)


def load_meta(dataset_name: str, subset: str, n: int | None = None) -> pd.DataFrame:
    # Meta subsets ship as Parquet in the repo. Read them directly (parquet
    # builder) instead of via the loading script: the script forces a schema
    # that fails to cast the nested fields (e.g. `images`) -> ArrowNotImplementedError.
    data_files = f"hf://datasets/{dataset_name}/{subset}/*.parquet"
    ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    records = list(ds.take(n)) if n else list(ds)
    return pd.DataFrame(records)
