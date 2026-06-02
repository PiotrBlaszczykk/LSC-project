import numpy as np
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print('Cleaning dataframe from redundant columns and empty values')

    df = df.drop(columns=["images"])
    df = df.dropna(subset=["text", "rating"])
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print('Adding text_len, title_len and timestamp columns')

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["text_len"] = df["text"].str.len()
    df["title_len"] = df["title"].str.len()
    return df


def clean_meta(meta: pd.DataFrame) -> pd.DataFrame:
    print('Cleaning metadata: numeric price and category levels')

    meta = meta.copy()
    # price comes as a string ("19.99" or "None") -> coerce to float.
    meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
    # categories goes from broad to specific; it loads as np.ndarray from
    # Parquet, so normalise to a list before pulling the top two levels.
    cats = meta["categories"].apply(
        lambda c: list(c) if isinstance(c, (list, tuple, np.ndarray)) else []
    )
    meta["category_l1"] = cats.apply(lambda c: c[0] if len(c) > 0 else None)
    meta["category_l2"] = cats.apply(lambda c: c[1] if len(c) > 1 else None)
    return meta


def merge_meta(reviews: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    print('Merging reviews with product metadata on parent_asin')

    meta_cols = [
        "parent_asin", "main_category", "category_l1", "category_l2",
        "price", "average_rating", "rating_number", "store",
    ]
    cols = [c for c in meta_cols if c in meta.columns]
    return reviews.merge(meta[cols], on="parent_asin", how="left")