import pandas as pd


def data_quality(df: pd.DataFrame) -> None:
    print("--- Missing values ---")
    print(df.isnull().sum())

    print(f"\nEmpty strings (text): {(df['text'] == '').sum()}")
    print(f"Empty strings (title): {(df['title'] == '').sum()}")
    print(f"\nDuplicates (user_id + asin): {df.duplicated(subset=['user_id', 'asin']).sum()}")
    print(f"Duplicates (full row): {df.duplicated().sum()}")
    print(f"\nDate range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"Unique products (asin): {df['asin'].nunique()}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Verified purchase: {df['verified_purchase'].value_counts().to_dict()}")

    print("\n--- Rating distribution ---")
    print(df["rating"].value_counts().sort_index())


def category_summary(df: pd.DataFrame, top: int = 15) -> None:
    # Expects reviews merged with metadata (see preprocessing.merge_meta).
    print("--- Reviews by main_category ---")
    by_cat = (
        df.groupby("main_category")
        .agg(n_reviews=("rating", "size"), avg_rating=("rating", "mean"))
        .sort_values("n_reviews", ascending=False)
    )
    print(by_cat.head(top))

    if df["price"].notna().any():
        print("\n--- Average rating by price bucket ---")
        buckets = pd.qcut(df["price"], q=4, duplicates="drop")
        print(df.groupby(buckets, observed=True)["rating"].mean())
