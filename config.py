DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
SUBSET = "raw_review_Electronics"

# Kept for the older visualization notebooks. The LSC benchmark below uses
# reviews only and does not join product metadata.
META_SUBSET = "raw_meta_Electronics"

N_SAMPLES = 5_000

N_VALUES = [5_000, 10_000, 25_000, 50_000]
METHODS = ["pca", "umap", "pacmap", "fitsne"]
METHOD_LABELS = {
    "pca": "PCA",
    "umap": "UMAP",
    "pacmap": "PaCMAP",
    "fitsne": "FIt-SNE",
}

MAX_SAMPLES = max(N_VALUES)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
