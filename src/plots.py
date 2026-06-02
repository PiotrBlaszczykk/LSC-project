import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure

import plotly.express as px
from plotly.graph_objects import Figure
import plotly.io as pio
import plotly.graph_objects as go

pio.templates["template"] = go.layout.Template(
    layout=dict(
        height=700,
        margin=dict(l=50, r=50, t=100, b=50),
    )
)

pio.templates.default = "template"

def rating_histogram(df: pd.DataFrame) -> Figure:
    fig = px.histogram(
        df,
        x="rating",
        nbins=5,
        title="Rating distribution",
        labels={"rating": "Rating", "count": "Count"}
    )
    fig.update_layout(bargap=0.1)
    return fig

def rating_histogram_mpl(df: pd.DataFrame) -> MplFigure:
    fig, ax = plt.subplots()
    df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Rating distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    return fig


def text_length_histogram(df: pd.DataFrame) -> Figure:
    return px.histogram(df, x="text_len", title="Review length distribution")


def text_length_histogram_mpl(df: pd.DataFrame) -> MplFigure:
    fig, ax = plt.subplots()
    ax.hist(df["text_len"], bins=50)
    ax.set_title("Review length distribution")
    ax.set_xlabel("text_len")
    return fig


def reviews_over_time(df: pd.DataFrame) -> Figure:
    counts = df.groupby(df["timestamp"].dt.year).size().reset_index(name="count")
    counts.columns = ["year", "count"]
    return px.bar(counts, x="year", y="count", title="Reviews over time")


def reviews_over_time_mpl(df: pd.DataFrame) -> MplFigure:
    counts = df.groupby(df["timestamp"].dt.year).size()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Reviews over time")
    ax.set_xlabel("year")
    ax.set_ylabel("count")
    return fig


def rating_over_time(df: pd.DataFrame) -> Figure:
    yearly = df.groupby(df["timestamp"].dt.year)["rating"].mean().reset_index()
    yearly.columns = ["year", "avg_rating"]
    fig = px.line(
        yearly, x="year", y="avg_rating", markers=True,
        title="Average rating over time",
        labels={"year": "Year", "avg_rating": "Average rating"},
    )
    # Interactive zoom on the time axis.
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def text_length_by_rating(df: pd.DataFrame) -> Figure:
    return px.violin(
        df, x="rating", y="text_len", box=True, points="outliers",
        title="Review length by rating",
        labels={"rating": "Rating", "text_len": "Review length"},
    )


def helpful_vs_length(df: pd.DataFrame) -> Figure:
    # Marginal distributions come "for free" alongside the scatter.
    return px.scatter(
        df, x="text_len", y="helpful_vote",
        marginal_x="histogram", marginal_y="box",
        opacity=0.5,
        title="Helpful votes vs review length",
        labels={"text_len": "Review length", "helpful_vote": "Helpful votes"},
    )


def seasonality_heatmap(df: pd.DataFrame) -> Figure:
    tmp = df.assign(year=df["timestamp"].dt.year, month=df["timestamp"].dt.month)
    fig = px.density_heatmap(
        tmp, x="month", y="year", nbinsx=12,
        title="Review seasonality (year x month)",
        labels={"month": "Month", "year": "Year"},
        color_continuous_scale="Viridis",
    )
    # Show every month tick (1-12) instead of plotly's sparse defaults.
    fig.update_xaxes(dtick=1)
    return fig


def reviews_per_product(df: pd.DataFrame) -> Figure:
    counts = df.groupby("parent_asin").size().reset_index(name="n_reviews")
    return px.histogram(
        counts, x="n_reviews", log_y=True,
        title="Reviews per product (long tail)",
        labels={"n_reviews": "Reviews per product"},
    )


def category_treemap(meta: pd.DataFrame) -> Figure:
    # Operates on product-level metadata (one row per product). main_category
    # is more informative than category_l1 (which is almost always "Electronics").
    data = meta.dropna(subset=["main_category"]).copy()
    data["category_l2"] = data["category_l2"].fillna("(other)")
    data["rating_number"] = data["rating_number"].fillna(0)
    return px.treemap(
        data,
        path=[px.Constant("all"), "main_category", "category_l2"],
        values="rating_number",
        title="Product categories by review volume",
    )


def price_vs_rating(meta: pd.DataFrame) -> Figure:
    data = meta.dropna(subset=["price", "average_rating", "rating_number"]).copy()
    # Trim extreme prices so the bubble chart stays readable.
    data = data[data["price"] <= data["price"].quantile(0.99)]
    return px.scatter(
        data, x="price", y="average_rating",
        size="rating_number", color="main_category",
        hover_name="title", opacity=0.6, size_max=40, log_x=True,
        title="Price vs average rating",
        labels={"price": "Price (log)", "average_rating": "Average rating"},
    )
