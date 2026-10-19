"""Assign real player units to fitted styles and summarize per-style facts not in the features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture

from pokerbot.archetypes.features import count_matrix
from pokerbot.stats.hand_stats import DEPTH_EDGES


def mixture(fmt_profile: dict) -> GaussianMixture:
    styles = fmt_profile["styles"]
    g = GaussianMixture(len(styles), covariance_type="full")
    g.weights_ = np.array([s["share"] for s in styles])
    g.weights_ /= g.weights_.sum()
    g.means_ = np.stack([s["mean"] for s in styles])
    g.covariances_ = np.stack([s["cov"] for s in styles])
    g.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(g.covariances_))
    return g


def assign(fmt_profile: dict, stats: pl.DataFrame, fmt: str, min_hands: int = 500):
    """Returns (pids, membership probabilities q (units x K))."""
    t = fmt_profile["transform"]
    sub = stats.filter(pl.col("pid").str.ends_with(f":{fmt}"))
    pids, _, n, x = count_matrix(sub, t.features, min_hands)
    z, _ = t.transform(n, x)
    return pids, mixture(fmt_profile).predict_proba(z)


def depth_shares(fmt_profile: dict, stats: pl.DataFrame, fmt: str) -> np.ndarray:
    """Per style, the share of its members' hands in each effective-stack bucket
    (``DEPTH_EDGES``: <25, 25-60, 60-150, >150 bb), weighted by membership. Shape (K, 4).
    Hands with unrecorded stacks (iPoker) have no depth counts and don't contribute."""
    pids, q = assign(fmt_profile, stats, fmt)
    buckets = [f"vpip_d{i}" for i in range(len(DEPTH_EDGES) + 1)]
    depth = (
        stats.filter(pl.col("pid").is_in(pids) & pl.col("stat").is_in(buckets))
        .pivot(on="stat", index="pid", values="n")
        .fill_null(0)
    )
    depth = pl.DataFrame({"pid": pids}).join(depth, on="pid", how="left").fill_null(0)
    d = np.column_stack(
        [depth[b].to_numpy() if b in depth.columns else np.zeros(len(pids)) for b in buckets]
    ).astype(float)
    shares = q.T @ d  # (K, 4) membership-weighted hand counts
    return shares / np.maximum(shares.sum(axis=1, keepdims=True), 1e-9)


def format_stats(stats_path: Path, fmt: str, features: tuple[str, ...]) -> pl.DataFrame:
    """The rows of the per-player stats table that membership, depth and hand counts need for one
    format (the full table is several GB in memory; this is a small slice of it)."""
    keep = [*features, "hands", *(f"vpip_d{i}" for i in range(len(DEPTH_EDGES) + 1))]
    return (
        pl.scan_parquet(stats_path)
        .filter(pl.col("pid").str.ends_with(f":{fmt}") & pl.col("stat").is_in(keep))
        .select("pid", "stat", "n", "x")
        .collect()
    )


def hand_quantiles(stats: pl.DataFrame, fmt: str, min_hands: int = 500) -> np.ndarray:
    """Percentiles 0..100 of hands per profiled player (units with >= min_hands) in a format."""
    h = stats.filter(
        pl.col("pid").str.ends_with(f":{fmt}")
        & (pl.col("stat") == "hands")
        & (pl.col("n") >= min_hands)
    )["n"].to_numpy()
    return np.percentile(h, np.arange(101))
