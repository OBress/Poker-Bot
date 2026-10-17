"""Player feature vectors for style discovery (docs/MATH.md §3.1).

From per-player counts (n opportunities, x occurrences) to standardized logit features:
  r̂ = (x + a) / (n + a + b)          shrink toward a population Beta(a, b) prior
  z = logit(r̂),  s² ≈ 1/((n+a+b) r̂(1-r̂))
  z' = (z - mean) / std               standardized across fitting players
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl

# Stats used as clustering features. Chosen to cover preflop looseness/aggression, positional
# awareness (EP vs BTN opening), response to aggression, postflop aggression and showdown tendency.
FEATURES: tuple[str, ...] = (
    "vpip",
    "pfr",
    "limp",
    "rfi_EP",
    "rfi_BTN",
    "3bet",
    "fold_to_3bet",
    "fold_to_steal",
    "cbet_flop",
    "cbet_turn",
    "fold_to_cbet_flop",
    "fold_to_cbet_turn",
    "xr_flop",
    "donk_flop",
    "agg_flop",
    "agg_turn",
    "agg_river",
    "wtsd",
    "wsd",
)


@dataclass(frozen=True)
class FeatureTransform:
    """Everything needed to map a player's counts into feature space (saved in profiles)."""

    features: tuple[str, ...]
    prior_a: np.ndarray  # Beta prior per feature
    prior_b: np.ndarray
    mean: np.ndarray  # standardization of logit features
    std: np.ndarray

    def transform(self, n: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Counts (players × D) -> standardized features z and their noise variances s²."""
        a, b = self.prior_a, self.prior_b
        r = (x + a) / (n + a + b)
        z = np.log(r / (1 - r))
        s2 = 1.0 / ((n + a + b) * r * (1 - r))
        return (z - self.mean) / self.std, s2 / self.std**2

    def to_dict(self) -> dict:
        return {
            "features": list(self.features),
            "prior_a": self.prior_a.tolist(),
            "prior_b": self.prior_b.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> FeatureTransform:
        return FeatureTransform(
            tuple(d["features"]),
            np.array(d["prior_a"]),
            np.array(d["prior_b"]),
            np.array(d["mean"]),
            np.array(d["std"]),
        )


def count_matrix(
    stats: pl.DataFrame, features: tuple[str, ...] = FEATURES, min_hands: int = 500
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Players with >= min_hands. Returns (pids, hands, n, x) with n, x shaped (players, D)."""
    hands = stats.filter((pl.col("stat") == "hands") & (pl.col("n") >= min_hands)).select(
        "pid", pl.col("n").alias("hands")
    )
    wide = (
        stats.filter(pl.col("stat").is_in(list(features)))
        .join(hands, on="pid", how="semi")
        .pivot(on="stat", index="pid", values=["n", "x"])
    )
    wide = hands.join(wide, on="pid", how="left").sort("pid")

    def column(name: str) -> np.ndarray:
        if name not in wide.columns:
            return np.zeros(wide.height)
        return wide[name].fill_null(0).to_numpy()

    n = np.column_stack([column(f"n_{f}") for f in features]).astype(float)
    x = np.column_stack([column(f"x_{f}") for f in features]).astype(float)
    return wide["pid"].to_list(), wide["hands"].to_numpy(), n, x


def fit_beta_prior(
    n: np.ndarray, x: np.ndarray, min_opps: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Method-of-moments Beta(a, b) per feature from players' raw rates, correcting the observed
    variance for binomial sampling noise. Players with < min_opps opportunities are ignored."""
    a = np.empty(n.shape[1])
    b = np.empty(n.shape[1])
    for j in range(n.shape[1]):
        ok = n[:, j] >= min_opps
        rates = x[ok, j] / n[ok, j]
        m = rates.mean()
        noise = np.mean(m * (1 - m) / n[ok, j])
        var = max(rates.var() - noise, 1e-4)
        common = m * (1 - m) / var - 1
        common = max(common, 2.0)  # weakest sensible prior: a + b >= 2
        a[j] = m * common
        b[j] = (1 - m) * common
    return a, b


def fit_transform(
    n: np.ndarray, x: np.ndarray, features: tuple[str, ...] = FEATURES
) -> FeatureTransform:
    a, b = fit_beta_prior(n, x)
    r = (x + a) / (n + a + b)
    z = np.log(r / (1 - r))
    return FeatureTransform(features, a, b, z.mean(axis=0), z.std(axis=0))


def is_holdout(pid: str, fraction: float = 0.2) -> bool:
    """Deterministic player split: held-out players are never used to fit styles."""
    h = int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], "big")
    return (h / 2**64) < fraction
