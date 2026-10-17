"""How much data is enough? Players with enough hands to profile, as data accumulates.

A player is identified by (site, obfuscated name): names are hashed per site, so the same string on
two sites is not assumed to be the same person.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

THRESHOLDS = (100, 500, 1000, 5000)


def player_hands(parquet_dir: Path) -> pl.LazyFrame:
    """One row per (hand, player) with the hand's date."""
    return (
        pl.scan_parquet(parquet_dir / "*.parquet")
        .select("site", "stake", "date", "players")
        .explode("players")
        .rename({"players": "player"})
        .with_columns(pl.concat_str("site", pl.lit(":"), "player").alias("pid"))
    )


def coverage_by_day(parquet_dir: Path) -> pl.DataFrame:
    """For data up to and including each day: hands, players, and players with >= T hands."""
    per_day = player_hands(parquet_dir).group_by("pid", "date").agg(pl.len().alias("n")).collect()
    hands_per_day = (
        pl.scan_parquet(parquet_dir / "*.parquet").group_by("date").agg(pl.len().alias("hands"))
    ).collect()
    rows = []
    for day in sorted(per_day["date"].unique().to_list()):
        upto = per_day.filter(pl.col("date") <= day).group_by("pid").agg(pl.col("n").sum())
        row = {
            "through": day,
            "hands": int(hands_per_day.filter(pl.col("date") <= day)["hands"].sum()),
            "players": upto.height,
        }
        for t in THRESHOLDS:
            row[f">={t}"] = int((upto["n"] >= t).sum())
        rows.append(row)
    return pl.DataFrame(rows)


def breakdown(parquet_dir: Path, min_hands: int = 500) -> pl.DataFrame:
    """Hands and qualified players per site and stake."""
    ph = player_hands(parquet_dir)
    counts = ph.group_by("site", "stake", "pid").agg(pl.len().alias("n"))
    return (
        counts.group_by("site", "stake")
        .agg(
            pl.len().alias("players"),
            (pl.col("n") >= min_hands).sum().alias(f"players>={min_hands}"),
            pl.col("n").sum().alias("player_hands"),
        )
        .sort("site", "stake")
        .collect()
    )
