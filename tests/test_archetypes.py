"""Style discovery recovers known synthetic styles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from pokerbot.archetypes.features import FEATURES, count_matrix, fit_transform, is_holdout
from pokerbot.archetypes.fit import fit_styles
from pokerbot.archetypes.profiles import load_profiles

# Three styles that differ on looseness/aggression; all other stats shared.
TRUE = {
    "tight": {"vpip": 0.14, "pfr": 0.11, "wtsd": 0.22, "fold_to_cbet_flop": 0.6},
    "loose_passive": {"vpip": 0.48, "pfr": 0.07, "wtsd": 0.38, "fold_to_cbet_flop": 0.3},
    "aggro": {"vpip": 0.36, "pfr": 0.30, "wtsd": 0.27, "fold_to_cbet_flop": 0.45},
}
BASE = 0.3


def synthetic_stats(players_per_style: int = 700, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for style, rates in TRUE.items():
        for i in range(players_per_style):
            hands = int(rng.integers(600, 3000))
            pid = f"PS:{style}{i}:sh"
            rows.append((pid, "hands", hands, hands))
            for f in FEATURES:
                p = np.clip(rates.get(f, BASE) + rng.normal(0, 0.02), 0.01, 0.99)
                n = hands if f in ("vpip", "pfr") else int(hands * 0.2)
                rows.append((pid, f, n, int(rng.binomial(n, p))))
    return pl.DataFrame(rows, schema=["pid", "stat", "n", "x"], orient="row")


def test_transform_round_trips(tmp_path: Path) -> None:
    pids, hands, n, x = count_matrix(synthetic_stats(50), FEATURES, 500)
    t = fit_transform(n, x)
    z, s2 = t.transform(n, x)
    assert z.shape == (150, len(FEATURES)) and np.all(s2 > 0)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6)


def test_fit_recovers_three_styles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    stats_path = tmp_path / "stats.parquet"
    synthetic_stats().write_parquet(stats_path)
    out = fit_styles(stats_path, tmp_path / "styles.json", "test", ks=range(2, 6), boots=2)
    prof = load_profiles(out)
    sh = prof["formats"]["sh"]
    assert len(sh["styles"]) == 3
    vpips = sorted(s["pooled_rate"]["vpip"] for s in sh["styles"])
    assert np.allclose(vpips, [0.14, 0.36, 0.48], atol=0.02)
    assert abs(sum(s["share"] for s in sh["styles"]) - 1) < 1e-6
    assert (tmp_path / "experiments").exists()  # tracked run written


def test_holdout_is_deterministic() -> None:
    a = [is_holdout(f"PS:p{i}:sh") for i in range(5000)]
    assert a == [is_holdout(f"PS:p{i}:sh") for i in range(5000)]
    assert 0.17 < np.mean(a) < 0.23
