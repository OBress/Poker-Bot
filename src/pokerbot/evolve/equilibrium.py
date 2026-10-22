"""An "equilibrium" pseudo-style: what the baseline itself looks like to a classifier.

Lesson from Track A: a classifier that only knows exploitable styles labels an equilibrium
player as one of them, and the exploiter then loses to the baseline. So the baseline gets a style
entry of its own, measured the same way real styles are: the baseline plays sessions of
``HANDS`` hands against bots drawn from the population, each session's counts go through the
profile's feature transform, and the entry stores the mean and covariance of those features
(spread = session-to-session variation, as real styles include measurement noise), pooled rates,
per-stat Beta fits and the shown-hand table. ``StyleProfiles.load(..., equilibrium=path)``
appends it with population share ``SHARE``.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pokerbot.archetypes.profiles import _weighted_beta, load_profiles
from pokerbot.eval.duplicate import make_policy
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.showdown import RangeTable
from pokerbot.opponents.simulate import simulate_stats

HANDS = 1000
SHARE = 0.15
NAME = "Equilibrium (blueprint)"


def _session(args):
    policy, bots_path, fmt, seed, hands = args
    rng = np.random.default_rng(seed)
    sampler = StyleSampler(Path(bots_path), fmt)
    hero = make_policy(policy, int(rng.integers(2**31)))
    opp = sampler.draw(rng).bot
    seats = [hero, opp] if seed % 2 == 0 else [opp, hero]
    hero_seat = seats.index(hero)
    table = RangeTable(2)
    style_of_seat = [0 if s == hero_seat else 1 for s in range(2)]
    counts = simulate_stats(seats, [f"s{i}" for i in range(2)], hands, int(rng.integers(2**31)),
                            on_hand=lambda ev: table.add_events(ev, style_of_seat))  # fmt: skip
    return counts[f"s{hero_seat}"], table.to_dict()


def measure(policy: str, profiles_path: Path, bots_path: Path, fmt: str = "hu",
            sessions: int = 200, workers: int = 8, seed: int = 0) -> dict:  # fmt: skip
    prof = load_profiles(profiles_path)["formats"][fmt]
    t = prof["transform"]
    jobs = [(policy, str(bots_path), fmt, seed * 100_000 + i, HANDS) for i in range(sessions)]
    with ProcessPoolExecutor(workers) as pool:
        out = list(pool.map(_session, jobs))
    n = np.array([[c.get(f, (0, 0))[0] for f in t.features] for c, _ in out], dtype=float)
    x = np.array([[c.get(f, (0, 0))[1] for f in t.features] for c, _ in out], dtype=float)
    z, _ = t.transform(n, x)
    mean = z.mean(axis=0)
    cov = np.cov(z, rowvar=False) + 1e-3 * np.eye(len(t.features))
    pooled = x.sum(axis=0) / np.maximum(n.sum(axis=0), 1)
    beta = {}
    for j, f in enumerate(t.features):
        ok = n[:, j] > 0
        rates = x[ok, j] / n[ok, j]
        beta[f] = (
            list(_weighted_beta(rates, np.ones(ok.sum()), n[ok, j])) if ok.sum() > 2 else [1, 1]
        )
    sd = RangeTable(2)
    for _, d in out:
        sd.merge(RangeTable.from_dict(d))
    entry = {"name": NAME, "share": SHARE, "policy": policy, "format": fmt, "sessions": sessions,
             "hands_per_session": HANDS, "mean": mean.tolist(), "cov": cov.tolist(),
             "pooled_rate": dict(zip(t.features, pooled.tolist(), strict=True)), "beta": beta,
             "showdown_counts": sd.counts[0].tolist()}  # fmt: skip
    entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
    return entry
