"""Duplicate matches against sampled opponents (ARCHITECTURE §4.11, MATH.md §7).

A *session* is a table of opponents drawn from a bots file plus one hero seat, played for
``hands`` cash-game hands (stacks reset every hand, button rotates). In a duplicate match each
session is played once per hero policy with the same deck seeds, the same opponents (fresh
clones with the same parameters and RNG seeds) and the same hero seat, so luck and opponent
draws cancel in the per-session difference. Optionally all-in runouts are replaced by their
expectation (``allin``).

Win rates are bb/100. Hands within a session are not independent (opponents carry state), so the
95% CI comes from a bootstrap over whole sessions (MATH.md §7.1).

Policies are named (``make_policy``) so that worker processes can build them.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokerbot.engine.table import Policy, play_hand
from pokerbot.eval.allin import allin_adjusted
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.validate import SEATS

BIG_BLIND = 100
SMALL_BLIND = 50
HERO_BUYIN_BB = 100


def make_policy(spec: str, seed: int) -> Policy:
    """``abc`` | ``hand-strength`` | ``blueprint:<path>`` -> a fresh policy."""
    if spec == "abc":
        from pokerbot.baselines.heuristic import abc_tag

        return abc_tag(seed)
    if spec == "hand-strength":
        from pokerbot.baselines.heuristic import HandStrengthBot

        return HandStrengthBot()
    if spec.startswith("blueprint:"):
        from pokerbot.baselines.blueprint import BlueprintPolicy

        return BlueprintPolicy.load(Path(spec.split(":", 1)[1]), seed)
    raise ValueError(f"unknown policy {spec!r}")


@dataclass(frozen=True)
class SessionSpec:
    bots_path: str
    fmt: str
    seats: int
    hands: int
    seed: int
    style: int | None  # all opponents from this style; None = drawn from the population mix
    spread: float
    allin: bool


def play_session(spec: SessionSpec, heroes: Sequence[str]) -> np.ndarray:
    """Hero net bb per hand for each hero policy: shape (len(heroes), hands)."""
    rng = np.random.default_rng(spec.seed)
    sampler = StyleSampler(Path(spec.bots_path), spec.fmt, spread=spec.spread)
    hero_seat = int(rng.integers(spec.seats))
    opponents = [sampler.draw(rng, style=spec.style).bot for _ in range(spec.seats - 1)]
    hero_seed = int(rng.integers(2**31))
    deck_seeds = rng.integers(0, 2**63 - 1, size=spec.hands)
    out = np.zeros((len(heroes), spec.hands))
    for k, name in enumerate(heroes):
        bots = [b.clone() for b in opponents]
        hero = make_policy(name, hero_seed)
        table: list[Policy] = [*bots[:hero_seat], hero, *bots[hero_seat:]]
        stacks = [
            int(round((HERO_BUYIN_BB if s == hero_seat else table[s].buyin_bb) * BIG_BLIND))
            for s in range(spec.seats)
        ]
        for h in range(spec.hands):
            result = play_hand(
                table, stacks, button=h % spec.seats, deck_seed=int(deck_seeds[h]),
                small_blind=SMALL_BLIND, big_blind=BIG_BLIND, hand_id=h,
            )  # fmt: skip
            pay = float(result.payoffs[hero_seat])
            if spec.allin:
                adj = allin_adjusted(result.events, seed=int(deck_seeds[h]) & 0xFFFFFFFF)
                if adj is not None:
                    pay = float(adj[hero_seat])
            out[k, h] = pay / BIG_BLIND
    return out


def _session(args) -> np.ndarray:
    return play_session(*args)


def bootstrap_ci(per_session: np.ndarray, hands: int, boots: int = 2000, seed: int = 0):
    """Mean bb/100 and its 95% bootstrap CI from per-session totals (all sessions equal length)."""
    rng = np.random.default_rng(seed)
    rate = per_session / hands * 100
    idx = rng.integers(0, len(rate), size=(boots, len(rate)))
    means = rate[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(rate.mean()), float(lo), float(hi)


def run_match(
    heroes: Sequence[str],
    bots_path: Path,
    fmt: str,
    sessions: int,
    hands: int,
    seed: int,
    style: int | None = None,
    spread: float = 1.0,
    allin: bool = True,
    workers: int = 8,
) -> dict:
    """Play ``sessions`` duplicate sessions for every hero; per hero bb/100 with CI, and for two
    heroes the duplicate difference (first − second) with CI."""
    specs = [
        SessionSpec(
            str(bots_path), fmt, SEATS[fmt], hands, seed * 1_000_003 + i, style, spread, allin
        )  # fmt: skip
        for i in range(sessions)
    ]
    with ProcessPoolExecutor(workers) as pool:
        per = np.array(list(pool.map(_session, [(s, tuple(heroes)) for s in specs])))
    totals = per.sum(axis=2)  # (sessions, heroes)
    out: dict = {"sessions": sessions, "hands_per_session": hands, "hands": sessions * hands,
                 "heroes": {}}  # fmt: skip
    for k, name in enumerate(heroes):
        mean, lo, hi = bootstrap_ci(totals[:, k], hands)
        out["heroes"][name] = {"bb100": mean, "ci95": [lo, hi]}
    if len(heroes) == 2:
        mean, lo, hi = bootstrap_ci(totals[:, 0] - totals[:, 1], hands)
        out["delta"] = {"bb100": mean, "ci95": [lo, hi], "of": f"{heroes[0]} - {heroes[1]}"}
    return out
