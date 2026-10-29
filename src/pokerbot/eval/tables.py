"""Duplicate tables: a hero and a reference player take the same seat, cards and opponents.

One session seats the hero at a table of opponents and plays it twice with identical deck seeds and
identically seeded opponents: once with the hero, once with the reference in the hero's seat.
d_h = u_h(hero) - u_h(reference) is the per-hand edge (MATH.md §7.1). The hero's seat rotates over
sessions, opponents are drawn from a pool and seated in a random order.

Two table formats:
- ``cash``: stacks reset every hand, button rotates (our standard measurement). With ``allin``
  all-in runouts are replaced by their expectation (``eval/allin.py``, MATH.md §7.2).
- ``match``: fullhouse/e-x.club tournament rules. Stacks carry over between hands, a seat with no
  chips is out for the rest of the match (no rebuys), the dealer counter moves one alive seat per
  hand. A busted hero scores 0 for the remaining hands. Once one copy's table differs (a bust,
  different stacks), cards stop lining up seat for seat, so duplicate correlation weakens here.

This complements ``eval/duplicate.py`` (opponents sampled from the style pool, cash only): here the
opponents come from an explicit pool of named players, e.g. the fullhouse reference bots.

Standard errors come from a bootstrap over whole sessions (opponent and hero state carry over
within a session, so hands aren't independent).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pokerbot.engine.table import Policy, play_hand
from pokerbot.eval.allin import allin_adjusted
from pokerbot.eval.players import DEFAULT_BOTS, make_policy


@dataclass(frozen=True)
class TableFormat:
    mode: Literal["cash", "match"]
    hands: int  # per session
    stack: int = 10_000
    small_blind: int = 50
    big_blind: int = 100
    allin: bool = False  # cash only: expected value instead of the realized runout


@dataclass(frozen=True)
class Seating:
    """Everything that defines one session apart from who sits in the hero's seat."""

    session: int
    hero_seat: int
    opponents: tuple[str, ...]  # specs, in seat order, hero's seat excluded
    opponent_seeds: tuple[int, ...]
    hero_seed: int
    deck_seed: int


@dataclass(frozen=True)
class SessionResult:
    seating: Seating
    hero: np.ndarray  # hero's net chips per hand
    ref: np.ndarray  # reference's net chips per hand, same seat and cards
    hero_errors: int = 0
    ref_errors: int = 0

    @property
    def hands(self) -> int:
        return len(self.hero)


def seatings(pool: Sequence[str], seats: int, sessions: int, seed: int) -> list[Seating]:
    """Session plans. Hero seats rotate; opponents are drawn without replacement when the pool is
    big enough, with replacement otherwise."""
    out = []
    for s, ss in enumerate(np.random.SeedSequence(seed).spawn(sessions)):
        rng = np.random.default_rng(ss)
        idx = rng.choice(len(pool), size=seats - 1, replace=len(pool) < seats - 1)
        out.append(
            Seating(
                session=s,
                hero_seat=s % seats,
                opponents=tuple(pool[i] for i in idx),
                opponent_seeds=tuple(int(x) for x in rng.integers(0, 2**63 - 1, size=seats - 1)),
                hero_seed=int(rng.integers(0, 2**63 - 1)),
                deck_seed=int(rng.integers(0, 2**63 - 1)),
            )
        )
    return out


def play_table(policies: Sequence[Policy], fmt: TableFormat, deck_seed: int) -> np.ndarray:
    """Net chips per hand and seat, shape (hands, seats)."""
    adjust = fmt.allin and fmt.mode == "cash"
    n = len(policies)
    deck_seeds = np.random.default_rng(deck_seed).integers(0, 2**63 - 1, size=fmt.hands)
    per_hand = np.zeros((fmt.hands, n))
    stacks = [fmt.stack] * n
    for h in range(fmt.hands):
        if fmt.mode == "cash":
            alive = list(range(n))
            button = h % n
        else:
            alive = [i for i in range(n) if stacks[i] > 0]
            if len(alive) < 2:
                break
            button = h % len(alive)
        result = play_hand(
            [policies[i] for i in alive],
            [stacks[i] for i in alive],
            button=button,
            deck_seed=int(deck_seeds[h]),
            small_blind=fmt.small_blind,
            big_blind=fmt.big_blind,
            hand_id=h,
        )
        payoffs = result.payoffs
        if adjust:
            adj = allin_adjusted(result.events, seed=int(deck_seeds[h]) & 0xFFFFFFFF)
            if adj is not None:
                payoffs = adj
        for k, i in enumerate(alive):
            per_hand[h, i] = payoffs[k]
            if fmt.mode == "match":
                stacks[i] += result.payoffs[k]
    return per_hand


def play_session(
    seating: Seating,
    hero: str,
    reference: str,
    fmt: TableFormat,
    bots_path: Path = DEFAULT_BOTS,
) -> SessionResult:
    stack_bb = fmt.stack / fmt.big_blind

    def run(spec: str) -> tuple[np.ndarray, int]:
        opponents = [
            make_policy(o, sd, stack_bb, bots_path)
            for o, sd in zip(seating.opponents, seating.opponent_seeds, strict=True)
        ]
        player = make_policy(spec, seating.hero_seed, stack_bb, bots_path)
        table = opponents[: seating.hero_seat] + [player] + opponents[seating.hero_seat :]
        chips = play_table(table, fmt, seating.deck_seed)[:, seating.hero_seat]
        return chips, getattr(player, "errors", 0)

    hero_chips, hero_errors = run(hero)
    ref_chips, ref_errors = run(reference)
    return SessionResult(seating, hero_chips, ref_chips, hero_errors, ref_errors)


def _play(args: tuple) -> SessionResult:
    return play_session(*args)


def run_duplicate(
    hero: str,
    reference: str,
    pool: Sequence[str],
    fmt: TableFormat,
    sessions: int,
    seed: int,
    seats: int = 6,
    workers: int = 1,
    bots_path: Path = DEFAULT_BOTS,
    progress: Callable[[int, int], None] | None = None,
) -> list[SessionResult]:
    jobs = [(st, hero, reference, fmt, bots_path) for st in seatings(pool, seats, sessions, seed)]
    if workers <= 1:
        results = []
        for job in jobs:
            results.append(_play(job))
            if progress:
                progress(len(results), len(jobs))
        return results
    with ProcessPoolExecutor(workers) as ex:
        results = []
        for r in ex.map(_play, jobs, chunksize=max(1, len(jobs) // (workers * 4))):
            results.append(r)
            if progress:
                progress(len(results), len(jobs))
        return results


# ------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Estimate:
    bb100: float
    ci: float  # half-width of the 95% interval
    se: float

    def __str__(self) -> str:
        return f"{self.bb100:+.2f} +/- {self.ci:.2f} bb/100"


def bootstrap(
    totals: np.ndarray, hands: np.ndarray, rng: np.random.Generator, n_boot: int = 2000
) -> Estimate:
    """bb/100 = 100 * sum(totals) / sum(hands), SE from resampling whole sessions."""
    point = 100 * totals.sum() / hands.sum()
    idx = rng.integers(0, len(totals), size=(n_boot, len(totals)))
    boots = 100 * totals[idx].sum(axis=1) / hands[idx].sum(axis=1)
    se = float(boots.std(ddof=1))
    return Estimate(float(point), 1.96 * se, se)


def summarize(results: Sequence[SessionResult], fmt: TableFormat, seed: int = 0) -> dict:
    bb = fmt.big_blind
    hands = np.array([r.hands for r in results], dtype=float)
    hero = np.array([r.hero.sum() for r in results], dtype=float) / bb
    ref = np.array([r.ref.sum() for r in results], dtype=float) / bb
    rng = np.random.default_rng(seed)
    delta = bootstrap(hero - ref, hands, rng)
    # SD of d per 100 hands, the s in MATH.md §7.4 (sizes future runs).
    per100 = 100 * (hero - ref) / hands
    s = float(per100.std(ddof=1) * np.sqrt(hands.mean() / 100)) if len(results) > 1 else 0.0
    out = {
        "sessions": len(results),
        "hands": int(hands.sum()),
        "delta": delta,
        "hero": bootstrap(hero, hands, rng),
        "reference": bootstrap(ref, hands, rng),
        "sd_d_per100": s,
        "hero_errors": sum(r.hero_errors for r in results),
        "ref_errors": sum(r.ref_errors for r in results),
    }
    if fmt.mode == "match":
        out["hero_chip_delta_mean"] = float(np.mean([r.hero.sum() for r in results]))
        out["ref_chip_delta_mean"] = float(np.mean([r.ref.sum() for r in results]))
        out["hero_bust_rate"] = float(np.mean([r.hero.sum() <= -fmt.stack for r in results]))
        out["ref_bust_rate"] = float(np.mean([r.ref.sum() <= -fmt.stack for r in results]))
    return out
