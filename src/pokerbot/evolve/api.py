"""Frozen types given to evolved programs (EVOLUTION.md §3). Programs may import only this module,
``numpy`` and ``math``.

- ``StyleProfiles``: the K styles of one table format, read-only arrays.
- ``SeatFeatures``: harness-maintained public counts for one seat (MATH.md §2): every stat as
  (opportunities n, occurrences x), split by position, stack depth, pot type and more, plus the
  shown-hand strength counts by betting line (MATH.md §4.5).
- ``Event`` and the event classes: the engine's public event stream (hole cards of other players
  appear only in ``Showdown`` reveals).
- ``param``: declares a numeric knob in ``PARAMS``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from pokerbot.engine.actions import ActionType, Street
from pokerbot.engine.events import (
    BlindPosted,
    BoardDealt,
    Event,
    HandEnd,
    HandStart,
    Muck,
    PlayerAction,
    Showdown,
)
from pokerbot.stats.showdown import BUCKETS, LINES, STRENGTH_EDGES

__all__ = [
    "LINES", "BUCKETS", "STRENGTH_EDGES", "ActionType", "Street", "Event", "HandStart",
    "BlindPosted", "PlayerAction", "BoardDealt", "Showdown", "Muck", "HandEnd", "Param", "param",
    "SeatFeatures", "StyleProfiles", "LeducState", "HandRecord", "OpponentFeatures", "StyleTable",
]  # fmt: skip


@dataclass(frozen=True)
class Param:
    default: float
    low: float
    high: float


def param(default: float, low: float, high: float) -> Param:
    if not low <= default <= high:
        raise ValueError(f"param default {default} outside [{low}, {high}]")
    return Param(float(default), float(low), float(high))


def _ro(a) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class SeatFeatures:
    """Public counts for one seat after ``hands`` hands. ``counts[stat] = (n, x)``; a missing
    stat had no opportunity. ``shown[line, bucket]`` counts the seat's hands shown at showdown."""

    hands: int
    counts: Mapping[str, tuple[int, int]]
    shown: np.ndarray  # (len(LINES), BUCKETS)

    def n(self, stat: str) -> int:
        return self.counts.get(stat, (0, 0))[0]

    def x(self, stat: str) -> int:
        return self.counts.get(stat, (0, 0))[1]

    @staticmethod
    def make(hands: int, counts: Mapping[str, tuple[int, int]], shown) -> SeatFeatures:
        return SeatFeatures(hands, MappingProxyType(dict(counts)), _ro(shown))


@dataclass(frozen=True)
class StyleProfiles:
    """Styles of one table format. Shapes: K styles, D features.

    names (K), shares π (K), features (D) stat names, prior_a/prior_b (D) population Beta prior,
    f_mean/f_std (D) standardization of logit features, means (K, D) and covs (K, D, D) of each
    style in standardized logit space, pooled_rate (K, D) pooled real rates, beta_a/beta_b (K, D)
    per-style Beta fits of member rates, showdown (K, len(LINES), BUCKETS) P(bucket | line, style)
    from real shown hands (rows sum to 1; uniform where no data).
    """

    names: tuple[str, ...]
    shares: np.ndarray
    features: tuple[str, ...]
    prior_a: np.ndarray
    prior_b: np.ndarray
    f_mean: np.ndarray
    f_std: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    pooled_rate: np.ndarray
    beta_a: np.ndarray
    beta_b: np.ndarray
    showdown: np.ndarray

    @property
    def k(self) -> int:
        return len(self.names)

    def transform(self, n: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Counts (D,) -> standardized features z and their noise variances (MATH.md §3.1)."""
        r = (x + self.prior_a) / (n + self.prior_a + self.prior_b)
        z = np.log(r / (1 - r))
        s2 = 1.0 / ((n + self.prior_a + self.prior_b) * r * (1 - r))
        return (z - self.f_mean) / self.f_std, s2 / self.f_std**2

    @staticmethod
    def load(
        profiles_path: Path,
        fmt: str,
        showdown_path: Path | None = None,
        equilibrium_path: Path | None = None,
    ) -> StyleProfiles:
        """``equilibrium_path``: a measured baseline pseudo-style (``evolve.equilibrium``),
        appended as the last style with its population share (the others scaled down)."""
        from pokerbot.archetypes.profiles import load_profiles

        prof = load_profiles(Path(profiles_path))["formats"][fmt]
        t = prof["transform"]
        styles = [dict(s) for s in prof["styles"]]
        eq = None
        if equilibrium_path:
            eq = json.loads(Path(equilibrium_path).read_text(encoding="utf-8"))
            total = sum(s["share"] for s in styles)
            for s in styles:
                s["share"] = s["share"] / total * (1 - eq["share"])
            styles.append(eq)
        k = len(styles)
        sd = np.full((k, len(LINES), BUCKETS), 1.0 / BUCKETS)
        if showdown_path and Path(showdown_path).exists():
            doc = json.loads(Path(showdown_path).read_text(encoding="utf-8"))
            if fmt in doc["formats"]:
                counts = np.array(doc["formats"][fmt]["counts"])
                counts = np.concatenate([counts, np.zeros((k - len(counts), *counts.shape[1:]))])
                tot = counts.sum(axis=2, keepdims=True)
                sd = np.where(tot > 0, (counts + 0.5) / (tot + 0.5 * BUCKETS), sd)
        if eq is not None:
            c = np.array(eq["showdown_counts"])
            tot = c.sum(axis=1, keepdims=True)
            sd[-1] = np.where(tot > 0, (c + 0.5) / (tot + 0.5 * BUCKETS), 1.0 / BUCKETS)
        shares = np.array([s["share"] for s in styles])
        return StyleProfiles(
            names=tuple(s["name"] for s in styles),
            shares=_ro(shares / shares.sum()),
            features=tuple(t.features),
            prior_a=_ro(t.prior_a), prior_b=_ro(t.prior_b), f_mean=_ro(t.mean), f_std=_ro(t.std),
            means=_ro([s["mean"] for s in styles]),
            covs=_ro([s["cov"] for s in styles]),
            pooled_rate=_ro([[s["pooled_rate"][f] for f in t.features] for s in styles]),
            beta_a=_ro([[s["beta"][f][0] for f in t.features] for s in styles]),
            beta_b=_ro([[s["beta"][f][1] for f in t.features] for s in styles]),
            showdown=_ro(sd),
        )  # fmt: skip


# ------------------------------------------------------------------------- Leduc (Track A)


@dataclass(frozen=True)
class LeducState:
    """The exploiter's decision point in Leduc poker (``games.limit.LEDUC``). Cards are ranks
    0 (J), 1 (Q), 2 (K); ``public`` is None before the flop. ``history`` holds one string per
    round from c (check/call), r (bet/raise), f (fold). ``legal`` lists the allowed actions in
    order (the output distribution follows it). ``seat`` 0 acts first in every round.
    ``put`` = chips each seat has put in, ``to_call`` = chips needed to call."""

    seat: int
    card: int
    public: int | None
    history: tuple[str, ...]
    legal: str
    put: tuple[int, int]
    to_call: int
    bet_size: int  # fixed bet/raise size of the current round (2, then 4)


@dataclass(frozen=True)
class HandRecord:
    """A finished Leduc hand as the exploiter saw it. ``opp_card`` is known only if the
    opponent's card was shown at showdown (else None). ``payoff`` is the exploiter's net chips."""

    seat: int
    card: int
    public: int | None
    history: tuple[str, ...]
    opp_card: int | None
    payoff: float


@dataclass(frozen=True)
class OpponentFeatures:
    """Public counts of the opponent's actions over the session so far.
    ``counts[(round, situation, action)]`` with round 0/1, situation "open" (no bet to call)
    or "facing" (a bet to call), action "c", "r" or "f". ``shown[(round_line, card)]`` counts
    cards the opponent showed at showdown by its final-round line (e.g. "rc"). ``hands``."""

    hands: int
    counts: Mapping[tuple[int, str, str], int]
    shown: Mapping[tuple[str, int], int]

    def rate(self, rnd: int, situation: str, action: str, prior: float = 1.0) -> float:
        """Smoothed frequency of an action in a situation (uniform prior over actions)."""
        acts = ("c", "r", "f") if situation == "facing" else ("c", "r")
        tot = sum(self.counts.get((rnd, situation, a), 0) for a in acts)
        x = self.counts.get((rnd, situation, action), 0)
        return (x + prior / len(acts)) / (tot + prior)


@dataclass(frozen=True)
class StyleTable:
    """The K Leduc opponent styles (``games.styles``): ``names``; ``centers`` (K, 3, 3) =
    [bet, call, raise] × strength [weak, medium, strong] probabilities of each style's typical
    player (players vary around it); ``shares`` (K) how often each style is met."""

    names: tuple[str, ...]
    centers: np.ndarray
    shares: np.ndarray

    @property
    def k(self) -> int:
        return len(self.names)
