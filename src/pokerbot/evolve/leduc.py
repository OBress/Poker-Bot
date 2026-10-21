"""Frozen harness for Leduc exploiter programs (Track A of Phase 6; EVOLUTION.md §3, MATH.md §7).

A *session* is ``HANDS`` hands against one opponent, seats alternating. Opponents per session:
  style   a player drawn from one of the five styles (``games.styles.sample_params``);
  nash    the frozen Nash baseline itself (the exploiter must not lose to it);
  switch  a player who changes style every ``SWITCH_EVERY`` hands (must not burn the exploiter).
Before every hand the exploiter gets beliefs from a frozen Bayesian classifier: the likelihood of
the opponent's past actions under each style's typical player, marginalized over its hidden card
(revealed cards are used), tempered by ``TEMPER`` because real players vary around the style.

Scoring is exact and duplicate: for the dealt cards, the hand's expected value is computed over
all action randomness of both players, once with the exploiter's distributions and once with the
baseline's, against the same opponent. d = EV(exploiter) − EV(baseline) per hand, chips/hand.
The realized hand (actions sampled by the harness's seeded RNG) drives what is observed next.

Session score (fitness is the mean over sessions; paired promotion tests use these):
  style  mean d                                   (Δ_mixed)
  nash   −λ1 · max(0, −mean d − ε)                (losses to the baseline beyond tolerance)
  switch −λ2 · max(0, −mean d)                    (getting burned by switchers)
and every session pays −λ3 · mean KL(σ ‖ σ_b) over the exploiter's decisions on the realized path
(baseline probabilities floored at ``KL_FLOOR``).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np

from pokerbot.baselines.cfr import Strategy
from pokerbot.baselines.track_a import load_strategy
from pokerbot.evolve.api import HandRecord, LeducState, OpponentFeatures, StyleTable
from pokerbot.games.limit import FOLD, LEDUC, RAISE
from pokerbot.games.styles import STYLES, center, sample_params, style_strategy

GAME = LEDUC
HANDS = 100
SWITCH_EVERY = 25
TEMPER = 0.7
LAMBDA_NASH, LAMBDA_SWITCH, LAMBDA_KL, EPSILON = 1.0, 1.0, 0.02, 0.01
SPLITS = {"train": 1, "val_a": 2, "val_b": 3, "test": 4}
KINDS = ("style",) * 14 + ("nash",) * 3 + ("switch",) * 3  # 70% / 15% / 15% of sessions
BASELINE = Path("profiles/baseline-leduc-v1.json")


def style_table() -> StyleTable:
    names = tuple(STYLES)
    centers = np.array([center(n) for n in names])
    centers.setflags(write=False)
    shares = np.full(len(names), 1 / len(names))
    shares.setflags(write=False)
    return StyleTable(names, centers, shares)


@lru_cache(maxsize=1)
def _nash() -> Strategy:
    return load_strategy(BASELINE)[1]


@lru_cache(maxsize=1)
def _centers() -> list[Strategy]:
    return [style_strategy(GAME, center(n)) for n in STYLES]


# ---------------------------------------------------------------------------------- sessions


@dataclass(frozen=True)
class SessionSpec:
    split: str
    index: int

    @property
    def seed(self) -> int:
        return SPLITS[self.split] * 10_000_000 + self.index

    @property
    def kind(self) -> str:
        return KINDS[self.index % len(KINDS)]


def opponents(spec: SessionSpec) -> tuple[list[Strategy], list[int | None]]:
    """Per hand: the opponent's strategy and its true style (None for nash)."""
    rng = np.random.default_rng(spec.seed)
    names = list(STYLES)
    if spec.kind == "nash":
        return [_nash()] * HANDS, [None] * HANDS
    if spec.kind == "style":
        before = (spec.index // len(KINDS)) * KINDS.count("style") + spec.index % len(KINDS)
        k = before % len(names)  # style sessions cycle through the styles evenly
        s = style_strategy(GAME, sample_params(names[k], rng))
        return [s] * HANDS, [k] * HANDS
    strategies, labels = [], []
    order = rng.permutation(len(names))
    for h in range(HANDS):
        k = int(order[(h // SWITCH_EVERY) % len(names)])
        if h % SWITCH_EVERY == 0:
            s = style_strategy(GAME, sample_params(names[k], rng))
        strategies.append(s)
        labels.append(k)
    return strategies, labels


# ---------------------------------------------------------------------------------- beliefs


class Beliefs:
    """Frozen Bayesian style classifier for Leduc (the exploiter's input, not evolved here)."""

    def __init__(self, table: StyleTable) -> None:
        self.log = np.log(table.shares).astype(float)
        self.ll = np.zeros(table.k)

    def posterior(self) -> np.ndarray:
        z = self.log + TEMPER * self.ll
        e = np.exp(z - z.max())
        return e / e.sum()

    def update(self, rec: HandRecord) -> None:
        opp = 1 - rec.seat
        cards = [rec.opp_card] if rec.opp_card is not None else range(GAME.ranks)
        lik = np.zeros(len(self.ll))
        for y in cards:
            w = _card_weight(rec.card, y, rec.public)
            if w <= 0:
                continue
            for k, strat in enumerate(_centers()):
                lik[k] += w * _actions_prob(strat, y, rec.public, rec.history, opp)
        self.ll += np.log(np.maximum(lik, 1e-12))


def _card_weight(x: int, y: int, public: int | None) -> float:
    """P(opponent card y | own card x [, public card])."""
    deals = dict(GAME.deals)
    p = deals.get((x, y), 0.0)
    if public is not None and p > 0:
        p *= dict(GAME.public_cards(x, y)).get(public, 0.0)
    return p


def _actions_prob(strat: Strategy, y: int, public, history, opp_seat: int) -> float:
    prob = 1.0
    for r, h in enumerate(history):
        pub = public if r > 0 else None
        for i, a in enumerate(h):
            if i % 2 != opp_seat:
                continue
            rounds = [*history[:r], h[:i]]
            sigma = strat[GAME.infoset(y, pub, rounds)]
            prob *= sigma[GAME.legal(h[:i]).index(a)]
    return prob


# ---------------------------------------------------------------------------------- features


class FeatureTracker:
    def __init__(self) -> None:
        self.counts: dict = defaultdict(int)
        self.shown: dict = defaultdict(int)
        self.hands = 0

    def update(self, rec: HandRecord) -> None:
        self.hands += 1
        opp = 1 - rec.seat
        for r, h in enumerate(rec.history):
            for i, a in enumerate(h):
                if i % 2 == opp:
                    facing = i > 0 and h[i - 1] == RAISE
                    self.counts[(r, "facing" if facing else "open", a)] += 1
        if rec.opp_card is not None:
            self.shown[(rec.history[-1], rec.opp_card)] += 1

    def features(self) -> OpponentFeatures:
        return OpponentFeatures(self.hands, MappingProxyType(dict(self.counts)),
                                MappingProxyType(dict(self.shown)))  # fmt: skip


# ---------------------------------------------------------------------------------- hands


def _state(seat: int, card: int, public, rounds: list[str]) -> LeducState:
    h = rounds[-1]
    put = GAME.contributions(rounds)
    to_call = abs(put[0] - put[1])
    return LeducState(seat, card, public, tuple(rounds), GAME.legal(h), put, to_call,
                      GAME.bet_sizes[len(rounds) - 1])  # fmt: skip


KL_FLOOR = 1e-3  # baseline probabilities are floored so a never-taken action costs finite KL


def kl(p: np.ndarray, base: np.ndarray) -> float:
    """KL(p || base) with base floored (unnormalized) at KL_FLOOR, clipped at 0: exactly 0 when
    p equals the baseline, large but finite for mass on actions the baseline never takes."""
    q = np.maximum(base, KL_FLOOR)
    m = p > 0
    return max(0.0, float(np.sum(p[m] * np.log(p[m] / q[m]))))


def validate(p, n: int, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        a = np.asarray(p, dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001 - garbage output is just invalid
        return fallback, False
    if a.shape != (n,) or not np.all(np.isfinite(a)) or np.any(a < 0) or a.sum() <= 0:
        return fallback, False
    return a / a.sum(), True


def hand_value(hero, opp: Strategy, seat: int, cards: tuple[int, int], public: int) -> float:
    """Exact EV for the hero over all action randomness; ``hero(card, public, rounds)`` -> dist."""

    def walk(rounds: list[str], pub) -> float:
        h = rounds[-1]
        if GAME.round_over(h):
            if FOLD in h or len(rounds) == GAME.rounds:
                u = GAME.payoff(rounds, cards, pub)
                return u if seat == 0 else -u
            return walk([*rounds, ""], public)
        actor = GAME.to_act(h)
        acts = GAME.legal(h)
        if actor == seat:
            sigma = hero(cards[seat], pub, rounds)
        else:
            sigma = opp[GAME.infoset(cards[actor], pub, rounds)]
        return sum(sigma[i] * walk([*rounds[:-1], h + a], pub)
                   for i, a in enumerate(acts) if sigma[i] > 0)  # fmt: skip

    return walk([""], None)


def _deal(rng: np.random.Generator) -> tuple[int, int, int]:
    deck = [r for r in range(GAME.ranks) for _ in range(GAME.suits)]
    x, y, p = rng.choice(len(deck), size=3, replace=False)
    return deck[x], deck[y], deck[p]


def run_session(make, spec: SessionSpec, oracle: bool = False, prior_only: bool = False) -> dict:
    """Play one session with a fresh program. Returns per-hand d, KL and bookkeeping.
    Ablations: ``oracle`` gives the true style as one-hot beliefs (population mix vs Nash);
    ``prior_only`` gives the population mix throughout (no classifier)."""
    table = style_table()
    nash = _nash()
    strategies, labels = opponents(spec)
    rng = np.random.default_rng(spec.seed + 7)
    prog = make()
    beliefs = Beliefs(table)
    feats = FeatureTracker()
    d = np.zeros(HANDS)
    kls: list[float] = []
    invalid = calls = 0
    seconds = 0.0
    for h in range(HANDS):
        seat = h % 2
        x, y, pub = _deal(rng)
        cards = (x, y) if seat == 0 else (y, x)
        if prior_only:
            b = table.shares.copy()
        elif oracle:
            b = np.eye(table.k)[labels[h]] if labels[h] is not None else table.shares.copy()
        else:
            b = beliefs.posterior()
        b.setflags(write=False)
        f = feats.features()
        cache: dict = {}

        def hero(card, public, rounds, b=b, f=f, seat=seat, cache=cache):
            nonlocal invalid, calls, seconds
            key = (public, tuple(rounds))
            if key not in cache:
                base = nash[GAME.infoset(card, public, rounds)]
                t0 = time.perf_counter()
                out = prog.policy(_state(seat, card, public, rounds), b, f, base.copy())
                seconds += time.perf_counter() - t0
                calls += 1
                p, ok = validate(out, len(base), base)
                invalid += not ok
                cache[key] = (p, base)
            return cache[key][0]

        opp = strategies[h]
        ev_hero = hand_value(hero, opp, seat, cards, pub)
        ev_base = hand_value(lambda c, p, r: nash[GAME.infoset(c, p, r)], opp, seat, cards, pub)
        d[h] = ev_hero - ev_base
        # realized hand: sample both players' actions
        rounds = [""]
        public = None
        while True:
            hh = rounds[-1]
            if GAME.round_over(hh):
                if FOLD in hh or len(rounds) == GAME.rounds:
                    break
                rounds.append("")
                public = pub
                continue
            actor = GAME.to_act(hh)
            acts = GAME.legal(hh)
            if actor == seat:
                sigma = hero(cards[seat], public, rounds)
                base = cache[(public, tuple(rounds))][1]
                kls.append(kl(sigma, base))
            else:
                sigma = opp[GAME.infoset(cards[actor], public, rounds)]
            a = acts[int(rng.choice(len(acts), p=sigma / sigma.sum()))]
            rounds[-1] = hh + a
        showdown = FOLD not in rounds[-1]
        payoff = GAME.payoff(rounds, cards, public if len(rounds) > 1 else None)
        rec = HandRecord(seat, cards[seat], public if len(rounds) > 1 else None, tuple(rounds),
                         cards[1 - seat] if showdown else None,
                         float(payoff if seat == 0 else -payoff))  # fmt: skip
        beliefs.update(rec)
        feats.update(rec)
        prog.observe_hand(rec)
    return {"kind": spec.kind, "d": d, "kl": float(np.mean(kls)) if kls else 0.0,
            "invalid": invalid, "calls": calls, "us": 1e6 * seconds / max(calls, 1),
            "labels": labels}  # fmt: skip


def session_score(r: dict) -> float:
    m = float(r["d"].mean())
    if r["kind"] == "style":
        s = m
    elif r["kind"] == "nash":
        s = -LAMBDA_NASH * max(0.0, -m - EPSILON)
    else:
        s = -LAMBDA_SWITCH * max(0.0, -m)
    return s - LAMBDA_KL * r["kl"]


def summarize(results: list[dict]) -> dict:
    scores = np.array([session_score(r) for r in results])
    out: dict = {"sessions": len(results), "fitness": float(scores.mean())}
    for kind in ("style", "nash", "switch"):
        ds = [r["d"].mean() for r in results if r["kind"] == kind]
        out[f"delta_{kind}"] = float(np.mean(ds)) if ds else float("nan")
    by_style = defaultdict(list)
    for r in results:
        if r["kind"] == "style":
            by_style[r["labels"][0]].append(r["d"].mean())
    names = list(STYLES)
    for k, v in sorted(by_style.items()):
        out[f"delta_vs_{names[k]}"] = float(np.mean(v))
    early = [r["d"][:20].mean() for r in results if r["kind"] == "style"]
    late = [r["d"][-40:].mean() for r in results if r["kind"] == "style"]
    out["delta_style_first20"] = float(np.mean(early)) if early else float("nan")
    out["delta_style_last40"] = float(np.mean(late)) if late else float("nan")
    out["kl"] = float(np.mean([r["kl"] for r in results]))
    calls = sum(r["calls"] for r in results)
    out["us_per_call"] = sum(r["us"] * r["calls"] for r in results) / max(calls, 1)
    out["invalid"] = sum(r["invalid"] for r in results)
    return out
