"""Frozen harness for heads-up hold'em exploiter programs (Track B; EVOLUTION.md §3, MATH.md §7).

A *session* is ``HANDS`` hands against one opponent, button alternating:
  style   a bot of one of the heads-up styles (``profiles/bots-v3.json``, spread 1);
  nash    the blueprint itself (the exploiter must not lose to it);
  switch  an opponent whose style changes every ``SWITCH_EVERY`` hands.
The session is played twice with the same deck seeds and an identical (cloned) opponent: once
with the exploiter in the hero seat, once with the blueprint. d = payoff difference per hand in
big blinds, all-in runouts replaced by their expectation (MATH.md §7.2).

Beliefs: before each hand the frozen heads-up classifier (``CLASSIFIER``, tuned for heads-up,
with the blueprint as an extra "equilibrium" style) gives P(style) for the opponent from the
public events and the harness's feature counts; the program receives all K+1 numbers.

Each decision: the blueprint's distribution over the menu (``MENU_NAMES``) is the baseline; the
program's output is validated, restricted to legal actions and sampled by the harness's RNG; the
menu action becomes a real bet with the blueprint's size mapping.

Session score (bb/100; fitness = mean over sessions, minus a time penalty):
  style  mean d;  nash  −λ1·max(0, −mean d − ε);  switch  −λ2·max(0, −mean d);  all  −λ3·KL.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from pokerbot.baselines.blueprint import MENU, BlueprintPolicy
from pokerbot.engine.actions import ActionType, Street
from pokerbot.engine.events import Event, HoleCards, PlayerAction, Showdown
from pokerbot.engine.state import DecisionState
from pokerbot.engine.table import play_hand
from pokerbot.eval.allin import allin_adjusted
from pokerbot.evolve import program
from pokerbot.evolve.api import HUHandRecord, HUState, StyleProfiles
from pokerbot.evolve.harness import FeatureTracker, redact
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.strength import postflop_equity, preflop_percentile

BLUEPRINT = Path("profiles/blueprint-hu-v1.npz")
BOTS = Path("profiles/bots-v3.json")
PROFILES = Path("profiles/styles-v1.json")
SHOWDOWNS = Path("data/processed/showdown_ranges.json")
EQUILIBRIUM = Path("profiles/equilibrium-hu-v1.json")
CLASSIFIER = Path("programs/classifier_hu/champion-20261022-165422-evolve-classifier-hu")
HANDS = 200
SWITCH_EVERY = 50
BB = 100
SPLITS = {"train": 1, "val_a": 2, "val_b": 3, "test": 4}
KINDS = ("style",) * 14 + ("nash",) * 3 + ("switch",) * 3
LAMBDA_NASH, LAMBDA_SWITCH, LAMBDA_KL, EPSILON = 1.0, 1.0, 2.0, 2.0  # bb/100 units
TIME_BUDGET_US, LAMBDA_TIME = 2000.0, 0.01
KL_FLOOR = 1e-3


@lru_cache(maxsize=1)
def profiles() -> StyleProfiles:
    return StyleProfiles.load(PROFILES, "hu", SHOWDOWNS, EQUILIBRIUM)


@lru_cache(maxsize=1)
def _classifier():
    import json

    src = CLASSIFIER.with_suffix(".py").read_text(encoding="utf-8")
    params = json.loads(CLASSIFIER.with_suffix(".json").read_text(encoding="utf-8"))["params"]
    return program.load(src)["Classifier"], params


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


def opponent_plan(spec: SessionSpec) -> list[tuple[int, object]]:
    """(true style, policy) per hand; style index K is the blueprint. Policies are fresh."""
    rng = np.random.default_rng(spec.seed)
    sampler = StyleSampler(BOTS, "hu")
    k = len(sampler.styles)
    if spec.kind == "nash":
        bp = BlueprintPolicy.load(BLUEPRINT, int(rng.integers(2**31)))
        return [(k, bp)] * HANDS
    if spec.kind == "style":
        before = (spec.index // len(KINDS)) * KINDS.count("style") + spec.index % len(KINDS)
        d = sampler.draw(rng, style=before % k)
        return [(d.style, d.bot)] * HANDS
    plan = []
    for h in range(HANDS):
        if h % SWITCH_EVERY == 0:
            d = sampler.draw(rng, style=int(rng.integers(k)))
        plan.append((d.style, d.bot))
    return plan


def _clone(policy):
    if isinstance(policy, BlueprintPolicy):
        return BlueprintPolicy.load(BLUEPRINT, policy.seed)
    return policy.clone()


def _actions(state_actions, me: int) -> tuple:
    out = []
    for a in state_actions:
        kind = {ActionType.FOLD: "f", ActionType.CHECK_CALL: "c", ActionType.BET_RAISE: "r"}
        out.append((int(a.street), "me" if a.seat == me else "opp", kind[a.type], a.to / BB))
    return tuple(out)


def _legal(state: DecisionState) -> np.ndarray:
    lg = state.legal
    raise_ok = lg.can_raise
    return np.array([lg.to_call > 0, True, raise_ok, raise_ok, raise_ok, raise_ok])


def kl(p: np.ndarray, base: np.ndarray) -> float:
    q = np.maximum(base, KL_FLOOR)
    m = p > 0
    return max(0.0, float(np.sum(p[m] * np.log(p[m] / q[m]))))


class ExploiterSeat:
    """Engine policy wrapping a program: baseline from the blueprint, validated output, sampling
    by the harness, real bet sizes from the blueprint's mapping."""

    buyin_bb = 100.0

    def __init__(self, prog, rng: np.random.Generator) -> None:
        self.prog = prog
        self.rng = rng
        self.bp = BlueprintPolicy.load(BLUEPRINT, 0)
        self.beliefs = profiles().shares.copy()
        self.opp_feats = None
        self.kls: list[float] = []
        self.invalid = self.calls = 0
        self.seconds = 0.0

    def act(self, state: DecisionState):
        base = self.bp.menu_distribution(state)
        legal = _legal(state)
        base = np.where(legal, base, 0.0)
        base = base / base.sum() if base.sum() > 0 else legal / legal.sum()
        if state.street == Street.PREFLOP:
            strength = 1.0 - preflop_percentile(state.hole)
        else:
            strength = postflop_equity(state.hole, state.board)
        hs = HUState(
            street=int(state.street),
            button=state.seat == state.button,
            hole=state.hole,
            board=state.board,
            strength=float(strength),
            pot=state.pot / BB,
            to_call=state.legal.to_call / BB,
            my_stack=state.stacks[state.seat] / BB,
            opp_stack=state.stacks[1 - state.seat] / BB,
            legal=tuple(bool(x) for x in legal),
            actions=_actions(state.actions, state.seat),
        )
        t0 = time.perf_counter()
        out = self.prog.policy(hs, self.beliefs, self.opp_feats, base.copy())
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        try:
            p = np.asarray(out, dtype=float).reshape(-1)
            ok = p.shape == (len(MENU),) and np.all(np.isfinite(p)) and np.all(p >= 0)
        except Exception:  # noqa: BLE001 - garbage output is just invalid
            ok = False
        if ok:
            p = np.where(legal, p, 0.0)
            ok = p.sum() > 0
        if not ok:
            self.invalid += 1
            p = base
        p = p / p.sum()
        self.kls.append(kl(p, base))
        return BlueprintPolicy._real(state, MENU[int(self.rng.choice(len(MENU), p=p))])


def _payoff(result, seat: int, deck_seed: int) -> float:
    adj = allin_adjusted(result.events, seed=deck_seed & 0xFFFFFFFF)
    return float(adj[seat] if adj is not None else result.payoffs[seat]) / BB


def run_session(make, spec: SessionSpec, beliefs_mode: str = "classifier") -> dict:
    """``make()`` -> a fresh Exploiter program. beliefs_mode: classifier | oracle | prior."""
    prof = profiles()
    plan = opponent_plan(spec)
    rng = np.random.default_rng(spec.seed + 7)
    deck_seeds = rng.integers(0, 2**63 - 1, size=HANDS)
    hero_seed = int(rng.integers(2**31))
    # exploiter run
    prog = make()
    seat = ExploiterSeat(prog, np.random.default_rng(hero_seed))
    Clf, cparams = _classifier()
    clf = Clf(prof, dict(cparams))
    tracker = FeatureTracker()
    pay_x = np.zeros(HANDS)
    opp = None
    for h in range(HANDS):
        style, base_opp = plan[h]
        if opp is None or base_opp is not plan[h - 1][1]:
            opp = _clone(base_opp)
        feats = tracker.features(1)
        seat.opp_feats = feats
        if beliefs_mode == "oracle":
            b = np.eye(prof.k)[style]
        elif beliefs_mode == "prior":
            b = prof.shares.copy()
        else:
            b, _ = _valid(clf.posterior(1, feats), prof.k)
        b.setflags(write=False)
        seat.beliefs = b
        stacks = [100 * BB, int(round(opp.buyin_bb * BB))]
        result = play_hand(
            [seat, opp],
            stacks,
            button=h % 2,
            deck_seed=int(deck_seeds[h]),
            small_blind=BB // 2,
            big_blind=BB,
            hand_id=h,
        )
        pay_x[h] = _payoff(result, 0, int(deck_seeds[h]))
        public = redact(result.events)
        for e in public:
            clf.observe(e)
        tracker.add_hand(public)
        prog.observe_hand(_record(result.events, h % 2 == 0, pay_x[h]))
    # baseline run: same decks, same hero RNG seed, identical opponents
    bp = BlueprintPolicy.load(BLUEPRINT, hero_seed)
    pay_b = np.zeros(HANDS)
    opp = None
    for h in range(HANDS):
        _, base_opp = plan[h]
        if opp is None or base_opp is not plan[h - 1][1]:
            opp = _clone(base_opp)
        stacks = [100 * BB, int(round(opp.buyin_bb * BB))]
        result = play_hand(
            [bp, opp],
            stacks,
            button=h % 2,
            deck_seed=int(deck_seeds[h]),
            small_blind=BB // 2,
            big_blind=BB,
            hand_id=h,
        )
        pay_b[h] = _payoff(result, 0, int(deck_seeds[h]))
    return {
        "kind": spec.kind,
        "d": pay_x - pay_b,
        "kl": float(np.mean(seat.kls)) if seat.kls else 0.0,
        "invalid": seat.invalid,
        "calls": seat.calls,
        "us": 1e6 * seat.seconds / max(seat.calls, 1),
        "labels": [s for s, _ in plan],
    }


def _valid(p, k: int) -> tuple[np.ndarray, bool]:
    a = np.asarray(p, dtype=float).reshape(-1)
    if a.shape != (k,) or not np.all(np.isfinite(a)) or np.any(a < 0) or a.sum() <= 0:
        return np.full(k, 1.0 / k), False
    return a / a.sum(), True


def _record(events: list[Event], button: bool, payoff: float) -> HUHandRecord:
    hole = next(e.cards for e in events if isinstance(e, HoleCards) and e.seat == 0)
    shows = {e.seat: e.cards for e in events if isinstance(e, Showdown)}
    boards = [e.board for e in events if hasattr(e, "board")]
    acts = [e for e in events if isinstance(e, PlayerAction)]
    return HUHandRecord(
        button, hole, boards[-1] if boards else (), shows.get(1), _actions(acts, 0), payoff
    )


def session_score(r: dict) -> float:
    m = 100.0 * float(r["d"].mean())
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
        ds = [100 * r["d"].mean() for r in results if r["kind"] == kind]
        out[f"bb100_{kind}"] = float(np.mean(ds)) if ds else float("nan")
    names = profiles().names
    by_style = defaultdict(list)
    for r in results:
        if r["kind"] == "style":
            by_style[r["labels"][0]].append(100 * r["d"].mean())
    for k, v in sorted(by_style.items()):
        out[f"bb100_vs_{names[k]}"] = float(np.mean(v))
    out["kl"] = float(np.mean([r["kl"] for r in results]))
    calls = sum(r["calls"] for r in results)
    out["us_per_call"] = sum(r["us"] * r["calls"] for r in results) / max(calls, 1)
    out["invalid"] = sum(r["invalid"] for r in results)
    out["fitness"] -= LAMBDA_TIME * max(0.0, out["us_per_call"] - TIME_BUDGET_US)
    return out
