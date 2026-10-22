"""Play the trained heads-up blueprint (``mccfr``) in the real engine.

Every real decision is translated into the abstract game (``hunl``): the hand's actions are
replayed into an abstract state, each real bet or raise mapped to the legal abstract size whose
pot fraction is closest in log space (all-ins map to ``a``), and real street changes force the
abstract state onto the same street. The blueprint's average strategy at the resulting
information set is sampled, and the abstract action is converted back with the real pot
(``f``/``c`` directly, sizes as pot fractions, ``a`` as all-in). Stacks other than 100 bb are
played with the same pot fractions. Information sets the blueprint never visited fall back to
``HandStrengthBot``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from pokerbot.baselines import hunl
from pokerbot.baselines.heuristic import HandStrengthBot
from pokerbot.engine.actions import Action, ActionType, Street
from pokerbot.engine.state import DecisionState


@lru_cache(maxsize=2)
def _load(path: str) -> tuple[dict[str, np.ndarray], list, dict]:
    z = np.load(path, allow_pickle=True)
    keys = z["keys"].tolist()
    probs = z["probs"].astype(np.float64)
    table = {k: probs[i] for i, k in enumerate(keys)}
    meta = eval(str(z["meta"]))  # noqa: S307 - our own repr of a plain dict
    return table, z["edges"].tolist(), meta


def _frac(to: int, current: int, pot_after_call: int) -> float:
    return max(to - current, 1) / max(pot_after_call, 1)


def abstract_state(state: DecisionState) -> hunl.State | None:
    """Replay the real hand's public actions into the abstract game (None if they don't fit)."""
    button = state.button
    s = hunl.State()
    street = Street.PREFLOP
    real_bets = [state.big_blind // 2, state.big_blind]  # current street, by abstract player
    real_pot = 0
    for a in state.actions:
        if a.street != street:
            real_pot += sum(real_bets)
            real_bets = [0, 0]
            street = a.street
            while s.street < a.street and not s.terminal():
                hunl._next_street(s)
        if s.terminal():
            return None
        p = 0 if a.seat == button else 1
        if s.to_act != p:
            return None
        legal = hunl.legal(s)
        if a.type == ActionType.FOLD:
            choice = "f"
        elif a.type == ActionType.CHECK_CALL:
            choice = "c"
        elif a.is_all_in and "a" in legal:
            choice = "a"
        else:
            sized = [x for x in legal if x in hunl.SIZES]
            if not sized:
                choice = "a" if "a" in legal else "c"
            else:
                current = max(real_bets)
                call = current - real_bets[p]
                f = _frac(a.to, current, real_pot + sum(real_bets) + call)
                choice = min(sized, key=lambda x: abs(math.log(hunl.SIZES[x] / f)))
        if choice not in legal:
            return None
        if a.type != ActionType.FOLD:
            real_bets[p] = a.to
        s = hunl.apply(s, choice)
    while s.street < state.street and not s.terminal():
        hunl._next_street(s)
    if s.terminal() or s.street != state.street:
        return None
    return s


MENU = "fc123a"  # fold, check/call, bet/raise 33%, 75%, 125% of the pot, all-in


class BlueprintPolicy:
    buyin_bb = 100.0

    def __init__(self, path: Path, seed: int) -> None:
        self.path = str(path)
        self.table, self.edges, self.meta = _load(self.path)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fallback = HandStrengthBot()
        self.lookups = 0
        self.misses = 0

    @classmethod
    def load(cls, path: Path, seed: int) -> BlueprintPolicy:
        return cls(path, seed)

    def act(self, state: DecisionState) -> Action:
        if state.n_players != 2:
            raise ValueError("the blueprint is heads-up only")
        self.lookups += 1
        s = abstract_state(state)
        if s is None or s.to_act != (0 if state.seat == state.button else 1):
            self.misses += 1
            return self.fallback.act(state)
        b = hunl.bucket(state.hole, state.board, int(state.street), self.edges)
        probs = self.table.get(hunl.infoset(s, b))
        acts = hunl.legal(s)
        if probs is None:
            self.misses += 1
            return self.fallback.act(state)
        p = probs[: len(acts)]
        p = p / p.sum() if p.sum() > 0 else np.full(len(acts), 1 / len(acts))
        return self._real(state, acts[int(self.rng.choice(len(acts), p=p))])

    def menu_distribution(self, state: DecisionState) -> np.ndarray:
        """The blueprint's probabilities over ``MENU`` (fold, check/call, bet 33/75/125%,
        all-in) at this decision; zeros for actions the abstraction doesn't allow here. Unseen
        spots give the hand-strength fallback's action as a one-hot. Does not touch the RNG,
        so ``act`` (the frozen baseline's play) is unchanged."""
        out = np.zeros(len(MENU))
        s = abstract_state(state)
        probs = None
        if s is not None and s.to_act == (0 if state.seat == state.button else 1):
            b = hunl.bucket(state.hole, state.board, int(state.street), self.edges)
            probs = self.table.get(hunl.infoset(s, b))
        if probs is None:
            a = self.fallback.act(state)
            if a.type == ActionType.FOLD:
                out[0] = 1.0
            elif a.type == ActionType.CHECK_CALL:
                out[1] = 1.0
            else:
                out[MENU.index("2")] = 1.0
            return out
        acts = hunl.legal(s)
        p = probs[: len(acts)]
        p = p / p.sum() if p.sum() > 0 else np.full(len(acts), 1 / len(acts))
        for code, prob in zip(acts, p, strict=True):
            out[MENU.index(code)] += prob
        return out

    @staticmethod
    def _real(state: DecisionState, a: str) -> Action:
        legal = state.legal
        if a == "f":
            return Action.fold() if legal.to_call > 0 else Action.check_call()
        if a == "c" or not legal.can_raise:
            return Action.check_call()
        if a == "a":
            return Action.raise_to(legal.max_raise_to)
        current = max(state.bets)
        to = current + hunl.SIZES[a] * (state.pot + legal.to_call)
        return Action.raise_to(int(min(max(to, legal.min_raise_to), legal.max_raise_to)))
