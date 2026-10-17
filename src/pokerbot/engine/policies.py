"""Trivial policies for tests, benchmarks and smoke runs. Real opponents live in ``opponents/``."""

from __future__ import annotations

import numpy as np

from pokerbot.engine.actions import Action
from pokerbot.engine.state import DecisionState


class CallingStation:
    def act(self, state: DecisionState) -> Action:
        return Action.check_call()


class RandomPolicy:
    """Folds, calls or raises at fixed rates; raise size uniform over the legal range."""

    def __init__(
        self, seed: int | np.random.Generator, fold: float = 0.2, raise_: float = 0.3
    ) -> None:
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.fold = fold
        self.raise_ = raise_

    def act(self, state: DecisionState) -> Action:
        legal = state.legal
        u = self.rng.random()
        if u < self.fold and legal.to_call > 0:
            return Action.fold()
        if u < self.fold + self.raise_ and legal.can_raise:
            return Action.raise_to(
                int(self.rng.integers(legal.min_raise_to, legal.max_raise_to + 1))
            )
        return Action.check_call()
