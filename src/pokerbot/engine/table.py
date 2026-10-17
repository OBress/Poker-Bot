"""Running hands and sessions with policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from pokerbot.engine.actions import Action
from pokerbot.engine.cards import shuffled_deck
from pokerbot.engine.events import Event, HoleCards
from pokerbot.engine.state import DecisionState, HandState


class Policy(Protocol):
    def act(self, state: DecisionState) -> Action: ...


@runtime_checkable
class Observer(Protocol):
    def observe(self, event: Event) -> None: ...


@dataclass(slots=True)
class HandResult:
    payoffs: tuple[int, ...]
    events: list[Event]
    showdown: bool


def play_hand(
    policies: Sequence[Policy],
    stacks: Sequence[int],
    button: int,
    deck_seed: int,
    small_blind: int = 1,
    big_blind: int = 2,
    hand_id: int = 0,
    observers: Sequence[Observer] = (),
) -> HandResult:
    """Play one hand. Policies that implement ``observe`` see public events plus their own
    hole cards; ``observers`` see public events only."""
    state = HandState(stacks, button, shuffled_deck(deck_seed), small_blind, big_blind, hand_id)
    listeners = [(seat, p) for seat, p in enumerate(policies) if isinstance(p, Observer)]
    sent = 0

    def flush() -> None:
        nonlocal sent
        for event in state.events[sent:]:
            private = isinstance(event, HoleCards)
            for seat, p in listeners:
                if not private or event.seat == seat:
                    p.observe(event)
            if not private:
                for o in observers:
                    o.observe(event)
        sent = len(state.events)

    while not state.finished:
        flush()
        seat = state.actor
        assert seat is not None
        state.apply(policies[seat].act(state.decision_state()))
    flush()
    return HandResult(state.payoffs, state.events, state.showdown)


@dataclass(slots=True)
class SessionResult:
    hands: int
    big_blind: int
    totals: np.ndarray  # net chips per seat
    per_hand: np.ndarray = field(repr=False)  # shape (hands, seats)

    def bb_per_100(self) -> np.ndarray:
        return self.totals / self.big_blind / self.hands * 100


def play_session(
    policies: Sequence[Policy],
    n_hands: int,
    seed: int,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    observers: Sequence[Observer] = (),
) -> SessionResult:
    """Cash-game session: stacks reset to ``starting_stack`` every hand, button rotates.

    Deck seeds come from ``seed`` alone, so two sessions with the same seed deal identical
    cards to identical seats (the basis for duplicate matches)."""
    n = len(policies)
    deck_seeds = np.random.default_rng(seed).integers(0, 2**63 - 1, size=n_hands)
    per_hand = np.zeros((n_hands, n), dtype=np.int64)
    for h in range(n_hands):
        result = play_hand(
            policies,
            [starting_stack] * n,
            button=h % n,
            deck_seed=int(deck_seeds[h]),
            small_blind=small_blind,
            big_blind=big_blind,
            hand_id=h,
            observers=observers,
        )
        per_hand[h] = result.payoffs
    return SessionResult(n_hands, big_blind, per_hand.sum(axis=0), per_hand)
