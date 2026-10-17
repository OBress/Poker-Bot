"""Public event stream emitted while a hand is played.

Simulated hands and parsed real hand histories both produce these events, so stats and
classifiers consume one format. Events carry only public information, except ``HoleCards``
which is private and delivered only to the seat it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerbot.engine.actions import ActionType, Street

# Real hands from sites that didn't record stacks (all of iPoker in HandHQ) are replayed with this
# stack, in big blinds. Stack-dependent stats (depth splits) must ignore stacks at or above it.
UNKNOWN_STACK_BB = 10_000


@dataclass(frozen=True, slots=True)
class HandStart:
    hand_id: int
    button: int
    small_blind: int
    big_blind: int
    stacks: tuple[int, ...]  # at the start of the hand, before blinds


@dataclass(frozen=True, slots=True)
class BlindPosted:
    seat: int
    amount: int


@dataclass(frozen=True, slots=True)
class HoleCards:
    seat: int
    cards: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PlayerAction:
    seat: int
    street: Street
    type: ActionType
    amount: int  # chips added to the pot by this action
    to: int  # seat's total bet on this street after the action
    pot_before: int  # pot (all streets, including current bets) before the action
    to_call: int  # amount the seat faced before acting
    is_all_in: bool


@dataclass(frozen=True, slots=True)
class BoardDealt:
    street: Street
    board: tuple[int, ...]  # full board so far


@dataclass(frozen=True, slots=True)
class Showdown:
    seat: int
    cards: tuple[int, int]
    rank: int


@dataclass(frozen=True, slots=True)
class Muck:
    """A player at showdown gave up without showing (public: the cards stay hidden)."""

    seat: int


@dataclass(frozen=True, slots=True)
class HandEnd:
    hand_id: int
    payoffs: tuple[int, ...]  # net chips won/lost per seat
    showdown: bool
    # False for real hands whose result the source didn't record (e.g. iPoker showdowns with no
    # revealed cards): payoffs are then not meaningful and outcome stats must skip the hand.
    payoffs_known: bool = True


Event = HandStart | BlindPosted | HoleCards | PlayerAction | BoardDealt | Showdown | Muck | HandEnd
