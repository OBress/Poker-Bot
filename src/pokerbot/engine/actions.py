"""Actions a player can take."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionType(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2  # amount is the total bet for this street ("raise to")


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass(frozen=True, slots=True)
class Action:
    type: ActionType
    amount: int = 0  # only used by BET_RAISE

    @staticmethod
    def fold() -> Action:
        return _FOLD

    @staticmethod
    def check_call() -> Action:
        return _CHECK_CALL

    @staticmethod
    def raise_to(amount: int) -> Action:
        return Action(ActionType.BET_RAISE, int(amount))


_FOLD = Action(ActionType.FOLD)
_CHECK_CALL = Action(ActionType.CHECK_CALL)
