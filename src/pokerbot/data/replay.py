"""Turn real PHH hands into the engine's event stream by replaying them in pokerkit.

pokerkit is the rules reference (dead blinds, straddles, short all-ins, uncalled bets),
so we let it play each hand and read the state before and after every action. The result uses the
same event classes as simulated hands, so stats and classifiers run unchanged on real data.

Units: amounts are integer **cents** of the original currency (exact for these sites); divide by
``HandStart.big_blind`` for big blinds. Seats follow PHH order: seat 0 is the first player left
of the button (small blind; big blind heads-up) and the button is the last seat.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import pokerkit as pk

from pokerbot.engine.actions import ActionType, Street
from pokerbot.engine.cards import cards as parse_cards
from pokerbot.engine.cards import hand_rank
from pokerbot.engine.events import (
    UNKNOWN_STACK_BB,
    BlindPosted,
    BoardDealt,
    Event,
    HandEnd,
    HandStart,
    Muck,
    PlayerAction,
    Showdown,
)

_BET = re.compile(r"^(p\d+ cbr )([\d.]+)$")
_STREET_AFTER_BOARD = {3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}


@dataclass(slots=True)
class RealHand:
    """A replayed real hand: events plus the players' ids (index = seat)."""

    players: list[str]
    events: list[Event]


def _cents(x: float) -> int:
    return int(round(x * 100))


def _stacks(row: dict) -> list[int]:
    """Starting stacks in cents; unrecorded (infinite) stacks become UNKNOWN_STACK_BB."""
    unknown = UNKNOWN_STACK_BB * _cents(row["big_blind"])
    return [unknown if math.isinf(x) else _cents(x) for x in row["starting_stacks"]]


def _history(row: dict) -> pk.HandHistory:
    actions = [
        (m[1] + str(_cents(float(m[2])))) if (m := _BET.match(a)) else a for a in row["actions"]
    ]
    return pk.HandHistory(
        variant="NT",
        antes=[_cents(x) for x in row["antes"]],
        blinds_or_straddles=[_cents(x) for x in row["blinds"]],
        min_bet=_cents(row["min_bet"]),
        starting_stacks=_stacks(row),
        actions=actions,
    )


def replay(row: dict, hand_id: int = 0) -> RealHand:
    """Replay one converted HandHQ row (see ``handhq.SCHEMA``). Raises if pokerkit rejects it."""
    n = len(row["players"])
    start = _stacks(row)
    blinds = [_cents(x) for x in row["blinds"]]
    events: list[Event] = [
        HandStart(
            hand_id, n - 1, blinds[0] if blinds else 0, _cents(row["big_blind"]), tuple(start)
        )
    ]
    board: list[int] = []
    street = Street.PREFLOP
    prev_stacks: list[int] | None = None
    prev_bets: list[int] | None = None
    showdown = False
    folded: set[int] = set()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for state, action in _history(row).state_actions:
            stacks, bets = list(state.stacks), list(state.bets)
            if action is None:  # initial state: blinds/antes posted
                for seat, b in enumerate(bets):
                    if b:
                        events.append(BlindPosted(seat, b))
            else:
                parts = action.split()
                if parts[0] == "d" and parts[1] == "db":
                    board.extend(parse_cards(parts[2]))
                    street = _STREET_AFTER_BOARD[len(board)]
                    events.append(BoardDealt(street, tuple(board)))
                elif parts[0].startswith("p"):
                    seat = int(parts[0][1:]) - 1
                    assert prev_stacks is not None and prev_bets is not None
                    kind = parts[1]
                    if kind == "sm":
                        showdown = True
                        shown = parts[2] if len(parts) > 2 else "????"
                        if "?" in shown:
                            events.append(Muck(seat))
                        else:
                            hole = parse_cards(shown)
                            rank = hand_rank(hole, tuple(board)) if len(board) == 5 else -1
                            events.append(Showdown(seat, (hole[0], hole[1]), rank))
                    elif kind in ("f", "cc", "cbr"):
                        added = prev_stacks[seat] - stacks[seat]
                        to_call = min(max(prev_bets) - prev_bets[seat], prev_stacks[seat])
                        if kind == "f":
                            folded.add(seat)
                        elif (
                            kind == "cc"
                            and to_call == 0
                            and not any(
                                prev_stacks[j] > 0
                                for j in range(n)
                                if j != seat and j not in folded
                            )
                        ):
                            # pokerkit-only no-op check (nobody left who could bet): not a decision
                            prev_stacks, prev_bets = stacks, bets
                            continue
                        events.append(
                            PlayerAction(
                                seat=seat,
                                street=street,
                                type={
                                    "f": ActionType.FOLD,
                                    "cc": ActionType.CHECK_CALL,
                                    "cbr": ActionType.BET_RAISE,
                                }[kind],
                                amount=added,
                                to=prev_bets[seat] + added,
                                pot_before=sum(start) - sum(prev_stacks),
                                to_call=to_call,
                                is_all_in=kind != "f" and stacks[seat] == 0,
                            )
                        )
            prev_stacks, prev_bets = stacks, bets
        final = list(state.stacks)

    events.append(
        HandEnd(hand_id, tuple(f - s for f, s in zip(final, start, strict=True)), showdown)
    )
    return RealHand([str(p) for p in row["players"]], events)


def replay_rows(rows: list[dict]) -> Iterator[RealHand | None]:
    """Replay many rows; yields None for a hand pokerkit rejects (counted by the caller)."""
    for i, row in enumerate(rows):
        try:
            yield replay(row, i)
        except Exception:  # noqa: BLE001 - malformed source hands are skipped, not fatal
            yield None
