"""Shown-hand strength by betting line (MATH.md §4.5), from an event stream.

For every hand shown at a showdown on a complete board we record the player's postflop *line*
and the shown hand's *strength bucket*. Works on simulated hands and on real hands replayed by
``data.replay`` (both emit the same events), so real and simulated tables are built by one function.

Strength: exact equity vs a uniformly random hand on the final board (all 990 opponent combos,
ties count half), bucketed at 0.35, 0.55, 0.75, 0.9 into h = 0..4 (0 = weakest).

Line: one token per street, from the player's own actions on it:
  B  bet or raised     C  called a bet (never raised)     X  only checked     -  no decision
and the three tokens (flop, turn, river) map to one of ``LINES``:
  bbb          B on every street ("bet-bet-bet")
  river_bet    bet or raised the river, not bbb
  call_down    called the river after calling an earlier street, never bet ("check-call x2+")
  river_call   called the river otherwise (after betting earlier, or after only checking)
  bet_check    checked the river after betting an earlier street
  call_check   checked the river after calling an earlier street, never bet
  check_down   checked every street it acted on
  all_in       no river decision (all-in earlier, including preflop)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from phevaluator import evaluate_cards

from pokerbot.engine.actions import ActionType, Street
from pokerbot.engine.events import BoardDealt, Event, PlayerAction, Showdown

STRENGTH_EDGES = (0.35, 0.55, 0.75, 0.9)
BUCKETS = len(STRENGTH_EDGES) + 1
LINES = (
    "bbb",
    "river_bet",
    "call_down",
    "river_call",
    "bet_check",
    "call_check",
    "check_down",
    "all_in",
)
_POSTFLOP = (Street.FLOP, Street.TURN, Street.RIVER)


@dataclass(frozen=True, slots=True)
class ShownHand:
    seat: int
    line: str
    bucket: int
    equity: float


def board_ranks(board: Sequence[int]) -> dict[tuple[int, int], int]:
    """Rank (lower = better) of every two-card holding on a complete board."""
    deck = [c for c in range(52) if c not in board]
    return {(a, b): evaluate_cards(a, b, *board) for a, b in combinations(deck, 2)}


def river_equity(
    hole: tuple[int, int], board: Sequence[int], ranks: dict[tuple[int, int], int] | None = None
) -> float:
    """Exact equity of ``hole`` vs a uniformly random hand on a complete 5-card board."""
    ranks = ranks if ranks is not None else board_ranks(board)
    me = evaluate_cards(*hole, *board)
    used = set(hole)
    score = total = 0
    for (a, b), r in ranks.items():
        if a in used or b in used:
            continue
        total += 1
        score += 2 if me < r else 1 if me == r else 0
    return score / (2 * total)


def strength_bucket(equity: float) -> int:
    for i, e in enumerate(STRENGTH_EDGES):
        if equity < e:
            return i
    return len(STRENGTH_EDGES)


def street_tokens(actions: Sequence[PlayerAction], seat: int) -> str:
    out = []
    for street in _POSTFLOP:
        mine = [a for a in actions if a.seat == seat and a.street == street]
        if any(a.type == ActionType.BET_RAISE for a in mine):
            out.append("B")
        elif any(a.type == ActionType.CHECK_CALL and a.to_call > 0 for a in mine):
            out.append("C")
        elif mine:
            out.append("X")
        else:
            out.append("-")
    return "".join(out)


def line_class(tokens: str) -> str:
    early, river = tokens[:2], tokens[2]
    if river == "-":
        return "all_in"
    if tokens == "BBB":
        return "bbb"
    if river == "B":
        return "river_bet"
    if river == "C":
        return "call_down" if "C" in early and "B" not in early else "river_call"
    if "B" in early:
        return "bet_check"
    return "call_check" if "C" in early else "check_down"


def shown_hands(events: Sequence[Event], board: Sequence[int] | None = None) -> list[ShownHand]:
    """Every hand shown at showdown on a complete board, with its line and strength."""
    shows = [e for e in events if isinstance(e, Showdown)]
    if not shows:
        return []
    if board is None:
        boards = [e.board for e in events if isinstance(e, BoardDealt)]
        board = boards[-1] if boards else ()
    if len(board) != 5:
        return []
    actions = [e for e in events if isinstance(e, PlayerAction)]
    ranks = board_ranks(board)
    out = []
    for s in shows:
        eq = river_equity(s.cards, board, ranks)
        line = line_class(street_tokens(actions, s.seat))
        out.append(ShownHand(s.seat, line, strength_bucket(eq), eq))
    return out
