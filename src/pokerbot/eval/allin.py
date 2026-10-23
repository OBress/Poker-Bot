"""All-in luck removal (MATH.md §7.2).

When the betting is over before the river (two or more players left, at most one with chips
behind), the cards still to come decide the hand but no decision does. The realized payoffs are
replaced by their expectation over the remaining board: exact enumeration for flop and turn
all-ins, a seeded Monte Carlo sample for preflop all-ins (C(48, 5) runouts is too many).
Pots are rebuilt from each seat's total contribution, exactly as the engine builds side pots;
split pots are shared fractionally, which is the expectation of the odd-chip rule.

This reads hole cards from the full event stream, so it is evaluator-only code.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
from phevaluator import evaluate_cards

from pokerbot.engine.actions import ActionType
from pokerbot.engine.events import (
    BlindPosted,
    BoardDealt,
    Event,
    HandEnd,
    HandStart,
    HoleCards,
    PlayerAction,
)

PREFLOP_SAMPLES = 600


def _settle(contrib: Sequence[int], live: Sequence[int], ranks: dict[int, int]) -> np.ndarray:
    """Expected chips won per seat (not net) given final ranks (lower is better)."""
    n = len(contrib)
    won = np.zeros(n)
    prev = 0
    carry = 0.0
    for level in sorted({c for c in contrib if c > 0}):
        amount = sum(min(c, level) - min(c, prev) for c in contrib) + carry
        eligible = [i for i in live if contrib[i] >= level]
        if eligible:
            best = min(ranks[i] for i in eligible)
            winners = [i for i in eligible if ranks[i] == best]
            for w in winners:
                won[w] += amount / len(winners)
            carry = 0.0
        else:
            carry = amount
        prev = level
    return won


def allin_adjusted(events: Sequence[Event], seed: int = 0) -> np.ndarray | None:
    """Expected net payoff per seat if the hand was decided by an all-in runout before the
    river; None when no adjustment applies (the realized payoff stands)."""
    start = next(e for e in events if isinstance(e, HandStart))
    n = len(start.stacks)
    end = next(e for e in events if isinstance(e, HandEnd))
    if not end.showdown:
        return None
    actions = [e for e in events if isinstance(e, PlayerAction)]
    boards = [e for e in events if isinstance(e, BoardDealt)]
    last_action_street = max((a.street for a in actions), default=0)
    board_at_end_of_betting = ()
    for b in boards:
        if b.street <= last_action_street:
            board_at_end_of_betting = b.board
    final_board = boards[-1].board if boards else ()
    if len(board_at_end_of_betting) >= 5 or len(final_board) < 5:
        return None
    if len(board_at_end_of_betting) == len(final_board):
        return None
    contrib = [0] * n
    for e in events:
        if isinstance(e, (BlindPosted, PlayerAction)):
            contrib[e.seat] += e.amount
    folded = {a.seat for a in actions if a.type == ActionType.FOLD}
    hole = {e.seat: e.cards for e in events if isinstance(e, HoleCards)}
    live = [s for s in range(n) if s not in folded and contrib[s] > 0]
    if len(live) < 2 or any(s not in hole for s in live):
        return None
    known = list(board_at_end_of_betting)
    used = set(known) | {c for s in live for c in hole[s]}
    deck = [c for c in range(52) if c not in used]
    need = 5 - len(known)
    if need >= 3:
        rng = np.random.default_rng(seed)
        runouts = [
            rng.choice(deck, size=need, replace=False).tolist() for _ in range(PREFLOP_SAMPLES)
        ]
    else:
        runouts = [list(c) for c in combinations(deck, need)]
    won = np.zeros(n)
    for r in runouts:
        board = known + r
        ranks = {s: evaluate_cards(*hole[s], *board) for s in live}
        won += _settle(contrib, live, ranks)
    return won / len(runouts) - np.array(contrib, dtype=float)
