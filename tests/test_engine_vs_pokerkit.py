"""Cross-check the engine against pokerkit on randomized hands.

For each hand we play our engine with random policies, then replay the same cards and actions
in pokerkit. Both must agree on who acts at every step, accept the same actions, and produce
identical final stacks (side pots, split pots, odd chips, uncalled bets)."""

from __future__ import annotations

import numpy as np
import pokerkit as pk
import pytest

from pokerbot.engine.actions import ActionType
from pokerbot.engine.cards import card_str, cards_str, shuffled_deck
from pokerbot.engine.events import PlayerAction
from pokerbot.engine.policies import RandomPolicy
from pokerbot.engine.state import HandState

AUTOMATIONS = (
    pk.Automation.ANTE_POSTING,
    pk.Automation.BET_COLLECTION,
    pk.Automation.BLIND_OR_STRADDLE_POSTING,
    pk.Automation.CARD_BURNING,
    pk.Automation.RUNOUT_COUNT_SELECTION,
    pk.Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    pk.Automation.HAND_KILLING,
    pk.Automation.CHIPS_PUSHING,
    pk.Automation.CHIPS_PULLING,
)


def pk_index(seat: int, button: int, n: int) -> int:
    """pokerkit orders players SB, BB, ..., button (heads-up: BB, button)."""
    return (seat - button - 1) % n


def play_ours(rng: np.random.Generator) -> HandState:
    n = int(rng.integers(2, 7))
    stacks = [int(s) for s in rng.integers(1, 400, size=n)]
    if rng.random() < 0.5:  # many hands with equal deep stacks too
        stacks = [200] * n
    button = int(rng.integers(0, n))
    deck = shuffled_deck(rng)
    policy = RandomPolicy(rng, fold=float(rng.uniform(0, 0.4)), raise_=float(rng.uniform(0, 0.5)))
    state = HandState(stacks, button, deck)
    while not state.finished:
        state.apply(policy.act(state.decision_state()))
    return state


def replay_in_pokerkit(ours: HandState) -> list[int]:
    n, button = ours.n, ours.button
    order = sorted(range(n), key=lambda s: pk_index(s, button, n))  # pk index -> our seat
    game = pk.NoLimitTexasHoldem.create_state(
        AUTOMATIONS,
        True,
        0,
        (ours.small_blind, ours.big_blind),
        ours.big_blind,
        [ours.start_stacks[s] for s in order],
        n,
        mode=pk.Mode.CASH_GAME,
    )
    dealt = [0] * n
    while game.can_deal_hole():
        p = game.hole_dealee_index
        seat = order[p]
        game.deal_hole(card_str(ours.hole[seat][dealt[p]]))
        dealt[p] += 1

    board_dealt = 0

    def deal_boards() -> None:
        # Deal pending board cards. pokerkit also asks for meaningless checks from a player
        # who can't raise and faces no bet (everyone else all-in); our engine skips those.
        nonlocal board_dealt
        while True:
            if game.can_deal_board():
                k = 3 if board_dealt == 0 else 1
                game.deal_board(cards_str(ours.board[board_dealt : board_dealt + k]))
                board_dealt += k
            elif (
                game.actor_index is not None
                and game.checking_or_calling_amount == 0
                and not game.can_complete_bet_or_raise_to()
            ):
                game.check_or_call()
            else:
                return

    for action in (e for e in ours.events if isinstance(e, PlayerAction)):
        deal_boards()
        assert game.actor_index == pk_index(action.seat, button, n), "actor mismatch"
        if action.type == ActionType.FOLD:
            game.fold()
        elif action.type == ActionType.CHECK_CALL:
            game.check_or_call()
        else:
            game.complete_bet_or_raise_to(action.to)
    deal_boards()
    assert not game.status, "pokerkit hand still running"
    return [game.stacks[pk_index(s, button, n)] for s in range(n)]


@pytest.mark.parametrize("block", range(20))
def test_random_hands_match_pokerkit(block: int) -> None:
    rng = np.random.default_rng(1000 + block)
    for _ in range(500):
        ours = play_ours(rng)
        assert replay_in_pokerkit(ours) == ours.stacks
        assert sum(ours.payoffs) == 0
