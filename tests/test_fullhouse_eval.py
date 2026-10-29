"""Fullhouse bot adapter and the duplicate table runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokerbot.engine.actions import ActionType
from pokerbot.engine.cards import card_str
from pokerbot.engine.state import HandState
from pokerbot.engine.table import play_hand
from pokerbot.eval.players import make_policy
from pokerbot.eval.tables import TableFormat, play_table, run_duplicate, seatings, summarize
from pokerbot.opponents.fullhouse import REFERENCE_BOTS, FullhouseBot, to_action

POOL = [f"fullhouse:{b}" for b in REFERENCE_BOTS]

RECORDER = """
STATES = []
REPLY = {"action": "call"}

def decide(game_state):
    STATES.append(game_state)
    if isinstance(REPLY, Exception):
        raise REPLY
    return REPLY
"""


@pytest.fixture
def recorder(tmp_path: Path) -> Path:
    path = tmp_path / "recorder.py"
    path.write_text(RECORDER, encoding="utf-8")
    return path


def _decision(stacks=(10_000,) * 3, button=0, actions=()):
    """A HandState after applying ``actions``, at the next decision."""
    state = HandState(list(stacks), button, list(range(52)), 50, 100)
    for a in actions:
        state.apply(a)
    return state


def test_game_state_matches_their_format_and_hides_cards(recorder: Path) -> None:
    bots = [FullhouseBot(recorder, seed=k) for k in range(3)]
    play_hand(bots, [10_000] * 3, button=0, deck_seed=7, small_blind=50, big_blind=100)
    states = bots[0].module.STATES
    assert states, "every seat acts at least once preflop"
    first = states[0]
    # 3-handed, button 0: seat 1 posts SB, seat 2 BB, seat 0 (button) opens.
    assert first["seat_to_act"] == 0 and first["street"] == "preflop"
    assert first["action_log"][:2] == [
        {"seat": 1, "action": "small_blind", "amount": 50},
        {"seat": 2, "action": "big_blind", "amount": 100},
    ]
    assert first["pot"] == 150 and first["current_bet"] == 100
    assert first["amount_owed"] == 100 and not first["can_check"]
    assert first["min_raise_to"] == 200
    assert first["your_stack"] == 10_000 and first["your_bet_this_street"] == 0
    assert len(first["your_cards"]) == 2
    for s in states:
        for p in s["players"]:
            assert p["hole_cards"] is None
        assert set(s) >= {"hand_id", "community_cards", "players", "action_log"}


def test_bot_sees_only_its_own_hole_cards(recorder: Path) -> None:
    bots = [FullhouseBot(recorder, seed=k) for k in range(2)]
    result = play_hand(bots, [10_000] * 2, button=0, deck_seed=3, small_blind=50, big_blind=100)
    from pokerbot.engine.events import HoleCards

    hole = {e.seat: e.cards for e in result.events if isinstance(e, HoleCards)}
    for seat, bot in enumerate(bots):
        for s in bot.module.STATES:
            assert s["your_cards"] == [card_str(c) for c in hole[seat]]


def test_replies_follow_their_validator() -> None:
    facing = _decision().decision_state()  # seat 0 facing the big blind
    assert facing.legal.to_call == 100
    assert to_action({"action": "check"}, facing).type == ActionType.CHECK_CALL
    assert to_action({"action": "call"}, facing).type == ActionType.CHECK_CALL
    assert to_action({"action": "raise", "amount": 120}, facing).amount == 200  # snaps up
    assert to_action({"action": "raise", "amount": 10**9}, facing).amount == 10_000
    assert to_action({"action": "all_in"}, facing).amount == 10_000
    assert to_action({"action": "raise", "amount": "lots"}, facing).amount == 200
    assert to_action({"action": "bluff"}, facing).type == ActionType.FOLD
    assert to_action({}, facing).type == ActionType.FOLD


def test_crashing_bot_folds_and_counts_errors(recorder: Path) -> None:
    bot = FullhouseBot(recorder, seed=0)
    bot.module.REPLY = RuntimeError("boom")
    assert bot.act(_decision().decision_state()).type == ActionType.FOLD
    bot.module.REPLY = "call"  # not a dict
    assert bot.act(_decision().decision_state()).type == ActionType.FOLD
    assert bot.errors == 2


def test_reference_bots_are_seeded_and_isolated() -> None:
    def actions(seed: int) -> list:
        bots = [make_policy("fullhouse:aggressor", seed + k) for k in range(3)]
        out = []
        for h in range(30):
            r = play_hand(bots, [10_000] * 3, button=h % 3, deck_seed=h, small_blind=50,
                          big_blind=100)  # fmt: skip
            out.append(r.events)  # raise sizes are random, so events differ by seed
        return out

    assert actions(1) == actions(1)
    assert actions(1) != actions(2)
    a, b = make_policy("fullhouse:aggressor", 5), make_policy("fullhouse:aggressor", 5)
    assert a.module is not b.module and a.module.random is not b.module.random


def test_match_mode_carries_stacks_and_removes_busted_seats() -> None:
    fmt = TableFormat("match", hands=400)
    table = [make_policy(s, k) for k, s in enumerate(["fullhouse:aggressor"] * 3 + POOL[:3])]
    per_hand = play_table(table, fmt, deck_seed=11)
    assert (per_hand.sum(axis=1) == 0).all()  # chips are conserved every hand
    running = fmt.stack + np.cumsum(per_hand, axis=0)
    assert (running >= 0).all()
    busted = np.argwhere(running == 0)
    assert len(busted), "three aggressors should bust someone in 400 hands"
    for h, seat in busted:
        assert (per_hand[h + 1 :, seat] == 0).all()


def test_cash_mode_resets_stacks() -> None:
    fmt = TableFormat("cash", hands=200)
    table = [make_policy(s, k) for k, s in enumerate(["fullhouse:aggressor"] * 6)]
    per_hand = play_table(table, fmt, deck_seed=2)
    assert (per_hand >= -fmt.stack).all() and (per_hand.sum(axis=1) == 0).all()


def test_seatings_rotate_hero_and_are_reproducible() -> None:
    plans = seatings(POOL, seats=6, sessions=12, seed=4)
    assert [p.hero_seat for p in plans] == [s % 6 for s in range(12)]
    assert all(sorted(p.opponents) == sorted(POOL) for p in plans)  # pool of 5 fills 5 seats
    assert plans == seatings(POOL, seats=6, sessions=12, seed=4)
    assert plans != seatings(POOL, seats=6, sessions=12, seed=5)


@pytest.mark.parametrize("mode", ["cash", "match"])
def test_duplicate_of_a_player_against_itself_is_exactly_zero(mode: str) -> None:
    fmt = TableFormat(mode, hands=60)
    results = run_duplicate("fullhouse:aggressor", "fullhouse:aggressor", POOL, fmt, 6, seed=0)
    s = summarize(results, fmt)
    assert s["delta"].bb100 == 0 and s["delta"].ci == 0
    assert s["hero"].bb100 == s["reference"].bb100
    assert s["hands"] == 6 * 60


def test_duplicate_detects_a_real_difference() -> None:
    fmt = TableFormat("cash", hands=200)
    results = run_duplicate("style:tag", "calling-station", POOL, fmt, 12, seed=0)
    s = summarize(results, fmt)
    assert s["delta"].bb100 != 0 and s["delta"].ci > 0
    assert np.isfinite(s["sd_d_per100"])


def test_allin_adjustment_conserves_chips_and_cuts_variance() -> None:
    table = [make_policy(s, k) for k, s in enumerate(["fullhouse:aggressor"] * 6)]
    raw = play_table(table, TableFormat("cash", hands=300), deck_seed=5)
    adj = play_table(table, TableFormat("cash", hands=300, allin=True), deck_seed=5)
    assert np.allclose(adj.sum(axis=1), 0)
    assert not np.allclose(raw, adj)
    assert adj.std() < raw.std()
