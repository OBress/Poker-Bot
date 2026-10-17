"""Real-data replay and stat counting.

The round-trip test plays hands in our engine, writes them as PHH rows (the real-data format),
replays them through ``data.replay`` and checks the event streams and stats agree exactly. Scenario
tests pin down individual stat definitions."""

from __future__ import annotations

import numpy as np
import pokerkit as pk

from pokerbot.data.replay import replay
from pokerbot.engine import Action
from pokerbot.engine.actions import ActionType
from pokerbot.engine.cards import card_str, cards, cards_str, shuffled_deck
from pokerbot.engine.events import Muck, PlayerAction, Showdown
from pokerbot.engine.policies import RandomPolicy
from pokerbot.engine.state import HandState
from pokerbot.stats import hand_counts, position_name
from tests.test_engine_rules import stacked_deck

AUTOMATIONS = (
    pk.Automation.ANTE_POSTING,
    pk.Automation.BET_COLLECTION,
    pk.Automation.BLIND_OR_STRADDLE_POSTING,
    pk.Automation.CARD_BURNING,
    pk.Automation.RUNOUT_COUNT_SELECTION,
    pk.Automation.HAND_KILLING,
    pk.Automation.CHIPS_PUSHING,
    pk.Automation.CHIPS_PULLING,
)


def to_phh_row(ours: HandState) -> tuple[dict, list[int]]:
    """Write a finished engine hand as a HandHQ-style row. Returns (row, our seat per PHH index)."""
    n, button = ours.n, ours.button
    order = sorted(range(n), key=lambda s: (s - button - 1) % n)  # PHH index -> our seat
    index = {s: k for k, s in enumerate(order)}
    game = pk.NoLimitTexasHoldem.create_state(
        AUTOMATIONS, True, 0, (1, 2), 2, [ours.start_stacks[s] for s in order], n
    )
    log = [f"d dh p{k + 1} ????" for k in range(n)]
    for k in range(n):
        game.deal_hole(cards_str(ours.hole[order[k]]))
    shown = {e.seat for e in ours.events if isinstance(e, Showdown)}
    dealt = 0

    def advance() -> None:
        nonlocal dealt
        while True:
            if game.can_deal_board():
                k = 3 if dealt == 0 else 1
                new = cards_str(ours.board[dealt : dealt + k])
                game.deal_board(new)
                log.append(f"d db {new}")
                dealt += k
            elif (
                game.actor_index is not None
                and game.checking_or_calling_amount == 0
                and not game.can_complete_bet_or_raise_to()
            ):
                log.append(f"p{game.actor_index + 1} cc")
                game.check_or_call()
            elif game.showdown_index is not None:
                k = game.showdown_index
                seat = order[k]
                if seat in shown:
                    log.append(f"p{k + 1} sm {cards_str(ours.hole[seat])}")
                    game.show_or_muck_hole_cards(True)
                else:
                    log.append(f"p{k + 1} sm ????")
                    game.show_or_muck_hole_cards(False)
            else:
                return

    for a in (e for e in ours.events if isinstance(e, PlayerAction)):
        advance()
        k = index[a.seat]
        if a.type == ActionType.FOLD:
            log.append(f"p{k + 1} f")
            game.fold()
        elif a.type == ActionType.CHECK_CALL:
            log.append(f"p{k + 1} cc")
            game.check_or_call()
        else:
            log.append(f"p{k + 1} cbr {a.to / 100}")
            game.complete_bet_or_raise_to(a.to)
    advance()
    row = {
        "players": [f"player{s}" for s in order],
        "starting_stacks": [ours.start_stacks[s] / 100 for s in order],
        "blinds": [0.01, 0.02] + [0.0] * (n - 2),
        "antes": [0.0] * n,
        "min_bet": 0.02,
        "big_blind": 0.02,
        "actions": log,
    }
    return row, order


def test_engine_hands_round_trip_through_real_data_parser() -> None:
    rng = np.random.default_rng(5)
    for _ in range(600):
        n = int(rng.integers(2, 7))
        stacks = [200] * n if rng.random() < 0.6 else [int(s) for s in rng.integers(5, 400, n)]
        state = HandState(stacks, int(rng.integers(0, n)), shuffled_deck(rng))
        policy = RandomPolicy(
            rng, fold=float(rng.uniform(0, 0.4)), raise_=float(rng.uniform(0, 0.4))
        )
        while not state.finished:
            state.apply(policy.act(state.decision_state()))

        row, order = to_phh_row(state)
        real = replay(row)
        ours_actions = [
            (e.seat, e.street, e.type, e.amount, e.to, e.pot_before, e.to_call, e.is_all_in)
            for e in state.events
            if isinstance(e, PlayerAction)
        ]
        real_actions = [
            (order[e.seat], e.street, e.type, e.amount, e.to, e.pot_before, e.to_call, e.is_all_in)
            for e in real.events
            if isinstance(e, PlayerAction)
        ]
        assert real_actions == ours_actions
        real_payoffs = real.events[-1].payoffs
        assert tuple(real_payoffs[order.index(s)] for s in range(n)) == state.payoffs
        ours_counts = hand_counts(state.events)
        real_counts = {order[k]: c for k, c in hand_counts(real.events).items()}
        assert real_counts == ours_counts


# ---------------------------------------------------------------- scenarios

C, F, R = Action.check_call, Action.fold, Action.raise_to


def play(stacks: list[int], button: int, script: list[Action]) -> HandState:
    holes = ["AhAd", "KhKd", "QhQd", "JhJd", "ThTd", "9h9d"][: len(stacks)]
    state = HandState(stacks, button, stacked_deck(holes, "2c7s8c3dKs"))
    for a in script:
        state.apply(a)
    while not state.finished:  # check it down
        state.apply(C())
    return state


def test_positions_6max() -> None:
    names = [position_name(s, 0, 6) for s in range(6)]
    assert names == ["BTN", "SB", "BB", "EP", "MP", "CO"]
    assert [position_name(s, 0, 9) for s in range(9)] == [
        "BTN", "SB", "BB", "EP", "EP", "EP", "MP", "MP", "CO",
    ]  # fmt: skip
    assert [position_name(s, 0, 2) for s in range(2)] == ["BTN", "BB"]


def test_steal_three_bet_and_fold() -> None:
    # 6-max, button 0. Order: EP(3) MP(4) CO(5) BTN(0) SB(1) BB(2).
    s = play([200] * 6, 0, [F(), F(), R(5), F(), F(), R(18), F()])
    c = hand_counts(s.events)
    assert c[5]["rfi_CO"] == (1, 1) and c[5]["steal"] == (1, 1)
    assert c[5]["fold_to_3bet"] == (1, 1) and c[5]["vpip"] == (1, 1)
    assert c[3]["rfi_EP"] == (1, 0) and "steal" not in c[3]
    assert c[1]["fold_to_steal"] == (1, 1) and c[1]["3bet"] == (1, 0)
    assert c[2]["fold_to_steal"] == (1, 0) and c[2]["3bet"] == (1, 1)
    assert c[2]["pfr_BB"] == (1, 1) and c[1]["vpip_SB"] == (1, 0)
    assert "wtsd" not in c[2]  # no flop


def test_cbet_fold_to_cbet_and_check_raise() -> None:
    # Heads-up, button 0 raises, BB calls. Flop: BB checks, BTN c-bets, BB check-raises, BTN folds.
    s = play([200, 200], 0, [R(6), C(), C(), R(6), R(20), F()])
    c = hand_counts(s.events)
    assert c[0]["cbet_flop"] == (1, 1) and c[0]["cbet_flop_hu"] == (1, 1)
    assert c[0]["cbet_flop_ip"] == (1, 1)
    assert c[1]["fold_to_cbet_flop"] == (1, 0) and c[1]["xr_flop"] == (1, 1)
    assert c[1]["donk_flop"] == (1, 0)  # checked instead of leading
    assert c[0]["agg_flop"] == (1, 1) and c[1]["agg_flop"] == (1, 0)
    assert c[0]["betsize_b2"] == (1, 1)  # 6 into 12 = 0.5 pot -> bucket 2 = [0.4, 0.6)
    assert c[0]["wtsd"] == (1, 0)


def test_limped_pot_showdown() -> None:
    # 3-handed, button 0: BTN limps, SB completes, BB checks, checked to showdown.
    s = play([200] * 3, 0, [C(), C(), C()])
    c = hand_counts(s.events)
    assert c[0]["limp"] == (1, 1) and c[0]["rfi_BTN"] == (1, 0) and c[0]["steal"] == (1, 0)
    assert c[1]["limp"] == (1, 1) and "rfi" not in c[1]
    assert c[2]["vpip"] == (1, 0)
    assert all(c[k]["wtsd"] == (1, 1) for k in range(3))
    winner = max(range(3), key=lambda k: s.payoffs[k])
    assert c[winner]["wsd"] == (1, 1)
    assert "cbet_flop" not in c[0]  # no preflop aggressor


def test_real_hand_row() -> None:
    row = {
        "players": ["a", "b", "c"],
        "starting_stacks": [247.15, 237.05, 47.90],
        "blinds": [1.0, 2.0, 0.0],
        "antes": [0.0, 0.0, 0.0],
        "min_bet": 2.0,
        "big_blind": 2.0,
        "actions": [
            "d dh p1 ????", "d dh p2 ????", "d dh p3 ????", "p3 cc", "p1 cc", "p2 cc",
            "d db 6hTdKd", "p1 cc", "p2 cbr 4", "p3 f", "p1 f",
        ],
    }  # fmt: skip
    hand = replay(row)
    assert hand.events[-1].payoffs == (-200, 400, -200)  # cents
    c = hand_counts(hand.events)
    assert c[2]["limp"] == (1, 1) and c[2]["rfi_BTN"] == (1, 0)
    assert c[1]["agg_flop"] == (1, 1) and c[1]["betsize_b3"] == (1, 1)  # 4 into 6
    assert c[0]["wtsd"] == (1, 0)


def test_showdown_reveals_parse() -> None:
    row = {
        "players": ["a", "b"],
        "starting_stacks": [100.0, 100.0],
        "blinds": [1.0, 2.0],
        "antes": [0.0, 0.0],
        "min_bet": 2.0,
        "big_blind": 2.0,
        "actions": [
            "d dh p1 ????", "d dh p2 ????", "p2 cc", "p1 cc", "d db 2c7s8c", "p1 cc", "p2 cc",
            "d db 3d", "p1 cc", "p2 cc", "d db Ks", "p1 cc", "p2 cc", "p1 sm AhAd", "p2 sm ????",
        ],
    }  # fmt: skip
    hand = replay(row)
    shows = [e for e in hand.events if isinstance(e, Showdown)]
    assert [(e.seat, e.cards) for e in shows] == [(0, cards("AhAd"))]
    assert any(isinstance(e, Muck) and e.seat == 1 for e in hand.events)
    assert card_str(shows[0].cards[0]) == "Ah"


def test_unknown_stacks_replay_without_depth_stats() -> None:
    row = {
        "players": ["a", "b"],
        "starting_stacks": [float("inf"), float("inf")],
        "blinds": [0.5, 1.0],
        "antes": [0.0, 0.0],
        "min_bet": 1.0,
        "big_blind": 1.0,
        "actions": ["d dh p1 AcQc", "d dh p2 ????", "p2 cbr 3", "p1 f"],
    }
    hand = replay(row)
    assert hand.events[-1].payoffs == (-100, 100)
    c = hand_counts(hand.events)
    assert c[1]["pfr"] == (1, 1) and c[1]["pfr_BTN"] == (1, 1)
    assert not any(k.startswith(("vpip_d", "pfr_d")) for k in c[0] | c[1])


def test_unrecorded_showdowns_count_for_wtsd_but_not_wsd() -> None:
    base = {
        "players": ["a", "b"],
        "starting_stacks": [float("inf"), float("inf")],
        "blinds": [1.0, 2.0],
        "antes": [0.0, 0.0],
        "min_bet": 2.0,
        "big_blind": 2.0,
    }
    # iPoker style: both hands unrevealed at showdown, no winnings recorded.
    hidden = replay(base | {"actions": [
        "d dh p1 ????", "d dh p2 ????", "p2 cc", "p1 cc", "d db 2c7s8c", "p1 cc", "p2 cc",
        "d db 3d", "p1 cc", "p2 cc", "d db Ks", "p1 cc", "p2 cc", "p1 sm ????", "p2 sm ????",
    ]})  # fmt: skip
    assert not hidden.events[-1].payoffs_known
    c = hand_counts(hidden.events)
    assert c[0]["wtsd"] == (1, 1) and c[1]["wtsd"] == (1, 1)
    assert "wsd" not in c[0] and "wsd" not in c[1]
    # All-in with no showdown records at all.
    allin = replay(base | {"starting_stacks": [100.0, 100.0], "actions": [
        "d dh p1 ????", "d dh p2 ????", "p2 cc", "p1 cc", "d db 2c7s8c", "p1 cbr 98", "p2 cc",
        "d db 3d", "d db Ks",
    ]})  # fmt: skip
    c = hand_counts(allin.events)
    assert c[0]["wtsd"] == (1, 1) and "wsd" not in c[0]
    # A revealed winner keeps W$SD.
    shown = replay(base | {"starting_stacks": [100.0, 100.0], "actions": [
        "d dh p1 ????", "d dh p2 ????", "p2 cc", "p1 cc", "d db 2c7s8c", "p1 cc", "p2 cc",
        "d db 3d", "p1 cc", "p2 cc", "d db Ks", "p1 cc", "p2 cc", "p1 sm AhAd", "p2 sm ????",
    ]})  # fmt: skip
    assert shown.events[-1].payoffs_known
    assert hand_counts(shown.events)[0]["wsd"] == (1, 1)
