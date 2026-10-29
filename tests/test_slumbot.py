"""Slumbot client and replay, against an in-process stand-in that speaks the same protocol."""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import card_str, cards, shuffled_deck
from pokerbot.engine.events import HandEnd, HandStart, HoleCards
from pokerbot.engine.state import HandState
from pokerbot.eval.players import make_policy
from pokerbot.eval.slumbot import (
    BIG_BLIND,
    SMALL_BLIND,
    STACK,
    ProtocolError,
    SlumbotClient,
    play_hand,
    play_session,
    replay,
    summarize,
    to_incr,
    tokens,
)


class FakeSlumbot:
    """Deals with our engine and answers like slumbot.com/slumbot/api: the client is seat 0,
    positions alternate starting as the big blind, the opponent is any of our policies."""

    def __init__(self, opponent="fullhouse:aggressor", seed=0, reject_bets=False):
        self.opponent = make_policy(opponent, seed)
        self.seed = seed
        self.reject_bets = reject_bets
        self.hands = 0
        self.state: HandState | None = None
        self.pos = 0
        self.rejections = 0

    def post(self, path: str, body: dict) -> dict:
        if path.endswith("/new_hand"):
            self.pos = self.hands % 2  # 0 = the client is the big blind
            deck = shuffled_deck(self.seed * 1000 + self.hands)
            self.state = HandState(
                [STACK, STACK], 1 if self.pos == 0 else 0, deck, SMALL_BLIND, BIG_BLIND, self.hands
            )
            self.hands += 1
        else:
            assert self.state is not None and self.state.actor == 0, "not the client's turn"
            incr = body["incr"]
            if incr[0] == "b" and self.reject_bets:
                self.rejections += 1
                return {"error_msg": "Bet too small"}
            self.state.apply(_action(incr))
        while not self.state.finished and self.state.actor == 1:
            self.state.apply(self.opponent.act(self.state.decision_state()))
        return self._response()

    def _response(self) -> dict:
        s = self.state
        out = {
            "token": "fake-token",
            "action": _action_string(s),
            "client_pos": self.pos,
            "hole_cards": [card_str(c) for c in s.hole[0]],
            "board": [card_str(c) for c in s.board],
        }
        if s.finished:
            out["winnings"] = s.payoffs[0]
            if s.showdown:
                out["bot_hole_cards"] = [card_str(c) for c in s.hole[1]]
        return out


def _action(incr: str) -> Action:
    if incr == "f":
        return Action.fold()
    if incr in ("c", "k"):
        return Action.check_call()
    return Action.raise_to(int(incr[1:]))


def _action_string(s: HandState) -> str:
    out, street = [], 0
    for a in s.actions:
        if int(a.street) != street:
            out.append("/" * (int(a.street) - street))
            street = int(a.street)
        if a.type == ActionType.FOLD:
            out.append("f")
        elif a.type == ActionType.CHECK_CALL:
            out.append("c" if a.to_call > 0 else "k")
        else:
            out.append(f"b{a.to}")
    return "".join(out)


def _client(fake: FakeSlumbot) -> SlumbotClient:
    return SlumbotClient(post=fake.post)


class Recorder:
    """Min-raises when it can; remembers every event it observes."""

    def __init__(self) -> None:
        self.events: list = []

    def act(self, state):
        if state.legal.can_raise:
            return Action.raise_to(state.legal.min_raise_to)
        return Action.check_call()

    def observe(self, event) -> None:
        self.events.append(event)


def test_tokens_ignore_street_separators_and_reject_junk() -> None:
    assert tokens("b200c/kk/kb400") == ["b200", "c", "k", "k", "k", "b400"]
    assert tokens("b20000c///") == ["b20000", "c"]
    assert tokens("") == []
    with pytest.raises(ProtocolError):
        tokens("b200x")


def test_replay_puts_us_to_act_with_the_right_state() -> None:
    hero = cards("AcKd")
    # We are the big blind (pos 0): the small blind opens to 300, we face 200 more.
    s = replay("b300", hero, (), client_pos=0)
    assert s.actor == 0 and s.button == 1
    d = s.decision_state()
    assert d.hole == hero and d.pot == 400 and d.legal.to_call == 200
    assert d.stacks == (STACK - 100, STACK - 300) and d.big_blind == BIG_BLIND
    # Small blind (pos 1) acts first preflop; postflop the big blind acts first.
    assert replay("", hero, (), client_pos=1).actor == 0
    flop = cards("2c7h9s")
    assert replay("b300c/", hero, flop, client_pos=1).actor == 1  # we are the button
    assert replay("b300c/k", hero, flop, client_pos=1).actor == 0
    assert replay("b20000c///", hero, cards("2c7h9sJdQd"), 1, cards("AsAd")).finished


def test_to_incr_clamps_to_legal_amounts() -> None:
    d = replay("b300", cards("AcKd"), (), client_pos=0).decision_state()
    assert to_incr(Action.fold(), d) == "f"
    assert to_incr(Action.check_call(), d) == "c"
    assert to_incr(Action.raise_to(1), d) == f"b{d.legal.min_raise_to}"
    assert to_incr(Action.raise_to(10**9), d) == f"b{STACK}"
    free = replay("c", cards("AcKd"), (), client_pos=0).decision_state()  # big blind's option
    assert to_incr(Action.fold(), free) == "k" and to_incr(Action.check_call(), free) == "k"


def test_session_matches_the_server_chip_for_chip() -> None:
    fake = FakeSlumbot()
    hero = make_policy("fullhouse:shark", 1)
    records = play_session(hero, 80, _client(fake))
    assert len(records) == 80
    assert all(r.checked for r in records), "our replay disagrees with the server's winnings"
    assert [r.client_pos for r in records[:4]] == [0, 1, 0, 1]
    assert sum(r.showdown for r in records) > 0
    s = summarize(records)
    assert s["mismatched"] == 0 and np.isfinite(s["adjusted"]["bb100"])
    assert s["raw"]["bb100"] == pytest.approx(np.mean([r.won for r in records]))


def test_all_in_adjustment_only_changes_showdown_hands() -> None:
    fake = FakeSlumbot(opponent="fullhouse:aggressor", seed=3)
    records = play_session(make_policy("fullhouse:aggressor", 2), 120, _client(fake))
    moved = [r for r in records if r.adjusted != r.won]
    assert moved and all(r.showdown for r in moved)
    assert all(abs(r.adjusted) <= STACK for r in records)


def test_policy_observes_its_own_cards_only_and_every_hand() -> None:
    fake = FakeSlumbot()
    hero = Recorder()
    play_session(hero, 30, _client(fake))
    assert sum(isinstance(e, HandStart) for e in hero.events) == 30
    assert sum(isinstance(e, HandEnd) for e in hero.events) == 30
    assert {e.seat for e in hero.events if isinstance(e, HoleCards)} == {0}


def test_refused_raise_is_replaced_by_check_or_call() -> None:
    fake = FakeSlumbot(opponent="fullhouse:template", reject_bets=True)
    records = play_session(Recorder(), 20, _client(fake))
    assert fake.rejections > 0
    assert sum(r.rejected for r in records) == fake.rejections
    assert all(r.checked for r in records)


def test_server_that_disagrees_with_our_replay_is_an_error() -> None:
    r = FakeSlumbot().post("/slumbot/api/new_hand", {})
    r["action"] += "kk"  # nonsense for a hand that is waiting on us
    with pytest.raises(ProtocolError):
        play_hand(SlumbotClient(post=lambda path, body: r), Recorder(), 0)


def test_heads_up_exploiter_and_blueprint_play_through_the_replay() -> None:
    for spec in ("blueprint:profiles/blueprint-hu-v1.npz", "exploiter-hu:champion"):
        fake = FakeSlumbot(opponent="fullhouse:shark", seed=5)
        hero = make_policy(spec, 4)
        records = play_session(hero, 25, _client(fake))
        assert all(r.checked for r in records), spec
    assert hero.tracker.hands[1] == 25 and hero.errors == 0
