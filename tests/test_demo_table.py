"""Demo table (demo/server): hands play out, chips are conserved, views hide private cards."""

from __future__ import annotations

import pytest

from demo.server.engine_seat import EngineSeat, tilted_params
from demo.server.table import BUYIN, Table
from pokerbot.baselines.heuristic import ABC_TAG
from pokerbot.engine.actions import Action


def _play(t: Table, hands: int, human=lambda t: Action.check_call()) -> None:
    for _ in range(hands):
        assert t.start_hand()
        while not t.hand.state.finished:
            s = t.seats[t.actor_seat()]
            t.apply(human(t) if s.kind == "human" else t.bot_action())


def test_full_table_plays_and_conserves_chips() -> None:
    t = Table(3)
    t.sit("me", "Owen", 0)
    _play(t, 12)
    assert sum(s.net for s in t.seats) == 0  # every chip won was lost by someone
    assert t.hand_no == 12 and any("wins" in line for line in t.log)


def test_views_show_only_your_own_cards_before_showdown() -> None:
    t = Table(4)
    t.sit("a", "A", 0)
    t.configure(1, "human")
    t.sit("b", "B", 1)
    assert t.start_hand()
    va, vb, spectator = t.view("a"), t.view("b"), t.view(None)
    assert va["seats"][0]["cards"] and va["seats"][1]["cards"] is None
    assert vb["seats"][1]["cards"] and vb["seats"][0]["cards"] is None
    assert all(s["cards"] is None for s in spectator["seats"])
    assert all(s["cards"] is None for s in va["seats"] if s["index"] != 0)


def test_heads_up_uses_the_evolved_exploiter_and_reads_heads_up_styles() -> None:
    t = Table(5)
    for i in range(8):
        t.configure(i, "empty")
    t.configure(2, "pokerbot")
    t.configure(5, "bot", 4)
    _play(t, 6)
    bot = t.seats[2].policy
    assert isinstance(bot, EngineSeat) and bot.mode == "heads-up exploiter"
    read = t.view(None)["seats"][5]["read"]
    assert read is not None and read["hands"] == 6


def test_multiway_engine_tilts_toward_the_opponents_weakness() -> None:
    vs_folders, vs_callers = tilted_params(0.0, 0.4), tilted_params(0.4, 0.0)
    assert vs_folders["bluff"] > ABC_TAG["bluff"] > vs_callers["bluff"]
    assert vs_callers["value_eq"] < ABC_TAG["value_eq"]
    assert tilted_params(0.0, 0.0) == ABC_TAG


def test_seat_rules() -> None:
    t = Table(6)
    t.sit("a", "A", 0)
    with pytest.raises(ValueError):
        t.sit("b", "B", 0)  # taken
    with pytest.raises(ValueError):
        t.sit("b", "B", 2)  # a PokerBot seat
    t.sit("a", "A", 0)  # sitting again is fine
    t.leave("a")
    assert t.seat_of("a") is None and t.seats[0].kind == "human"
    t.configure(3, "empty")
    t.sit("b", "B", 3)
    assert t.seats[3].kind == "human" and t.seats[3].stack == BUYIN
