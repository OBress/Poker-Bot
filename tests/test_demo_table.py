"""Demo table (demo/server): hands play out, chips are conserved, views hide private cards."""

from __future__ import annotations

import pytest

from demo.server.engine_seat import EngineSeat, tilted_params
from demo.server.table import Table
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
    assert t.seats[3].kind == "human" and t.seats[3].stack == t.buyin


def test_read_challenge_ends_after_its_hands_with_a_read_and_a_record(tmp_path) -> None:
    import json

    from demo.server import history

    t = Table(7, mode="read", target=12)
    t.sit("me", "Owen", 0)
    t.running = True
    _play(t, 12)
    assert t.finished and not t.running and not t.start_hand()
    me = t.view("me")["myRead"]
    assert me["hands"] == 12 and abs(sum(me["probs"].values()) - 1) < 1e-6
    assert len(me["trajectory"]) == 12 and {s["key"] for s in me["stats"]} >= {"vpip", "pfr"}
    vpip = next(s for s in me["stats"] if s["key"] == "vpip")
    assert vpip["n"] == 12 and set(vpip["styles"]) == set(me["probs"])
    assert t.view(None)["finalReads"]["0"]["style"] == me["style"]
    history.save(t.record(), tmp_path)
    (row,) = history.summaries(tmp_path)
    assert row["mode"] == "read" and row["hands"] == 12 and row["humans"][0]["name"] == "Owen"
    assert json.loads(json.dumps(history.load(t.id, tmp_path)))["players"]


def test_standing_up_on_your_turn_does_not_stall_the_game(tmp_path, monkeypatch) -> None:
    import asyncio

    from demo.server import history
    from demo.server.main import Game

    monkeypatch.setattr(history, "GAMES", tmp_path)

    async def scenario() -> int:
        g = Game(0)
        g.clients["me"] = set()  # connected, as far as the server knows
        g.new_game("me", "Owen", "cash", None)
        g.bot_delay = 0.0
        await g.handle("me", {"type": "start"})
        for _ in range(400):  # until it's my turn
            if g.pending is not None:
                break
            await asyncio.sleep(0.01)
        assert g.pending is not None
        hand = g.table.hand_no
        await g.handle("me", {"type": "leave"})
        for _ in range(400):  # the hand finishes without waiting out my 45 s clock
            if g.table.hand_no > hand:
                break
            await asyncio.sleep(0.01)
        g.stop_loop()
        return g.table.hand_no - hand

    assert asyncio.run(scenario()) >= 1


def test_custom_setup_all_pokerbots_blinds_and_buyin() -> None:
    from demo.server.table import parse_setup

    setup = {"players": 5, "seats": [{}, *[{"kind": "pokerbot"}] * 4], "sb": 5, "bb": 10,
             "buyin": 400}  # fmt: skip
    t = Table(8, setup=setup)
    t.sit("me", "Owen", 0)
    assert [s.kind for s in t.seats] == ["human"] + ["pokerbot"] * 4 + ["empty"] * 3
    assert t.sb == 50 and t.buyin == 4000  # engine chips: big blind 100, 40 big blinds
    assert t.view("me")["money"] == {"sb": 5.0, "bb": 10.0} and t.amount(250) == "25"
    _play(t, 5)
    assert sum(s.net for s in t.seats) == 0
    for bad in (
        {"players": 1},
        {"sb": 3, "bb": 2},
        {"bb": 2, "buyin": 10},
        {"seats": [{}, {"kind": "x"}]},
    ):
        with pytest.raises(ValueError):
            parse_setup(bad)


def test_a_saved_game_resumes_with_its_reads(tmp_path) -> None:
    from demo.server import history

    t = Table(9)
    t.sit("old-tab", "Owen", 0)
    _play(t, 8)
    before = t.view("old-tab")["myRead"]
    history.save_table(t, tmp_path)
    assert history.resumable(t.id, tmp_path)
    r = history.load_table(t.id, tmp_path)
    assert r.resume_as("new-tab", "Owen") == 0 and r.seat_of("new-tab") == 0
    assert r.view("new-tab")["myRead"]["probs"] == pytest.approx(before["probs"])
    _play(r, 2)  # same seats, so the reader keeps its memory instead of resetting
    assert r.hand_no == 10 and r.view("new-tab")["myRead"]["hands"] == 10
    r.target, r.mode = 10, "read"  # once finished, the saved state goes away
    history.save_table(r, tmp_path)
    assert not history.resumable(t.id, tmp_path)
