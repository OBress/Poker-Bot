"""Parametric bots, hand strength, calibration plumbing and showdown-range tables."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pokerbot.archetypes.features import FeatureTransform
from pokerbot.data.replay import replay
from pokerbot.engine.actions import ActionType, Street
from pokerbot.engine.cards import cards
from pokerbot.engine.events import PlayerAction
from pokerbot.engine.state import DecisionState, LegalActions
from pokerbot.engine.table import play_hand
from pokerbot.opponents.bot import PARAMS, STRUCTURE_PARAMS, StyleBot
from pokerbot.opponents.calibrate import (
    CALIBRATED,
    LAYOUT,
    Population,
    Target,
    calibrate_style,
    init_params,
    loss,
    showdown_z,
    simulate_features,
    split,
    to_params,
    to_x,
    warm_style,
)
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.showdown import RangeTable, compare
from pokerbot.opponents.simulate import BIG_BLIND, rates, simulate_stats
from pokerbot.opponents.strength import hand_class, postflop_equity, preflop_percentile
from pokerbot.stats.showdown import (
    LINES,
    line_class,
    river_equity,
    shown_hands,
    street_tokens,
    strength_bucket,
)

FEATURES = ("vpip", "pfr", "limp", "rfi_EP", "rfi_BTN", "wtsd")


def test_hand_classes_and_percentiles() -> None:
    assert hand_class(cards("AsKs")) == "AKs" and hand_class(cards("Kd As")) == "AKo"
    assert hand_class(cards("7c7d")) == "77"
    assert preflop_percentile(cards("AhAd")) < 0.01
    assert preflop_percentile(cards("7c2d")) > 0.9
    assert preflop_percentile(cards("AsKs")) < preflop_percentile(cards("AsKd"))


def test_postflop_equity_is_deterministic_and_sensible() -> None:
    nuts = postflop_equity(cards("AhKh"), cards("QhJhTh"))
    assert nuts == postflop_equity(cards("AhKh"), cards("QhJhTh")) and nuts > 0.95
    assert postflop_equity(cards("7c2d"), cards("AsKdQh")) < 0.35


def test_bots_play_legal_poker_without_crashing() -> None:
    rng = np.random.default_rng(0)
    for h in range(300):
        bots = [
            StyleBot(
                {k: float(rng.uniform(lo, hi)) for k, (lo, hi, _) in PARAMS.items()},
                h * 6 + i,
                structure=2 + h % 2,
            )
            for i in range(6)
        ]
        stacks = [int(b.buyin_bb * BIG_BLIND) for b in bots]
        result = play_hand(bots, stacks, h % 6, h, small_blind=50, big_blind=100)
        assert sum(result.payoffs) == 0


def test_params_round_trip_through_cma_space() -> None:
    p = {k: PARAMS[k][2] for k in CALIBRATED}
    back = to_params(to_x(p))
    assert all(abs(back[k] - p[k]) < 1e-6 * max(1, abs(p[k])) for k in CALIBRATED)
    extreme = to_params(np.full(len(CALIBRATED), 50.0))
    assert all(PARAMS[k][0] <= extreme[k] <= PARAMS[k][1] for k in CALIBRATED)


def test_wider_opening_params_open_more() -> None:
    def rfi_btn(open_btn: float) -> float:
        bots = [StyleBot({"open_btn": open_btn, "noise": 0}, i) for i in range(6)]
        return rates(simulate_stats(bots, ["a"] * 6, 1500, 3)["a"])["rfi_BTN"]

    assert rfi_btn(0.15) + 0.15 < rfi_btn(0.6)


def _target() -> Target:
    d = len(FEATURES)
    rate = np.array([0.25, 0.18, 0.03, 0.14, 0.40, 0.28])
    transform = FeatureTransform(
        FEATURES, prior_a=rate * 4, prior_b=(1 - rate) * 4, mean=np.zeros(d), std=np.ones(d)
    )
    return Target(
        index=0,
        name="t",
        transform=transform,
        mean=np.log(rate / (1 - rate)),
        sd=np.full(d, 0.3),
        rate=rate,
        depth=np.array([0.0, 0.0, 1.0, 0.0]),
        share=1.0,
    )


def test_calibration_reduces_loss() -> None:
    t = _target()
    start = {**init_params(dict(zip(FEATURES, [0.5, 0.05, 0.4, 0.02, 0.1, 0.5], strict=True)))}
    pop = Population([start], [t.depth], np.array([1.0]))
    before = loss(simulate_features(start, t, pop, 1, 600), t)
    res = calibrate_style(t, pop, to_x(start), iters=6, popsize=8, hands=300, seed=0)
    assert res.loss < before and len(res.history) == 6


def test_layouts_fill_every_format() -> None:
    t = _target()
    pop = Population([init_params({})], [t.depth], np.array([1.0]))
    for fmt, layout in LAYOUT.items():
        z, sums = split(simulate_features(init_params({}), t, pop, 3, 60, layout=layout), t)
        assert z.shape == (len(FEATURES),) and np.isfinite(z).all(), fmt
        assert sums.sum() == 0  # no showdown target: shown hands aren't tracked


def test_showdown_terms_enter_the_loss() -> None:
    t = _target()
    t.showdown = np.array([0.8, 0.75, 0.45, np.nan])  # no real all-in data
    pop = Population([init_params({})], [t.depth], np.array([1.0]))
    sim = simulate_features(init_params({}), t, pop, 4, 400)
    total, _, n = split(sim, t)[1]
    assert n[:3].sum() >= 10 and np.all(total <= n)
    sdz = showdown_z(sim, t)
    assert sdz.shape == (4,) and sdz[3] == 0.0 and np.isfinite(sdz).all()
    far = replace(t, showdown=np.array([0.0, 0.0, 0.0, np.nan]))
    assert loss(sim, far) > loss(sim, t)


def test_frozen_bounds_decode_v2_files_unchanged() -> None:
    from pokerbot.opponents.bot import STRUCTURE_BOUNDS

    names = STRUCTURE_PARAMS[2]
    x = np.zeros(len(names))
    old = to_params(x, names, STRUCTURE_BOUNDS[2])
    assert old["defend"] == (0.5 + 6.0) / 2 and old["noise"] == 0.1
    assert to_params(x, names)["defend"] != old["defend"]  # current bounds differ


def test_warm_start_maps_older_parameter_sets() -> None:
    old = STRUCTURE_PARAMS[2]
    style = {
        "params": {k: PARAMS[k][2] for k in old} | {"open_btn": 0.5},
        "x_cov": (0.04 * np.eye(len(old))).tolist(),
    }
    x, cov = warm_style(style, old)
    assert x.shape == (len(CALIBRATED),) and cov.shape == (len(CALIBRATED),) * 2
    assert abs(to_params(x)["open_btn"] - 0.5) < 1e-6
    i, j = CALIBRATED.index("open_btn"), CALIBRATED.index("commit")
    assert cov[i, i] == 0.04 and cov[j, j] == 0.25 and cov[i, j] == 0


def _style(d: int) -> dict:
    return {
        "index": 0, "name": "s", "share": 1.0,
        "x_mean": [0.0] * d, "x_cov": np.eye(d).tolist(), "depth": [0.5, 0.0, 0.5, 0.0],
    }  # fmt: skip


def test_sampler_draws_bounded_distinct_bots(tmp_path: Path) -> None:
    path = tmp_path / "bots.json"
    doc = {"version": "t", "structure": 3, "formats": {"sh": {"styles": [_style(len(CALIBRATED))]}}}
    path.write_text(json.dumps(doc))
    s = StyleSampler(path, "sh")
    rng = np.random.default_rng(1)
    draws = [s.draw(rng) for _ in range(50)]
    assert all(PARAMS["open_ep"][0] <= d.bot.p["open_ep"] <= PARAMS["open_ep"][1] for d in draws)
    assert len({round(d.bot.p["open_ep"], 6) for d in draws}) == 50
    assert {d.bot.buyin_bb < 25 for d in draws} == {True, False}  # both depth buckets used
    clones = StyleSampler(path, "sh", spread=0.0)
    assert len({round(clones.draw(rng).bot.p["open_ep"], 9) for _ in range(5)}) == 1


def test_sampler_keeps_frozen_files_on_their_structure(tmp_path: Path) -> None:
    old = STRUCTURE_PARAMS[2]
    path = tmp_path / "bots.json"  # a bots-v2 file: no structure field, 24 parameters
    path.write_text(
        json.dumps({"version": "v2", "formats": {"sh": {"styles": [_style(len(old))]}}})
    )
    s = StyleSampler(path, "sh")
    bot = s.draw(np.random.default_rng(0)).bot
    assert s.structure == 2 and bot.structure == 2 and s.calibrated == old
    assert bot.p["commit"] == PARAMS["commit"][2]  # not in v2: default, unused by structure 2


def _state(
    to_call: int,
    stacks: tuple[int, ...],
    actions: tuple[PlayerAction, ...] = (),
    street: Street = Street.PREFLOP,
    board: tuple[int, ...] = (),
    hole: str = "7c2d",
    pot: int = 150,
) -> DecisionState:
    n = len(stacks)
    return DecisionState(
        seat=0, hole=cards(hole), board=board, street=street, pot=pot + to_call,
        legal=LegalActions(to_call, True, 200, stacks[0]), stacks=stacks,
        bets=(0,) * n, folded=(False,) * n, button=n - 1, big_blind=100, actions=actions,
    )  # fmt: skip


def test_noise_never_limps_in_structure_3() -> None:
    state = _state(100, (10_000,) * 6)  # seat 0 is UTG, unopened
    for structure, can_limp in ((3, False), (2, True)):
        bot = StyleBot({"noise": 1.0}, 0, structure=structure)
        kinds = {bot.act(state).type for _ in range(300)}
        assert (ActionType.CHECK_CALL in kinds) == can_limp
        assert ActionType.FOLD in kinds and ActionType.BET_RAISE in kinds


def test_committed_bot_calls_with_pot_odds() -> None:
    bet = PlayerAction(1, Street.TURN, ActionType.BET_RAISE, 300, 300, 1000, 0, False)
    state = _state(300, (350, 10_000), (bet,), Street.TURN, cards("Kd8s3c2h"), "Qh8h", 1000)
    tight = {"noise": 0.0, "late_call_base": 0.7, "call_strict": 2.5}
    assert StyleBot(tight | {"commit": 0.0}, 0).act(state).type == ActionType.FOLD
    assert StyleBot(tight | {"commit": 1.0}, 0).act(state).type == ActionType.CHECK_CALL
    deep = replace(state, stacks=(10_000, 10_000))  # same spot with chips behind: not committed
    assert StyleBot(tight | {"commit": 1.0}, 0).act(deep).type == ActionType.FOLD


def test_respect_folds_more_to_cbets_than_to_stabs() -> None:
    def fold_rate(aggressor: int, respect: float) -> float:
        pre = PlayerAction(aggressor, Street.PREFLOP, ActionType.BET_RAISE, 300, 300, 150, 0, False)
        bet = PlayerAction(1, Street.FLOP, ActionType.BET_RAISE, 400, 400, 700, 0, False)
        state = _state(400, (10_000, 10_000), (pre, bet), Street.FLOP, cards("Kd8s3c"), "Qh8h", 700)
        bot = StyleBot({"noise": 0.0, "respect": respect, "call_base": 0.3, "bluff_raise": 0}, 0)
        return np.mean([bot.act(state).type == ActionType.FOLD for _ in range(50)])

    assert fold_rate(1, 0.0) == 0.0 and fold_rate(1, 0.3) == 1.0  # c-bet by the raiser
    assert fold_rate(0, 0.3) == 0.0  # we raised preflop: their bet is a donk, not a c-bet


def _act(seat: int, street: Street, kind: ActionType, to_call: int = 0) -> PlayerAction:
    return PlayerAction(seat, street, kind, 0, 0, 100, to_call, False)


def test_line_classes() -> None:
    assert line_class("BBB") == "bbb" and line_class("XXB") == "river_bet"
    assert line_class("CCC") == "call_down" and line_class("XCC") == "call_down"
    assert line_class("BXC") == "river_call" and line_class("XXC") == "river_call"
    assert line_class("BBX") == "bet_check" and line_class("CXX") == "call_check"
    assert line_class("XXX") == "check_down" and line_class("B--") == "all_in"
    assert line_class("---") == "all_in"
    b, c = ActionType.BET_RAISE, ActionType.CHECK_CALL
    acts = [
        _act(0, Street.FLOP, c), _act(1, Street.FLOP, b), _act(0, Street.FLOP, c, 100),
        _act(0, Street.TURN, c), _act(1, Street.TURN, c),
        _act(0, Street.RIVER, b), _act(1, Street.RIVER, b, 100), _act(0, Street.RIVER, c, 100),
    ]  # fmt: skip
    assert street_tokens(acts, 0) == "CXB" and street_tokens(acts, 1) == "BXB"
    every = {line_class(f + t + r) for f in "BCX-" for t in "BCX-" for r in "BCX-"}
    assert every == set(LINES)


def test_river_equity_and_buckets() -> None:
    board = cards("AsKsQsJs2d")
    assert river_equity(cards("Ts3h"), board) == 1.0  # royal flush
    assert river_equity(cards("7c3h"), board) < 0.2
    top_pair = river_equity(cards("Ad3h"), cards("Ac8d5h4c2s"))
    assert 0.75 < top_pair < 0.95
    assert [strength_bucket(e) for e in (0.1, 0.35, 0.6, 0.8, 0.9, 1.0)] == [0, 1, 2, 3, 4, 4]


def test_shown_hands_from_real_and_simulated_hands() -> None:
    row = {
        "players": ["a", "b"], "starting_stacks": [100.0, 100.0], "blinds": [1.0, 2.0],
        "antes": [0.0, 0.0], "min_bet": 2.0, "big_blind": 2.0,
        "actions": [
            "d dh p1 ????", "d dh p2 ????", "p2 cbr 6", "p1 cc", "d db 2c7s8c", "p1 cc",
            "p2 cbr 4", "p1 cc", "d db 3d", "p1 cc", "p2 cbr 10", "p1 cc", "d db Ks", "p1 cc",
            "p2 cbr 20", "p1 cc", "p2 sm AhAd", "p1 sm 7h7d",
        ],
    }  # fmt: skip
    by_seat = {s.seat: s for s in shown_hands(replay(row).events)}
    assert by_seat[1].line == "bbb" and by_seat[0].line == "call_down"
    assert by_seat[0].bucket == 4  # a set of sevens
    rng = np.random.default_rng(5)
    bots = [StyleBot({"noise": 0.0, "call_base": 0.1}, int(rng.integers(1000))) for _ in range(3)]
    table = RangeTable(2)
    for h in range(200):
        result = play_hand(bots, [10_000] * 3, h % 3, h, small_blind=50, big_blind=100)
        table.add_events(result.events, [0, 1, 1])
    assert table.counts[0].sum() > 5 and table.counts[1].sum() > 5
    assert np.isclose(table.counts.sum(), table.w2.sum())  # unit weights


def test_range_table_compare() -> None:
    a, b = RangeTable(1), RangeTable(1)
    rng = np.random.default_rng(0)
    for _ in range(400):
        eq = float(rng.uniform())
        a.add(np.array([0.5]), "bbb", strength_bucket(eq), eq)
        b.add(np.array([1.0]), "bbb", strength_bucket(eq), eq)
    same = compare(a, RangeTable.from_dict(b.to_dict()), ["s"])["s"]
    cell = same["lines"]["bbb"]
    assert same["lines_tested"] == 1 and abs(cell["equity_z"]) < 1e-6 and cell["tv"] < 1e-9
    for _ in range(400):
        b.add(np.array([1.0]), "bbb", 0, 0.05)
    worse = compare(a, b, ["s"])["s"]["lines"]["bbb"]
    assert worse["equity_z"] < -5 and worse["chi2_p"] < 1e-6


def test_draw_hands_follows_real_counts_and_scales_long_samples() -> None:
    from pokerbot.opponents.calibrate import draw_hands

    q = np.linspace(500, 20_000, 101)
    rng = np.random.default_rng(0)
    draws = [draw_hands(q, rng, 4000) for _ in range(2000)]
    assert all(500 <= h <= 4000 for h, _ in draws)
    assert all(abs(h * s - min(max(h * s, 500), 20_000)) < 1e-6 for h, s in draws)
    assert all(s == 1.0 for h, s in draws if h < 4000) and any(s > 1 for _, s in draws)
    real = np.mean([h * s for h, s in draws])
    assert abs(real - q.mean()) < 600  # the scaled counts keep the real mean


def test_solve_scale_inverts_noise_plus_spread() -> None:
    from pokerbot.opponents.validate import solve_scale

    pts = [(s, float(np.sqrt(0.4 + 0.6 * (s / 0.8) ** 2))) for s in (0.2, 0.4, 0.8, 1.2)]
    assert abs(solve_scale(pts, 0.2, 1.5) - 0.8) < 1e-6
    noisy_floor = [(s, 1.1 + 0.01 * s) for s in (0.2, 0.5, 1.0)]
    assert solve_scale(noisy_floor, 0.2, 1.5) == 0.2


def test_write_bots_merges_formats_of_one_version(tmp_path: Path) -> None:
    from pokerbot.opponents.calibrate import write_bots

    path = tmp_path / "bots.json"
    write_bots(path, "v9", "run-sh", {"sh": {"styles": []}})
    write_bots(path, "v9", "run-hu", {"hu": {"styles": []}})
    doc = json.loads(path.read_text())
    assert set(doc["formats"]) == {"sh", "hu"} and doc["run_id"] == {"sh": "run-sh", "hu": "run-hu"}
    write_bots(path, "v10", "run-x", {"fr": {"styles": []}})  # another version replaces the file
    assert set(json.loads(path.read_text())["formats"]) == {"fr"}


def test_bots_v3_draws_every_format() -> None:
    path = Path(__file__).parents[1] / "profiles" / "bots-v3.json"
    rng = np.random.default_rng(0)
    for fmt, seats in (("hu", 2), ("sh", 6), ("fr", 9)):
        s = StyleSampler(path, fmt)
        assert s.structure == 3 and "bluff_eq" in s.calibrated
        bots = [d.bot for d in s.table(rng, seats)]
        result = play_hand(bots, [int(b.buyin_bb * BIG_BLIND) for b in bots], 0, 1, 50, 100)
        assert sum(result.payoffs) == 0
