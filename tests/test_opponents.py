"""Parametric bots, hand strength, and calibration plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pokerbot.engine.cards import cards
from pokerbot.engine.table import play_hand
from pokerbot.opponents.bot import PARAMS, StyleBot
from pokerbot.opponents.calibrate import (
    CALIBRATED,
    Population,
    Target,
    calibrate_style,
    init_params,
    loss,
    simulate_rates,
    to_params,
    to_x,
)
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.simulate import BIG_BLIND, rates, simulate_stats
from pokerbot.opponents.strength import hand_class, postflop_equity, preflop_percentile

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
                {k: float(rng.uniform(lo, hi)) for k, (lo, hi, _) in PARAMS.items()}, h * 6 + i
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
    return Target(
        index=0,
        name="t",
        features=FEATURES,
        rate=np.array([0.25, 0.18, 0.03, 0.14, 0.40, 0.28]),
        sigma=np.full(len(FEATURES), 0.3),
        depth=np.array([0.0, 0.0, 1.0, 0.0]),
        share=1.0,
    )


def test_calibration_reduces_loss() -> None:
    t = _target()
    start = {**init_params(dict(zip(FEATURES, [0.5, 0.05, 0.4, 0.02, 0.1, 0.5], strict=True)))}
    pop = Population([start], [t.depth], np.array([1.0]))
    before = loss(simulate_rates(start, t, pop, 1, 600), t)
    res = calibrate_style(t, pop, to_x(start), iters=6, popsize=8, hands=300, seed=0)
    assert res.loss < before and len(res.history) == 6


def test_sampler_draws_bounded_distinct_bots(tmp_path: Path) -> None:
    d = len(CALIBRATED)
    style = {
        "index": 0, "name": "s", "share": 1.0,
        "x_mean": [0.0] * d, "x_cov": np.eye(d).tolist(), "depth": [0.5, 0.0, 0.5, 0.0],
    }  # fmt: skip
    path = tmp_path / "bots.json"
    path.write_text(json.dumps({"version": "t", "formats": {"sh": {"styles": [style]}}}))
    s = StyleSampler(path, "sh")
    rng = np.random.default_rng(1)
    draws = [s.draw(rng) for _ in range(50)]
    assert all(PARAMS["open_ep"][0] <= d.bot.p["open_ep"] <= PARAMS["open_ep"][1] for d in draws)
    assert len({round(d.bot.p["open_ep"], 6) for d in draws}) == 50
    assert {d.bot.buyin_bb < 25 for d in draws} == {True, False}  # both depth buckets used
    clones = StyleSampler(path, "sh", spread=0.0)
    assert len({round(clones.draw(rng).bot.p["open_ep"], 9) for _ in range(5)}) == 1
