"""Track A: small-game rules, CFR+, best response and exploitability."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokerbot.baselines.cfr import (
    CFRPlus,
    best_response,
    best_response_value,
    expected_value,
    exploitability,
    infosets,
    matchup,
    uniform,
)
from pokerbot.baselines.track_a import load_strategy, save_strategy
from pokerbot.games.limit import KUHN, LEDUC
from pokerbot.games.styles import STYLES, center, sample_params, style_strategy


def _kuhn_nash(alpha: float) -> dict[str, np.ndarray]:
    """The analytic Kuhn equilibrium family (player 0 bets J with prob alpha)."""
    bet0 = {0: alpha, 1: 0.0, 2: 3 * alpha}
    call0 = {0: 0.0, 1: alpha + 1 / 3, 2: 1.0}
    bet1 = {0: 1 / 3, 1: 0.0, 2: 1.0}
    call1 = {0: 0.0, 1: 1 / 3, 2: 1.0}
    s = {}
    for c in range(3):
        s[f"{c}|-|"] = np.array([1 - bet0[c], bet0[c]])
        s[f"{c}|-|cr"] = np.array([1 - call0[c], call0[c]])
        s[f"{c}|-|c"] = np.array([1 - bet1[c], bet1[c]])
        s[f"{c}|-|r"] = np.array([1 - call1[c], call1[c]])
    return s


def test_rules_and_chance() -> None:
    for g in (KUHN, LEDUC):
        assert abs(sum(p for _, p in g.deals) - 1) < 1e-12
        for (a, b), _ in g.deals:
            if g.rounds > 1:
                assert abs(sum(q for _, q in g.public_cards(a, b)) - 1) < 1e-12
    assert len(infosets(KUHN)) == 12 and len(infosets(LEDUC)) == 288
    assert LEDUC.contributions(["rrc", "crc"]) == (1 + 4 + 4, 1 + 4 + 4)
    assert LEDUC.payoff(["rrc", "crc"], (0, 2), 0) == 9  # pair of jacks beats a king
    assert LEDUC.payoff(["rf"], (0, 2), None) == 1  # player 1 folded to the bet
    assert LEDUC.payoff(["cc", "cc"], (1, 1), 2) == 0  # split


def test_exploitability_is_zero_at_known_kuhn_equilibria() -> None:
    for alpha in (0.0, 0.2, 1 / 3):
        s = _kuhn_nash(alpha)
        assert abs(exploitability(KUHN, s)) < 1e-12
        assert abs(expected_value(KUHN, s, s) + 1 / 18) < 1e-12


def test_cfr_plus_converges() -> None:
    kuhn = CFRPlus(KUHN)
    kuhn.iterate(1000)
    assert exploitability(KUHN, kuhn.average()) < 5e-4
    leduc = CFRPlus(LEDUC)
    leduc.iterate(150)
    s = leduc.average()
    assert exploitability(LEDUC, s) < 0.02
    assert abs(expected_value(LEDUC, s, s) + 0.0856) < 0.005  # known Leduc value for seat 0


def test_best_response_beats_everything_it_faces() -> None:
    opp = uniform(LEDUC)
    br = best_response(LEDUC, opp)
    v = best_response_value(LEDUC, opp, 0)
    assert abs(expected_value(LEDUC, br, opp) - v) < 1e-9  # the recorded policy attains it
    assert matchup(LEDUC, br, opp) >= matchup(LEDUC, uniform(LEDUC), opp)


def test_styles_are_valid_and_beaten_by_nash() -> None:
    solver = CFRPlus(KUHN)
    solver.iterate(500)
    nash = solver.average()
    rng = np.random.default_rng(0)
    for name in STYLES:
        for params in (center(name), sample_params(name, rng)):
            strat = style_strategy(LEDUC, params)
            assert all(np.isclose(p.sum(), 1) and (p >= 0).all() for p in strat.values())
        assert matchup(KUHN, nash, style_strategy(KUHN, center(name))) > 0


def test_frozen_strategy_round_trip(tmp_path: Path) -> None:
    s = _kuhn_nash(0.1)
    path = tmp_path / "b.json"
    digest = save_strategy(path, KUHN, s, {"iterations": 0})
    game, back, meta = load_strategy(path)
    assert game is KUHN and meta["hash"] == digest
    assert all(np.allclose(back[k], s[k]) for k in s)
