"""Evaluation: all-in luck removal, duplicate matches, heuristic baselines."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from phevaluator import evaluate_cards

from pokerbot.engine import Action
from pokerbot.engine.cards import cards
from pokerbot.engine.state import HandState
from pokerbot.eval.allin import allin_adjusted
from pokerbot.eval.duplicate import SessionSpec, bootstrap_ci, make_policy, play_session
from tests.test_engine_rules import run, stacked_deck

BOTS = Path(__file__).parents[1] / "profiles" / "bots-v3.json"


def test_flop_allin_is_replaced_by_exact_equity() -> None:
    deck = stacked_deck(["AhAd", "KcKs"], "2c7d9hJsQs")
    state = HandState([10_000, 10_000], 0, deck, 50, 100)
    script = [Action.check_call(), Action.check_call()]  # limp, check
    script += [Action.check_call(), Action.raise_to(9_900), Action.check_call()]  # flop all-in
    run(state, script)
    adj = allin_adjusted(state.events)
    assert adj is not None and abs(adj.sum()) < 1e-9
    flop, hole_a, hole_k = cards("2c7d9h"), cards("AhAd"), cards("KcKs")
    left = [c for c in range(52) if c not in (*flop, *hole_a, *hole_k)]
    eq = 0.0
    for t, r in combinations(left, 2):
        a = evaluate_cards(*hole_a, *flop, t, r)
        k = evaluate_cards(*hole_k, *flop, t, r)
        eq += 1.0 if a < k else 0.5 if a == k else 0.0
    eq /= len(left) * (len(left) - 1) / 2
    assert abs(adj[0] - (eq * 20_000 - 10_000)) < 1e-6


def test_no_adjustment_without_an_early_allin() -> None:
    deck = stacked_deck(["AhAd", "KcKs"], "2c7d9hJsQs")
    state = HandState([10_000, 10_000], 0, deck, 50, 100)
    run(state, [Action.check_call(), Action.check_call()] + [Action.check_call()] * 6)
    assert allin_adjusted(state.events) is None
    fold = HandState([10_000, 10_000], 0, deck, 50, 100)
    run(fold, [Action.fold()])
    assert allin_adjusted(fold.events) is None


def test_preflop_allin_is_zero_sum_and_seeded() -> None:
    deck = stacked_deck(["AhAd", "KcKs"], "2c7d9hJsQs")
    state = HandState([10_000, 10_000], 0, deck, 50, 100)
    run(state, [Action.raise_to(10_000), Action.check_call()])
    a, b = allin_adjusted(state.events, seed=3), allin_adjusted(state.events, seed=3)
    assert a is not None and np.allclose(a, b) and abs(a.sum()) < 1e-9
    assert 0.5 * 10_000 < a[0] < 0.75 * 10_000  # AA vs KK: ~82% of a 200 bb pot


def test_duplicate_session_is_reproducible_and_cancels_for_identical_heroes() -> None:
    spec = SessionSpec(str(BOTS), "sh", 6, 40, seed=5, style=None, spread=1.0, allin=True)
    a = play_session(spec, ["abc", "abc", "hand-strength"])
    b = play_session(spec, ["abc", "abc", "hand-strength"])
    assert np.array_equal(a, b)
    assert np.array_equal(a[0], a[1])  # same hero, same cards, same opponents -> same results
    assert not np.array_equal(a[0], a[2])


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    totals = rng.normal(5.0, 30.0, size=200)  # bb per 100-hand session
    mean, lo, hi = bootstrap_ci(totals, 100)
    assert lo < mean < hi and abs(mean - totals.mean()) < 1e-9 and hi - lo < 15


def test_heuristic_policies_build() -> None:
    assert make_policy("abc", 1).p["noise"] == 0.0
    assert make_policy("hand-strength", 1).buyin_bb == 100.0
