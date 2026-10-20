"""Track B: abstract heads-up game, MCCFR tables and the blueprint's translation to the engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokerbot.baselines import hunl, mccfr
from pokerbot.baselines.blueprint import BlueprintPolicy, abstract_state
from pokerbot.engine.cards import hand_rank
from pokerbot.engine.state import HandState
from pokerbot.engine.table import play_hand
from pokerbot.opponents.bot import StyleBot

EDGES = [list(np.linspace(0.1, 0.9, hunl.BUCKETS - 1))] * 3


def _walk(s: hunl.State, seen: list[hunl.State]) -> None:
    if s.terminal():
        seen.append(s)
        return
    acts = hunl.legal(s)
    assert len(set(acts)) == len(acts) and acts[0] in "fc"
    for a in acts:
        _walk(hunl.apply(s, a), seen)


def test_abstract_tree_is_finite_and_chip_consistent() -> None:
    leaves: list[hunl.State] = []
    _walk(hunl.State(), leaves)
    assert 20_000 < len(leaves) < 200_000
    for s in leaves:
        assert all(0 < p <= hunl.STACK for p in s.put)
        if s.folded < 0:  # showdown: both put in the same amount (equal stacks)
            assert s.put[0] == s.put[1]


def test_preflop_sequences() -> None:
    s = hunl.apply(hunl.State(), "c")  # small blind completes
    assert s.street == 0 and s.to_act == 1 and "f" not in hunl.legal(s)
    s = hunl.apply(s, "c")  # big blind checks: flop, big blind first
    assert s.street == 1 and s.to_act == 1 and s.hist == "cc/"
    shove = hunl.apply(hunl.apply(hunl.State(), "a"), "c")
    assert shove.terminal() and shove.put == [hunl.STACK, hunl.STACK]
    folded = hunl.apply(hunl.apply(hunl.State(), "2"), "f")
    holes, board = ((0, 1), (2, 3)), (10, 20, 30, 40, 50)
    assert hunl.showdown_payoff(folded, holes, board, hand_rank) == hunl.BB


def test_mccfr_learns_and_tables_grow() -> None:
    t = mccfr.Tables(capacity=16)
    tr = mccfr.Trainer(t, EDGES, seed=0)
    for _ in range(30):
        tr.iterate()
    assert len(t) > 100 and t.nact[: len(t)].min() >= 1
    avg = t.average()
    assert all(np.isclose(v.sum(), 1) for v in avg.values())


def test_blueprint_translates_real_hands(tmp_path: Path) -> None:
    t = mccfr.Tables()
    tr = mccfr.Trainer(t, EDGES, seed=1)
    for _ in range(200):
        tr.iterate()
    path = tmp_path / "bp.npz"
    digest = mccfr.save(path, t, EDGES, {"iterations": 200})
    assert len(digest) == 16
    bp = BlueprintPolicy(path, seed=0)
    rng = np.random.default_rng(0)
    for h in range(60):
        opp = StyleBot({"noise": 0.0}, int(rng.integers(1000)))
        seats = [bp, opp] if h % 2 else [opp, bp]
        result = play_hand(seats, [10_000, 10_000], h % 2, h, small_blind=50, big_blind=100)
        assert sum(result.payoffs) == 0
    assert bp.lookups > 20 and bp.misses < bp.lookups


def test_abstract_state_matches_engine_streets() -> None:
    deck = list(range(52))
    state = HandState([10_000, 10_000], 0, deck, 50, 100)
    from pokerbot.engine import Action

    for a in (Action.raise_to(250), Action.check_call()):  # open 2.5 bb, call
        state.apply(a)
    s = abstract_state(state.decision_state())
    assert s is not None and s.street == 1 and s.to_act == 1 and s.hist.startswith("2c/")
