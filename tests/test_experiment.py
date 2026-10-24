"""Phase 7 final experiment: decision rule, belief ablation and config."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokerbot.evolve import hu, leduc, program
from pokerbot.evolve.leduc_task import TEMPLATE as LEDUC_TEMPLATE
from pokerbot.experiment import _nash_indices, holm, load_config

ROOT = Path(__file__).parents[1]


def test_holm_steps_down_and_stops_at_the_first_failure() -> None:
    d = holm({"a": 0.001, "b": 0.004, "c": 0.02, "d": 0.003}, alpha=0.01)
    assert d["a"]["threshold"] == 0.01 / 4 and d["a"]["supported"]
    assert d["d"]["threshold"] == 0.01 / 3 and d["d"]["supported"]
    assert d["b"]["threshold"] == 0.01 / 2 and d["b"]["supported"]
    assert not d["c"]["supported"]
    # the smallest p misses alpha / 3, so the others fail although each is below its threshold
    d = holm({"a": 0.0034, "b": 0.0045, "c": 0.0049}, alpha=0.01)
    assert not any(v["supported"] for v in d.values())


def test_argmax_beliefs_are_one_hot_and_differ_from_the_posterior() -> None:
    base = program.template(LEDUC_TEMPLATE)
    seen: dict[str, list[np.ndarray]] = {"bayes": [], "argmax": []}

    def make(key):
        ns = program.load(base, LEDUC_TEMPLATE)

        class Spy(ns["Exploiter"]):
            def policy(self, state, beliefs, feats, baseline):
                seen[key].append(np.array(beliefs))
                return baseline

        return lambda: Spy(leduc.style_table(), {})

    spec = leduc.SessionSpec("train", 0)
    leduc.run_session(make("bayes"), spec)
    leduc.run_session(make("argmax"), spec, argmax=True)
    a = np.array(seen["argmax"])
    assert np.allclose(a.sum(axis=1), 1) and set(np.unique(a)) <= {0.0, 1.0}
    b = np.array(seen["bayes"])
    assert np.array_equal(a.argmax(axis=1), b.argmax(axis=1))
    assert not np.allclose(a, b)


def test_safety_indices_are_baseline_sessions_after_the_main_block() -> None:
    for game, g in (("leduc", leduc), ("hu", hu)):
        idx = _nash_indices(game, 1_001_000, 50, "test")
        assert len(idx) == 50 and idx[0] >= 1_001_000
        assert all(g.SessionSpec("test", i).kind == "nash" for i in idx)


def test_registered_config_is_complete_and_fresh() -> None:
    cfg = load_config(ROOT / "configs/final-v1.yaml")
    assert cfg["first_index"] >= 100_000  # above every index used before Phase 7
    for key in ("leduc", "hu", "classifier"):
        for spec in cfg[key]["programs"].values():
            assert (ROOT / spec["path"]).exists()
            assert "params" not in spec or (ROOT / spec["params"]).exists()
    for game in ("leduc", "hu"):
        assert {"evolved", "seed"} <= set(cfg[game]["programs"])
        assert cfg[game]["beliefs"][0] == "classifier" and cfg[game]["margin"] > 0
