"""Evolution harness: program gates, sandbox, scoring and the loop with a fake LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from pokerbot.evolve import dataset, program, prompts, scoring
from pokerbot.evolve.api import SeatFeatures, StyleProfiles
from pokerbot.evolve.harness import validate_output
from pokerbot.evolve.program import ProgramError

ROOT = Path(__file__).parents[1]
SEED_A = (ROOT / "programs/classifier/seed_a.py").read_text(encoding="utf-8")
SEED_B = (ROOT / "programs/classifier/seed_b.py").read_text(encoding="utf-8")
PROFILES = ROOT / "profiles/styles-v1.json"


def _with(posterior: str) -> str:
    return program.replace_block(SEED_A, "posterior", posterior)


@pytest.mark.parametrize(
    "bad,reason",
    [
        ("        import os\n        return np.ones(8)", "import"),
        ("        open('x')\n        return np.ones(8)", "forbidden_name"),
        ("        return self.__dict__", "dunder"),
        ("        return np.load('x')", "forbidden_attr"),
        ("        global Z\n        return np.ones(8)", "global_state"),
        ("        return eval('1')", "forbidden_name"),
        ("        return np.ones(8) +", "syntax"),
    ],
)
def test_gates_reject_unsafe_programs(bad: str, reason: str) -> None:
    with pytest.raises(ProgramError) as e:
        program.check(_with(bad))
    assert e.value.reason == reason


def test_frozen_region_must_match_the_template() -> None:
    with pytest.raises(ProgramError) as e:
        program.check(SEED_A.replace("self.profiles = profiles", "self.profiles = None"))
    assert e.value.reason == "frozen_changed"


def test_seeds_load_and_score_valid_posteriors() -> None:
    prof = StyleProfiles.load(PROFILES, "sh")
    feats = SeatFeatures.make(30, {"vpip": (30, 12), "pfr": (30, 3), "limp": (25, 9)},
                              np.zeros((8, 5)))  # fmt: skip
    for src in (SEED_A, SEED_B):
        ns = program.load(src)
        clf = ns["Classifier"](prof, program.defaults(ns))
        p, ok = validate_output(clf.posterior(0, feats), prof.k)
        assert ok and np.isclose(p.sum(), 1)
    loose = int(np.argmax(p))
    assert prof.pooled_rate[loose, prof.features.index("vpip")] > 0.3  # a 40% VPIP looks loose


def test_answers_replace_blocks_with_the_right_indentation() -> None:
    answer = "IDEA: flatter prior\n<<<BLOCK posterior>>>\nreturn np.ones(self.profiles.k)\n<<<END BLOCK>>>"
    src, idea = prompts.apply_answer(SEED_A, answer)
    assert idea == "flatter prior" and "        return np.ones(self.profiles.k)" in src
    program.check(src)
    with pytest.raises(ProgramError):
        prompts.apply_answer(SEED_A, "no blocks here")


def test_scoring_rules() -> None:
    k = 4
    perfect = [np.tile(np.eye(k)[[0, 1]][:, None, :], (1, 5, 1))]
    s = scoring.score(perfect, [[0, 1]], us_per_call=10.0)
    assert s["accuracy@10"] == 1 and s["logloss@10"] < 1e-5 and s["ece@100"] < 1e-9
    uniform = [np.full((2, 5, k), 1 / k)]
    u = scoring.score(uniform, [[0, 1]], us_per_call=10.0)
    assert abs(u["logloss@10"] - np.log(k)) < 1e-9 and u["fitness"] < s["fitness"]
    slow = scoring.score(perfect, [[0, 1]], us_per_call=scoring.TIME_BUDGET_US + 1000)
    assert slow["fitness"] < s["fitness"]


def test_test_split_is_refused_during_evolution() -> None:
    with pytest.raises(PermissionError):
        dataset.load("sh", "test")


@dataclass
class _Answer:
    text: str
    cached: bool = False


@dataclass
class FakeLLM:
    """Stands in for the Luminal client: proposes a valid edit, then a broken one."""

    usage: object = field(default_factory=lambda: type("U", (), {
        "calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})())  # fmt: skip

    def complete_many(self, msgs, **kw):
        self.usage.calls += len(msgs)
        good = ("IDEA: sharper\n<<<BLOCK params>>>\nPARAMS = {\n    \"weight\": param(0.5, 0.05, "
                "1.0),\n    \"tau\": param(0.9, 0.3, 4.0),\n    \"showdown\": param(1.0, 0.0, 2.0),"
                "\n}\n<<<END BLOCK>>>")  # fmt: skip
        bad = "IDEA: broken\n<<<BLOCK posterior>>>\nimport os\n<<<END BLOCK>>>"
        return [_Answer(good if i % 2 == 0 else bad) for i in range(len(msgs))]

    def complete(self, msgs, **kw):
        self.usage.calls += 1
        return _Answer(
            "IDEA: still broken\n<<<BLOCK posterior>>>\nreturn eval('1')\n<<<END BLOCK>>>"
        )


def test_loop_runs_with_a_fake_llm(tmp_path: Path) -> None:
    from pokerbot.evolve.loop import evolve_classifier
    from pokerbot.track import Run

    for split in ("train", "val_a", "val_b"):
        dataset.build(ROOT / "profiles/bots-v3.json", "sh", split, 8, workers=2, root=tmp_path)
    with Run.start("t", kind="evolve-classifier", root=tmp_path / "runs") as run:
        out = evolve_classifier(run, FakeLLM(), "sh", PROFILES, ROOT / "missing.json",
                                {"seed_a": SEED_A}, generations=1, proposals=2, workers=2,
                                tune_iters=1, data_root=tmp_path)  # fmt: skip
    assert out["champion"]["id"] in {"seed_a", "g01-0"}
    cands = {p.name for p in (run.dir / "candidates").iterdir()}
    assert {"seed_a", "g01-0"} <= cands and "g01-1" not in cands  # the broken one was rejected
    events = (run.dir / "events.jsonl").read_text(encoding="utf-8")
    assert "candidate_failed" in events
