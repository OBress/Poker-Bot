"""Experiment tracker: everything a run logs must be readable back, including after a crash."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pokerbot.llm import LLMClient, LLMSettings
from pokerbot.track import Run, find_run, latest_metrics, list_runs, load_metrics, plot_run
from pokerbot.track.read import load_events, load_meta
from tests.test_llm_client import FakeBackend


def test_run_records_everything(tmp_path: Path) -> None:
    data = tmp_path / "data.parquet"
    data.write_bytes(b"1234")
    with Run.start(
        "demo", kind="evolve", config={"lr": 0.1}, seeds={"val": [1, 2]}, data={"d": data},
        root=tmp_path,
    ) as run:  # fmt: skip
        run.log_params(population=8)
        for step in range(5):
            run.log_metrics(
                {"val/logloss@25": 2.0 - 0.1 * step, "val/acc": np.float64(0.2 * step)},
                step=step,
                split="val",
            )
        run.save_candidate("g1-c0", "PARAMS = {}\n", parents=["seed"], score=1.5)
        run.update_candidate("g1-c0", promoted=True)
        run.save_artifact("profiles.json", {"k": 8})
        run.event("promoted", candidate="g1-c0", p_value=0.004)

    d = find_run("demo", tmp_path)
    meta = load_meta(d)
    assert meta["status"] == "completed" and meta["seconds"] >= 0
    assert meta["config"] == {"lr": 0.1} and meta["params"] == {"population": 8}
    assert meta["seeds"] == {"val": [1, 2]} and meta["data"]["d"]["bytes"] == 4
    assert meta["env"]["packages"]["numpy"]

    m = load_metrics(d)
    assert m.height == 10 and set(m["split"].unique()) == {"val"}
    last = {r["name"]: r for r in latest_metrics(d).to_dicts()}
    assert last["val/logloss@25"]["last"] == pytest.approx(1.6)
    assert last["val/logloss@25"]["min"] == pytest.approx(1.6)

    cand = json.loads((d / "candidates" / "g1-c0" / "meta.json").read_text())
    assert cand == {"parents": ["seed"], "score": 1.5, "promoted": True}
    assert (d / "candidates" / "g1-c0" / "program.py").read_text() == "PARAMS = {}\n"
    kinds = load_events(d)["kind"].to_list()
    assert kinds[0] == "run_started" and "promoted" in kinds and kinds[-1] == "run_finished"
    assert plot_run(d).exists()
    assert list_runs(tmp_path)["status"].to_list() == ["completed"]


def test_crash_is_recorded(tmp_path: Path) -> None:
    with pytest.raises(ValueError), Run.start("boom", kind="evolve", root=tmp_path) as run:
        run.log_metric("x", 1.0, step=0)
        raise ValueError("bad candidate")
    d = find_run("boom", tmp_path)
    assert load_meta(d)["status"] == "failed"
    errors = load_events(d).filter(load_events(d)["kind"] == "error")
    assert "bad candidate" in errors["error"][0]
    assert load_metrics(d).height == 1  # logged before the crash, still there


def test_llm_calls_are_logged_to_the_run(tmp_path: Path) -> None:
    with Run.start("llm", kind="evolve", root=tmp_path) as run:
        client = LLMClient(
            LLMSettings(api_key="test", cache_dir=tmp_path / "cache"), client=FakeBackend(), run=run
        )
        msgs = [{"role": "user", "content": "write a classifier"}]
        client.complete(msgs, purpose="mutate:classifier")
        client.complete(msgs, purpose="mutate:classifier")  # cache hit, still logged
    lines = [
        json.loads(x) for x in (find_run("llm", tmp_path) / "llm.jsonl").read_text().splitlines()
    ]
    assert [x["cached"] for x in lines] == [False, True]
    assert lines[0]["purpose"] == "mutate:classifier" and lines[0]["messages"] == msgs
    assert lines[0]["response"].startswith("reply 1") and lines[0]["prompt_tokens"] == 10


def test_non_finite_values_are_stored(tmp_path: Path) -> None:
    with Run.start("nan", kind="eval", root=tmp_path) as run:
        run.log_metric("fitness", float("-inf"), step=0)
    raw = (find_run("nan", tmp_path) / "metrics.jsonl").read_text()
    assert '"-inf"' in raw


def test_find_run_rejects_ambiguous(tmp_path: Path) -> None:
    for _ in range(2):
        Run.start("same", kind="eval", root=tmp_path).finish()
    with pytest.raises(LookupError):
        find_run("same", tmp_path)
    assert find_run("latest", tmp_path).name.startswith(
        sorted(p.name for p in tmp_path.iterdir())[-1][:15]
    )
