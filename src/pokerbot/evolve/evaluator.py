"""Sandboxed scoring of candidate programs (EVOLUTION.md §4–§5).

A pool of worker processes preloads the allowed dataset splits once (never ``test``) and runs
candidates there, never in the evolution process. Each evaluation splits the sessions across the
workers; if any part exceeds the time limit the whole pool is terminated and rebuilt, and the
candidate fails with ``timeout``. Exceptions inside a candidate fail it with ``crash`` (last
traceback lines kept for the repair prompt).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass

import numpy as np

from pokerbot.evolve import dataset, harness, program, scoring
from pokerbot.evolve.api import StyleProfiles
from pokerbot.evolve.program import ProgramError

_DATA: dict[str, list[dict]] = {}
_PROFILES: list[StyleProfiles] = []
EVOLVE_SPLITS = ("train", "val_a", "val_b")


def _init(
    fmt: str, profiles_path: str, showdown_path: str, splits: tuple[str, ...], root: str,
    equilibrium: str | None = None,
) -> None:  # fmt: skip
    for split in splits:
        _DATA[split] = dataset.load(fmt, split, root=root)  # refuses "test"
    _PROFILES.append(StyleProfiles.load(profiles_path, fmt, showdown_path, equilibrium))


def _run(source: str, params: dict, split: str, idx: list[int], reverse: bool):
    try:
        ns = program.load(source)
        prof = _PROFILES[0]
        sessions = [_DATA[split][i] for i in idx]
        if reverse:
            sessions = sessions[::-1]
        r = harness.run_classifier(lambda: ns["Classifier"](prof, dict(params)), prof, sessions)
        posts = r["posteriors"][::-1] if reverse else r["posteriors"]
        return {"ok": True, "posteriors": posts, "labels": [_DATA[split][i]["labels"] for i in idx],
                "invalid": r["invalid"], "calls": r["calls"], "us": r["us_per_call"]}  # fmt: skip
    except ProgramError as e:
        return {"ok": False, "reason": e.reason, "detail": e.detail}
    except Exception:  # noqa: BLE001 - a candidate crash is a result, not our failure
        tb = traceback.format_exc().splitlines()
        return {"ok": False, "reason": "crash", "detail": "\n".join(tb[-6:])}


@dataclass
class Result:
    metrics: dict
    session_scores: np.ndarray
    posteriors: list
    labels: list
    invalid: int


class Evaluator:
    def __init__(self, fmt: str, profiles_path: str, showdown_path: str, workers: int = 8,
                 timeout_s: float = 120.0, root=dataset.ROOT,
                 equilibrium: str | None = None) -> None:  # fmt: skip
        self.args = (fmt, str(profiles_path), str(showdown_path), EVOLVE_SPLITS, str(root),
                     str(equilibrium) if equilibrium else None)  # fmt: skip
        self.workers = workers
        self.timeout_s = timeout_s
        self.sizes = {s: len(dataset.load(fmt, s, root=root)) for s in EVOLVE_SPLITS}
        self._start()

    def _start(self) -> None:
        self.pool = mp.get_context("spawn").Pool(self.workers, _init, self.args)

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()

    def evaluate(self, source: str, params: dict, split: str, n: int | None = None,
                 reverse: bool = False) -> Result:  # fmt: skip
        """Score on the first ``n`` sessions of ``split``. Raises ``ProgramError``."""
        if split not in EVOLVE_SPLITS:
            raise PermissionError(split)
        idx = list(range(min(n or self.sizes[split], self.sizes[split])))
        parts = [idx[i :: self.workers] for i in range(self.workers) if idx[i :: self.workers]]
        jobs = [self.pool.apply_async(_run, (source, params, split, p, reverse)) for p in parts]
        outs = []
        try:
            for j in jobs:
                outs.append(j.get(self.timeout_s))
        except mp.TimeoutError:
            self.close()
            self._start()
            raise ProgramError("timeout", f"evaluation exceeded {self.timeout_s:.0f}s") from None
        for o in outs:
            if not o["ok"]:
                raise ProgramError(o["reason"], o["detail"])
        order = np.argsort(np.concatenate(parts))
        posts = [p for o in outs for p in o["posteriors"]]
        labels = [y for o in outs for y in o["labels"]]
        posts = [posts[i] for i in order]
        labels = [labels[i] for i in order]
        calls = sum(o["calls"] for o in outs)
        us = sum(o["us"] * o["calls"] for o in outs) / max(calls, 1)
        metrics = scoring.score(posts, labels, us)
        metrics["invalid"] = sum(o["invalid"] for o in outs)
        return Result(metrics, scoring.session_scores(posts, labels), posts, labels,
                      metrics["invalid"])  # fmt: skip

    def gate(self, source: str, params: dict) -> None:
        """Static checks plus a short run that must be valid and order-independent."""
        program.check(source)
        a = self.evaluate(source, params, "train", n=16)
        if a.invalid:
            raise ProgramError("invalid_output", f"{a.invalid} invalid posteriors on 16 sessions")
        b = self.evaluate(source, params, "train", n=16, reverse=True)
        if not all(np.allclose(x, y) for x, y in zip(a.posteriors, b.posteriors, strict=True)):
            raise ProgramError("state_leak", "results depend on session order")
