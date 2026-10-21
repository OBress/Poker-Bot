"""Final Leduc exploiter evaluation (Phase 6 exit, Track A): test sessions, run once per champion.

For each program and belief mode (``bayes``: the frozen classifier, as in evolution; ``oracle``:
true style labels; ``prior``: population mix only, i.e. no classifier) on the 200 test sessions:
the edge d (chips/hand vs the Nash baseline, duplicate and exact per hand) against style players,
against the baseline itself and against switchers, each with a 95% bootstrap CI over sessions;
per style; KL; fitness. For two programs, a paired one-sided test of their session scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import ttest_rel

from pokerbot.evolve import leduc
from pokerbot.evolve.leduc_task import LeducEvaluator

MODES = {"bayes": {}, "oracle": {"oracle": True}, "prior": {"prior_only": True}}


def _ci(v: np.ndarray, boots: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(boots, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5]).tolist()


def evaluate(programs: Sequence[tuple[str, str, dict]], workers: int = 8) -> dict:
    out: dict = {"sessions": leduc_task_sizes()["test"], "programs": {}}
    scores: dict[str, np.ndarray] = {}
    for mode, kw in MODES.items():
        ev = LeducEvaluator(workers=workers, allow_test=True, **kw)
        try:
            for name, source, params in programs:
                r = ev.evaluate(source, params, "test")
                res = r.posteriors
                entry: dict = {"fitness": r.metrics["fitness"], "kl": r.metrics["kl"]}
                for kind in ("style", "nash", "switch"):
                    v = np.array([x["d"].mean() for x in res if x["kind"] == kind])
                    entry[kind] = {"mean": float(v.mean()), "ci95": _ci(v), "sessions": len(v)}
                names = list(leduc.STYLES)
                for k, sname in enumerate(names):
                    v = np.array(
                        [x["d"].mean() for x in res if x["kind"] == "style" and x["labels"][0] == k]
                    )
                    entry[f"vs {sname}"] = {"mean": float(v.mean()), "ci95": _ci(v)}
                out["programs"].setdefault(name, {})[mode] = entry
                if mode == "bayes":
                    scores[name] = r.session_scores
        finally:
            ev.close()
    if len(programs) == 2:
        a, b = programs[0][0], programs[1][0]
        d = scores[a] - scores[b]
        out["paired"] = {
            "of": f"{a} - {b}",
            "score_delta": float(d.mean()),
            "ci95": _ci(d),
            "p_one_sided": float(ttest_rel(scores[a], scores[b], alternative="greater").pvalue),
        }
    return out


def leduc_task_sizes() -> dict:
    from pokerbot.evolve.leduc_task import SIZES

    return SIZES
