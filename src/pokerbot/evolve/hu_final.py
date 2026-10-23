"""Final heads-up exploiter evaluation (Phase 6 exit, Track B): test sessions, once per champion.

For each program and belief mode (``classifier``: the frozen heads-up classifier, as in
evolution; ``oracle``: true style labels; ``prior``: population shares only) on the 400 test
sessions: bb/100 over the blueprint vs style bots, vs the blueprint itself and vs switchers,
with 95% bootstrap CIs over sessions; per style; KL; fitness. For two programs, a paired
one-sided test of their session scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import ttest_rel

from pokerbot.evolve import hu
from pokerbot.evolve.hu_task import SIZES, HUEvaluator

MODES = ("classifier", "oracle", "prior")


def _ci(v: np.ndarray, boots: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(boots, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5]).tolist()


def evaluate(
    programs: Sequence[tuple[str, str, dict]],
    workers: int = 8,
    split: str = "test",
    modes: Sequence[str] = MODES,
) -> dict:
    out: dict = {"split": split, "sessions": SIZES[split], "programs": {}}
    scores: dict[str, np.ndarray] = {}
    names = hu.profiles().names
    for mode in modes:
        ev = HUEvaluator(workers=workers, allow_test=True, mode=mode)
        try:
            for name, source, params in programs:
                r = ev.evaluate(source, params, split)
                res = r.posteriors
                entry: dict = {"fitness": r.metrics["fitness"], "kl": r.metrics["kl"]}
                for kind in ("style", "nash", "switch"):
                    v = np.array([100 * x["d"].mean() for x in res if x["kind"] == kind])
                    entry[kind] = {"bb100": float(v.mean()), "ci95": _ci(v), "sessions": len(v)}
                for k in range(len(names) - 1):
                    v = np.array(
                        [
                            100 * x["d"].mean()
                            for x in res
                            if x["kind"] == "style" and x["labels"][0] == k
                        ]
                    )
                    if len(v):
                        entry[f"vs {names[k]}"] = {"bb100": float(v.mean()), "ci95": _ci(v)}
                out["programs"].setdefault(name, {})[mode] = entry
                if mode == "classifier":
                    scores[name] = r.session_scores
        finally:
            ev.close()
    if len(programs) == 2 and len(scores) == 2:
        a, b = programs[0][0], programs[1][0]
        d = scores[a] - scores[b]
        out["paired"] = {
            "of": f"{a} - {b}",
            "score_delta": float(d.mean()),
            "ci95": _ci(d),
            "p_one_sided": float(ttest_rel(scores[a], scores[b], alternative="greater").pvalue),
        }
    return out


def safety(
    source: str, params: dict, sessions: int = 600, workers: int = 8, first_index: int = 10_000
) -> dict:
    """bb/100 over the blueprint against the blueprint only, on fresh test-range seeds (session
    indices from ``first_index``, never used by evolution or the main test evaluation). The
    regular test split has only 60 such sessions, too few to bound a loss tightly."""
    from pokerbot.evolve.hu_task import _run

    idx = [
        i
        for i in range(first_index, first_index + 20 * sessions)
        if hu.SessionSpec("test", i).kind == "nash"
    ][:sessions]
    ev = HUEvaluator(workers=workers, allow_test=True)
    try:
        parts = [idx[i::workers] for i in range(workers)]
        jobs = [ev.pool.apply_async(_run, (source, params, "test", p, "classifier")) for p in parts]
        res = [r for j in jobs for r in j.get(3600)["results"]]
    finally:
        ev.close()
    v = np.array([100 * r["d"].mean() for r in res])
    return {
        "sessions": len(v),
        "bb100": float(v.mean()),
        "ci95": _ci(v),
        "first_index": first_index,
    }
