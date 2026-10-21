"""Final classifier evaluation (Phase 5 exit): test seeds and real players, run once per champion.

For every program (with its tuned PARAMS):
  test       synthetic test sessions (never seen by evolution): log loss, accuracy, Brier, ECE
             per checkpoint, with 95% bootstrap CIs over sessions;
  real_val   fit-set real players; real_test held-out real players: cross-entropy H(q, p) and
             accuracy against argmax q per checkpoint (MATH.md §6.2), CIs over players.
For a pair (first vs second program): per-session paired differences of the weighted log loss
on test (one-sided t-test and bootstrap CI), and accuracy differences per checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from pokerbot.evolve import dataset, harness, program, realdata, scoring
from pokerbot.evolve.api import StyleProfiles


def _ci(values: np.ndarray, boots: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(boots, len(values)))
    return np.percentile(values[idx].mean(axis=1), [2.5, 97.5]).tolist()


def _posteriors(source: str, params: dict, prof: StyleProfiles, sessions) -> list[np.ndarray]:
    ns = program.load(source)
    r = harness.run_classifier(lambda: ns["Classifier"](prof, dict(params)), prof, sessions)
    return r["posteriors"]


def evaluate_programs(programs: Sequence[tuple[str, str, dict]], fmt: str, profiles_path: Path,
                      showdown_path: Path, real_path: Path | None) -> dict:  # fmt: skip
    """``programs``: (name, source, params). Reads the test split and held-out players."""
    prof = StyleProfiles.load(profiles_path, fmt, showdown_path)
    test = dataset.load(fmt, "test", allow_test=True)
    labels = [s["labels"] for s in test]
    real = {}
    if real_path and Path(real_path).exists():
        for g in ("real_val", "real_test"):
            real[g] = realdata.load_real(real_path, g, allow_test=True)
    out: dict = {"test_sessions": len(test), "real_players": {g: len(v) for g, v in real.items()},
                 "programs": {}}  # fmt: skip
    session_scores = {}
    accs = {}
    for name, source, params in programs:
        posts = _posteriors(source, params, prof, test)
        m = scoring.score(posts, labels, 0.0)
        ll = scoring.per_seat_ll(posts, labels)
        y = np.concatenate([np.asarray(v) for v in labels])
        p_all = np.concatenate(posts)
        entry: dict = {"test": {}}
        for i, t in enumerate(harness.CHECKPOINTS):
            hit = (p_all[:, i].argmax(axis=1) == y).astype(float)
            entry["test"][t] = {"logloss": m[f"logloss@{t}"], "logloss_ci": _ci(ll[:, i]),
                                "accuracy": m[f"accuracy@{t}"], "accuracy_ci": _ci(hit),
                                "brier": m[f"brier@{t}"], "ece": m[f"ece@{t}"]}  # fmt: skip
        entry["test"]["fitness"] = m["fitness"]
        session_scores[name] = scoring.session_scores(posts, labels)
        accs[name] = p_all.argmax(axis=2) == y[:, None]
        for g, sess in real.items():
            rp = np.concatenate(_posteriors(source, params, prof, sess))  # (players, T, K)
            q = np.array([s["q"] for s in sess])
            entry[g] = {}
            for i, t in enumerate(harness.CHECKPOINTS):
                xent = -(q * np.log(np.maximum(rp[:, i], scoring.FLOOR))).sum(axis=1)
                hit = (rp[:, i].argmax(axis=1) == q.argmax(axis=1)).astype(float)
                entry[g][t] = {"xent": float(xent.mean()), "xent_ci": _ci(xent),
                               "accuracy": float(hit.mean()), "accuracy_ci": _ci(hit)}  # fmt: skip
        out["programs"][name] = entry
    if len(programs) == 2:
        a, b = programs[0][0], programs[1][0]
        d = session_scores[a] - session_scores[b]
        out["paired"] = {
            "of": f"{a} - {b}", "score_delta": float(d.mean()), "score_delta_ci": _ci(d),
            "p_one_sided": float(ttest_rel(session_scores[a], session_scores[b],
                                           alternative="greater").pvalue),
            "accuracy_delta": {t: float(accs[a][:, i].mean() - accs[b][:, i].mean())
                               for i, t in enumerate(harness.CHECKPOINTS)},
        }  # fmt: skip
    return out
