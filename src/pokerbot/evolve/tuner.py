"""Numeric tuning of a candidate's PARAMS with CMA-ES on train sessions (EVOLUTION.md §5 step 5).

Each parameter is searched in an unbounded space mapped to [low, high] by a logistic function,
starting from the candidate's current values. The objective is −fitness on the first ``n``
train sessions (the same sessions every time, so candidates are compared on equal footing).
"""

from __future__ import annotations

import cma
import numpy as np

from pokerbot.evolve.api import Param
from pokerbot.evolve.evaluator import Evaluator
from pokerbot.evolve.program import ProgramError


def _to_x(v: float, p: Param) -> float:
    u = np.clip((v - p.low) / (p.high - p.low), 1e-3, 1 - 1e-3)
    return float(np.log(u / (1 - u)))


def _to_v(x: float, p: Param) -> float:
    return float(p.low + (p.high - p.low) / (1 + np.exp(-x)))


def tune(
    ev: Evaluator, source: str, spec: dict[str, Param], start: dict[str, float],
    n: int = 96, iters: int = 8, popsize: int = 8, seed: int = 0,
) -> tuple[dict, float]:  # fmt: skip
    """Returns (best params, best train fitness)."""
    names = list(spec)
    best = dict(start)
    best_f = ev.evaluate(source, best, "train", n).metrics["fitness"]
    if not names:
        return best, best_f
    x0 = [_to_x(start[k], spec[k]) for k in names]
    es = cma.CMAEvolutionStrategy(x0, 0.5, {"popsize": popsize, "maxiter": iters, "seed": seed + 1,
                                            "verbose": -9})  # fmt: skip
    while not es.stop():
        xs = es.ask()
        losses = []
        for x in xs:
            params = {k: _to_v(xi, spec[k]) for k, xi in zip(names, x, strict=True)}
            try:
                f = ev.evaluate(source, params, "train", n).metrics["fitness"]
            except ProgramError:
                f = -1e3
            losses.append(-f)
            if f > best_f:
                best, best_f = params, f
        es.tell(xs, losses)
    return best, best_f
