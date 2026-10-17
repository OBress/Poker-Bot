"""Calibrate StyleBot parameters so each style's simulated stats match the real style.

Loss for a style with target pooled rates t_j and real within-style spread σ_j (logit units):
    L(θ) = mean_j ((logit ŝ_j(θ) − logit t_j) / σ_j)²
where ŝ_j are the bot's simulated rates. L ≈ 1 means the bot sits about one typical
player-to-player difference from the style's center on each stat.

Simulated stats depend on the opponents, so tables mix the focal bot (3 of 6 seats) with bots of
the other styles drawn by population share; all styles are calibrated in alternating rounds.
CMA-ES works in an unbounded space x, mapped to each parameter's bounds by a logistic function.
Within a generation every candidate faces the same cards (common random numbers).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cma
import numpy as np

from pokerbot.opponents.bot import PARAMS, StyleBot
from pokerbot.opponents.simulate import simulate_stats

CALIBRATED = tuple(k for k in PARAMS if k != "buyin")
LO = np.array([PARAMS[k][0] for k in CALIBRATED])
HI = np.array([PARAMS[k][1] for k in CALIBRATED])
DEPTH_RANGES = ((12.0, 25.0), (25.0, 60.0), (60.0, 150.0), (150.0, 250.0))  # bb, per bucket
SEATS = 6
FOCAL_SEATS = (0, 2, 4)


def to_params(x: np.ndarray) -> dict[str, float]:
    theta = LO + (HI - LO) / (1 + np.exp(-np.asarray(x)))
    return dict(zip(CALIBRATED, theta.tolist(), strict=True))


def to_x(params: dict[str, float]) -> np.ndarray:
    theta = np.array([params[k] for k in CALIBRATED])
    u = np.clip((theta - LO) / (HI - LO), 1e-3, 1 - 1e-3)
    return np.log(u / (1 - u))


def sample_buyin(depth: np.ndarray, rng: np.random.Generator) -> float:
    lo, hi = DEPTH_RANGES[int(rng.choice(len(depth), p=depth))]
    return float(rng.uniform(lo, hi))


@dataclass
class Target:
    index: int
    name: str
    features: tuple[str, ...]
    rate: np.ndarray  # pooled real rates
    sigma: np.ndarray  # within-style spread of logit rates
    depth: np.ndarray  # stack-depth bucket shares
    share: float


def make_targets(fmt_profile: dict, depth: np.ndarray) -> list[Target]:
    t = fmt_profile["transform"]
    out = []
    for s in fmt_profile["styles"]:
        out.append(
            Target(
                index=s["index"],
                name=s["name"],
                features=t.features,
                rate=np.array([s["pooled_rate"][f] for f in t.features]),
                sigma=np.sqrt(np.diag(s["cov"])) * t.std,
                depth=depth[s["index"]],
                share=s["share"],
            )
        )
    return out


def init_params(rate: dict[str, float]) -> dict[str, float]:
    """Analytic starting point: preflop widths read almost directly off the stats."""
    p = {k: v[2] for k, v in PARAMS.items()}
    defaults = {"rfi_BTN": 0.4, "limp": 0.03, "3bet": 0.05, "vpip": 0.25, "pfr": 0.17}
    defaults |= {"fold_to_3bet": 0.55, "fold_to_steal": 0.75, "cbet_flop": 0.65}
    r = defaults | rate  # stats missing in a format fall back to typical values
    p["open_ep"] = r.get("rfi_EP", r["rfi_BTN"] * 0.4)
    p["open_btn"] = r["rfi_BTN"]
    p["limp_width"] = r["limp"]
    p["three_bet"] = r["3bet"]
    p["call_width"] = max(0.0, r["vpip"] - r["pfr"] - r["limp"]) * 0.8
    p["cont_vs_3bet"] = 1 - r["fold_to_3bet"]
    widths = p["three_bet"] + p["call_width"]
    p["defend"] = (1 - r["fold_to_steal"]) / max(widths, 0.02)
    p["cbet_bonus"] = r["cbet_flop"] * 0.6
    return {k: float(np.clip(v, PARAMS[k][0], PARAMS[k][1])) for k, v in p.items()}


@dataclass
class Population:
    """The other styles' current parameters, used to fill non-focal seats."""

    params: list[dict[str, float]]
    depth: list[np.ndarray]
    share: np.ndarray

    def bots(self, rng: np.random.Generator, count: int) -> list[StyleBot]:
        styles = rng.choice(len(self.params), size=count, p=self.share / self.share.sum())
        return [
            StyleBot(
                {**self.params[k], "buyin": sample_buyin(self.depth[k], rng)},
                int(rng.integers(2**31)),
            )
            for k in styles
        ]


def simulate_rates(
    params: dict[str, float], target: Target, pop: Population, seed: int, hands: int
) -> np.ndarray:
    """Focal bot's simulated rates for the target's features (smoothed so logit is finite)."""
    rng = np.random.default_rng(seed)
    others = pop.bots(rng, SEATS - len(FOCAL_SEATS))
    seats, labels = [], []
    for s in range(SEATS):
        if s in FOCAL_SEATS:
            buyin = sample_buyin(target.depth, rng)
            seats.append(StyleBot({**params, "buyin": buyin}, int(rng.integers(2**31))))
            labels.append("focal")
        else:
            seats.append(others.pop())
            labels.append("other")
    counts = simulate_stats(seats, labels, hands, int(rng.integers(2**31)))["focal"]
    out = []
    for f in target.features:
        n, x = counts.get(f, (0, 0))
        out.append((x + 0.5) / (n + 1.0))
    return np.array(out)


def logit(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return np.log(r / (1 - r))


def loss(sim: np.ndarray, target: Target) -> float:
    return float(np.mean(((logit(sim) - logit(target.rate)) / target.sigma) ** 2))


@dataclass
class StyleResult:
    index: int
    x: np.ndarray
    loss: float
    history: list[dict] = field(default_factory=list)  # per generation
    samples_x: list[list[float]] = field(default_factory=list)
    samples_logit: list[list[float]] = field(default_factory=list)


def calibrate_style(
    target: Target, pop: Population, x0: np.ndarray, iters: int, popsize: int, hands: int, seed: int
) -> StyleResult:
    es = cma.CMAEvolutionStrategy(
        x0, 0.6, {"popsize": popsize, "seed": seed + 1, "maxiter": iters, "verbose": -9}
    )
    result = StyleResult(target.index, x0, float("inf"))
    gen = 0
    while not es.stop():
        xs = es.ask()
        gen_seed = seed * 10_000 + gen  # common random numbers within a generation
        sims = [simulate_rates(to_params(x), target, pop, gen_seed, hands) for x in xs]
        losses = [loss(s, target) for s in sims]
        es.tell(xs, losses)
        result.samples_x += [list(map(float, x)) for x in xs]
        result.samples_logit += [logit(s).tolist() for s in sims]
        result.history.append(
            {"gen": gen, "best": min(losses), "mean": float(np.mean(losses)), "sigma": es.sigma}
        )
        gen += 1
    result.x = np.asarray(es.result.xfavorite)  # distribution mean: robust to evaluation noise
    final = simulate_rates(to_params(result.x), target, pop, seed * 10_000 + 9_999, hands * 3)
    result.loss = loss(final, target)
    return result


def _run(args) -> StyleResult:
    return calibrate_style(*args)


def spread(result: StyleResult, target: Target, ridge: float = 1e-2) -> np.ndarray:
    """Parameter covariance (in x space) that reproduces the real within-style spread.

    Fits sim logit rates ≈ A x + b on the samples near the optimum, then maps the real spread
    back through the pseudo-inverse: Σ_x = A⁺ Σ_real A⁺ᵀ (Σ_real diagonal in logit units)."""
    X = np.array(result.samples_x)
    Y = np.array(result.samples_logit)
    d = np.linalg.norm(X - result.x, axis=1)
    near = d <= np.quantile(d, 0.5)
    Xc = np.column_stack([X[near] - result.x, np.ones(near.sum())])
    coef, *_ = np.linalg.lstsq(Xc, Y[near], rcond=None)
    A = coef[:-1].T  # (features, params)
    pinv = np.linalg.pinv(A, rcond=1e-2)
    cov = pinv @ np.diag(target.sigma**2) @ pinv.T
    return cov + ridge * np.eye(len(CALIBRATED))


def calibrate_format(
    targets: list[Target],
    rounds: int,
    iters: int,
    popsize: int,
    hands: int,
    workers: int,
    seed: int,
    log=None,
) -> tuple[list[StyleResult], Population]:
    params = [init_params(dict(zip(t.features, t.rate.tolist(), strict=True))) for t in targets]
    pop = Population(params, [t.depth for t in targets], np.array([t.share for t in targets]))
    results: list[StyleResult] = []
    for r in range(rounds):
        jobs = [
            (t, pop, to_x(pop.params[i]), iters, popsize, hands, seed + 97 * r + i)
            for i, t in enumerate(targets)
        ]
        with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
            results = list(pool.map(_run, jobs))
        pop = Population(
            [{**to_params(res.x), "buyin": 100.0} for res in results], pop.depth, pop.share
        )
        if log:
            log(r, results)
    return results, pop


def write_bots(path: Path, version: str, run_id: str, formats: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": version, "run_id": run_id, "formats": formats}, indent=1),
        encoding="utf-8",
    )
