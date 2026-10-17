"""Calibrate StyleBot parameters so each style's simulated stats match the real style.

Loss for a style with target pooled rates t_j and real within-style spread σ_j (logit units):
    L(θ) = mean_j ((logit ŝ_j(θ) − logit t_j) / σ_j)²
where ŝ_j are the bot's simulated rates. L ≈ 1 means the bot sits about one typical
player-to-player difference from the style's center on each stat.

Calibration is mean-field: a candidate x is the *center* of a style, and is scored by the stats
of bots sampled around it (with the style's spread) at tables of bots sampled from the whole
population, exactly as opponents are used later. The focal style takes 2 of 6 seats; all styles
are calibrated in alternating rounds against the latest population.
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
FOCAL_SEATS = (0, 3)
DEFAULT_SCALE = 0.33  # spread scale before it is measured (tune_scales); typical measured value


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
    noise: list[np.ndarray | None] | None = None  # scaled Cholesky factor of each style spread
    scales: list[float] | None = None

    def bots(self, rng: np.random.Generator, count: int) -> list[StyleBot]:
        styles = rng.choice(len(self.params), size=count, p=self.share / self.share.sum())
        out = []
        for k in styles:
            noise = self.noise[k] if self.noise else None
            out.append(make_bot(self.params[k], noise, self.depth[k], rng))
        return out


def make_bot(
    center: dict[str, float], noise: np.ndarray | None, depth: np.ndarray, rng: np.random.Generator
) -> StyleBot:
    """A bot drawn around a style center: x ~ N(x_center, noise @ noise.T); buy-in from depth."""
    params = dict(center)
    if noise is not None:
        params = to_params(to_x(center) + noise @ rng.standard_normal(noise.shape[1]))
    return StyleBot({**params, "buyin": sample_buyin(depth, rng)}, int(rng.integers(2**31)))


def simulate_rates(
    params: dict[str, float],
    target: Target,
    pop: Population,
    seed: int,
    hands: int,
    focal_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Focal style's simulated rates for the target's features (smoothed so logit is finite).
    With ``focal_noise``, each focal seat is a different bot sampled around ``params``."""
    rng = np.random.default_rng(seed)
    others = pop.bots(rng, SEATS - len(FOCAL_SEATS))
    seats, labels = [], []
    for s in range(SEATS):
        if s in FOCAL_SEATS:
            seats.append(make_bot(params, focal_noise, target.depth, rng))
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
    target: Target,
    pop: Population,
    x0: np.ndarray,
    iters: int,
    popsize: int,
    hands: int,
    seed: int,
    focal_noise: np.ndarray | None = None,
    sigma0: float = 0.6,
) -> StyleResult:
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {"popsize": popsize, "seed": seed + 1, "maxiter": iters, "verbose": -9}
    )
    result = StyleResult(target.index, x0, float("inf"))
    gen = 0
    while not es.stop():
        xs = es.ask()
        gen_seed = seed * 10_000 + gen  # common random numbers within a generation
        sims = [simulate_rates(to_params(x), target, pop, gen_seed, hands, focal_noise) for x in xs]
        losses = [loss(s, target) for s in sims]
        es.tell(xs, losses)
        result.samples_x += [list(map(float, x)) for x in xs]
        result.samples_logit += [logit(s).tolist() for s in sims]
        result.history.append(
            {"gen": gen, "best": min(losses), "mean": float(np.mean(losses)), "sigma": es.sigma}
        )
        gen += 1
    result.x = np.asarray(es.result.xfavorite)  # distribution mean: robust to evaluation noise
    final = simulate_rates(
        to_params(result.x), target, pop, seed * 10_000 + 9_999, hands * 3, focal_noise
    )
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
    warm: list[dict] | None = None,
) -> tuple[list[StyleResult], Population]:
    """``warm``: styles from an earlier bots file (x_mean, x_cov, x_scale) to start from; their
    spread is used from round 0. Without it, round 0 calibrates centers with no spread."""
    depth = [t.depth for t in targets]
    share = np.array([t.share for t in targets])
    if warm:
        params = [to_params(np.array(w["x_mean"])) for w in warm]
        scales = [w.get("x_scale") or DEFAULT_SCALE for w in warm]
        noise = [_noise(np.array(w["x_cov"]), sc) for w, sc in zip(warm, scales, strict=True)]
        sigma0 = 0.3
    else:
        params = [init_params(dict(zip(t.features, t.rate.tolist(), strict=True))) for t in targets]
        scales = [DEFAULT_SCALE] * len(targets)
        noise = [None] * len(targets)
        sigma0 = 0.6
    pop = Population(params, depth, share, noise)
    results: list[StyleResult] = []
    for r in range(rounds):
        jobs = [
            (
                t,
                pop,
                to_x(pop.params[i]),
                iters,
                popsize,
                hands,
                seed + 97 * r + i,
                noise[i],
                sigma0,
            )
            for i, t in enumerate(targets)
        ]
        with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
            results = list(pool.map(_run, jobs))
        noise = [
            _noise(spread(res, t), scales[i])
            for i, (res, t) in enumerate(zip(results, targets, strict=True))
        ]
        pop = Population([to_params(res.x) for res in results], depth, share, noise)
        pop.scales = scales
        if log:
            log(r, results)
    return results, pop


def _noise(cov: np.ndarray, scale: float) -> np.ndarray:
    return scale * np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))


def write_bots(path: Path, version: str, run_id: str, formats: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": version, "run_id": run_id, "formats": formats}, indent=1),
        encoding="utf-8",
    )
