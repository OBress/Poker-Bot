"""Calibrate StyleBot parameters so each style's simulated stats match the real style.

Bots are measured exactly like real players: each focal bot's stat counts over ``hands`` hands go
through the profile's ``FeatureTransform`` (Beta-prior shrinkage, logit, standardization) to a
feature vector z in the style mixture's space. With the style's mean μ_k and spread
sd_kj = sqrt(Σ_kjj) there, the loss for a candidate is
    L(θ) = mean_j ((z̄_j(θ) − μ_kj) / sd_kj)²
where z̄ averages the focal bots' features. L ≈ 1 means the bots sit about one typical
player-to-player difference from the style's center on each stat. (bots-v1/v2 matched pooled
rates instead; the mixture mean is what validation compares against, and it includes the same
shrinkage real players with ~1,000 hands get.)

Calibration is mean-field: a candidate x is the *center* of a style, and is scored by bots
sampled around it (with the style's spread) at tables of bots sampled from the whole population,
exactly as opponents are used later. Table layouts per format are in ``LAYOUT``; all styles are
calibrated in alternating rounds against the latest population.
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

from pokerbot.archetypes.features import FeatureTransform
from pokerbot.opponents.bot import PARAMS, STRUCTURE, STRUCTURE_BOUNDS, StyleBot
from pokerbot.opponents.simulate import simulate_stats
from pokerbot.stats.showdown import LINES, shown_hands

CALIBRATED = tuple(k for k in PARAMS if k != "buyin")
DEPTH_RANGES = ((12.0, 25.0), (25.0, 60.0), (60.0, 150.0), (150.0, 250.0))  # bb, per bucket
DEFAULT_SCALE = 0.33  # spread scale before it is measured (tune_scales); typical measured value
NEW_PARAM_SD = 0.5  # x-space spread given to parameters a warm start doesn't have
# Showdown terms of the loss: mean shown equity per group of betting lines vs real players of the
# style (``opponents.showdown``). A miss of SHOWDOWN_TOL equity counts like a 1σ stat miss.
LINE_GROUPS = {
    "aggressive": ("bbb", "river_bet"),
    "calling": ("call_down", "river_call"),
    "passive": ("bet_check", "call_check", "check_down"),
    "all_in": ("all_in",),
}
GROUP_OF_LINE = {line: g for g, lines in LINE_GROUPS.items() for line in lines}
GROUPS = tuple(LINE_GROUPS)
SHOWDOWN_TOL = 0.04


@dataclass(frozen=True)
class Layout:
    seats: int  # players dealt
    focal: tuple[int, ...]  # seats taken by the style being calibrated
    tables: int  # independent tables per evaluation


LAYOUT = {
    "hu": Layout(2, (0,), 2),
    "sh": Layout(6, (0, 3), 1),
    "fr": Layout(9, (0, 3, 6), 1),
}


Bounds = dict[str, tuple[float, float]]


def _bounds(names: tuple[str, ...], bounds: Bounds | None) -> tuple[np.ndarray, np.ndarray]:
    b = bounds or STRUCTURE_BOUNDS[STRUCTURE]
    return np.array([b[k][0] for k in names]), np.array([b[k][1] for k in names])


def to_params(
    x: np.ndarray, names: tuple[str, ...] = CALIBRATED, bounds: Bounds | None = None
) -> dict[str, float]:
    """Unbounded x -> parameters. ``bounds`` default to the current structure's; a frozen bots
    file decodes with the bounds it was calibrated under."""
    lo, hi = _bounds(names, bounds)
    theta = lo + (hi - lo) / (1 + np.exp(-np.asarray(x)))
    return dict(zip(names, theta.tolist(), strict=True))


def to_x(
    params: dict[str, float], names: tuple[str, ...] = CALIBRATED, bounds: Bounds | None = None
) -> np.ndarray:
    lo, hi = _bounds(names, bounds)
    theta = np.array([params.get(k, PARAMS[k][2]) for k in names])
    u = np.clip((theta - lo) / (hi - lo), 1e-3, 1 - 1e-3)
    return np.log(u / (1 - u))


def draw_hands(quantiles: np.ndarray, rng: np.random.Generator, cap: int) -> tuple[int, float]:
    """A real player's hand count, drawn from the format's percentiles, as (hands to simulate,
    count scale). Above ``cap`` the bot plays ``cap`` hands and its counts are scaled up, which
    keeps the Beta-prior shrinkage of a long-sample player (only the sampling noise is smaller)."""
    h = float(np.interp(rng.uniform(0, 100), np.arange(len(quantiles)), quantiles))
    if h <= cap:
        return int(round(h)), 1.0
    return cap, h / cap


def sample_buyin(depth: np.ndarray, rng: np.random.Generator) -> float:
    lo, hi = DEPTH_RANGES[int(rng.choice(len(depth), p=depth))]
    return float(rng.uniform(lo, hi))


@dataclass
class Target:
    index: int
    name: str
    transform: FeatureTransform
    mean: np.ndarray  # style center μ_k (standardized features)
    sd: np.ndarray  # within-style spread sqrt(diag Σ_k) (standardized features)
    rate: np.ndarray  # pooled real rates (starting point and reports)
    depth: np.ndarray  # stack-depth bucket shares
    share: float
    showdown: np.ndarray | None = None  # real mean shown equity per GROUPS entry (nan: no data)

    @property
    def features(self) -> tuple[str, ...]:
        return self.transform.features


def group_equity(real) -> np.ndarray:
    """(K, G) mean shown equity per style and line group from a real ``RangeTable``; nan where
    the group has fewer than 200 effective shown hands."""
    out = np.full((real.k, len(GROUPS)), np.nan)
    for g, lines in enumerate(LINE_GROUPS.values()):
        idx = [LINES.index(line) for line in lines]
        w = real.counts[:, idx].sum(axis=(1, 2))
        n_eff = w**2 / np.maximum(real.w2[:, idx].sum(axis=1), 1e-12)
        mean = real.eq_sum[:, idx].sum(axis=1) / np.maximum(w, 1e-12)
        out[:, g] = np.where(n_eff >= 200, mean, np.nan)
    return out


def make_targets(
    fmt_profile: dict, depth: np.ndarray, showdown: np.ndarray | None = None
) -> list[Target]:
    """``showdown``: (K, G) real mean shown equity per line group (``group_equity``)."""
    t = fmt_profile["transform"]
    return [
        Target(
            index=s["index"],
            name=s["name"],
            transform=t,
            mean=np.array(s["mean"]),
            sd=np.sqrt(np.diag(s["cov"])),
            rate=np.array([s["pooled_rate"][f] for f in t.features]),
            depth=depth[s["index"]],
            share=s["share"],
            showdown=None if showdown is None else showdown[s["index"]],
        )
        for s in fmt_profile["styles"]
    ]


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


def simulate_features(
    params: dict[str, float],
    target: Target,
    pop: Population,
    seed: int,
    hands: int,
    focal_noise: np.ndarray | None = None,
    layout: Layout = LAYOUT["sh"],
    hands_q: np.ndarray | None = None,
) -> np.ndarray:
    """Mean standardized feature vector of the focal bots, each measured over ``hands`` hands,
    or, with ``hands_q`` (percentiles of real players' hand counts), over a hand count drawn per
    table from it with ``hands`` as the cap (``draw_hands``).
    With ``focal_noise``, each focal seat is a different bot sampled around ``params``.

    Returns D features followed by 3·G showdown sums of the focal bots' shown hands per line
    group: equity sums, squared-equity sums, counts (``split`` separates them)."""
    rng = np.random.default_rng(seed)
    zs = []
    sd = np.zeros((3, len(GROUPS)))
    for _ in range(layout.tables):
        sim_hands, scale = (hands, 1.0) if hands_q is None else draw_hands(hands_q, rng, hands)
        others = pop.bots(rng, layout.seats - len(layout.focal))
        seats, labels = [], []
        for s in range(layout.seats):
            if s in layout.focal:
                seats.append(make_bot(params, focal_noise, target.depth, rng))
                labels.append(f"focal{s}")
            else:
                seats.append(others.pop())
                labels.append("other")

        def on_hand(events) -> None:
            for sh in shown_hands(events):
                if sh.seat in layout.focal:
                    g = GROUPS.index(GROUP_OF_LINE[sh.line])
                    sd[:, g] += (sh.equity, sh.equity**2, 1.0)

        track = on_hand if target.showdown is not None else None
        counts = simulate_stats(seats, labels, sim_hands, int(rng.integers(2**31)), on_hand=track)
        for s in layout.focal:
            c = counts[f"focal{s}"]
            n = scale * np.array([[c.get(f, (0, 0))[0] for f in target.features]], dtype=float)
            x = scale * np.array([[c.get(f, (0, 0))[1] for f in target.features]], dtype=float)
            zs.append(target.transform.transform(n, x)[0][0])
    return np.concatenate([np.mean(zs, axis=0), sd.ravel()])


def split(sim: np.ndarray, target: Target) -> tuple[np.ndarray, np.ndarray]:
    """(features, showdown sums (3, G)) of a ``simulate_features`` vector."""
    d = len(target.features)
    return sim[:d], sim[d:].reshape(3, len(GROUPS))


def logit(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return np.log(r / (1 - r))


def feature_z(sim: np.ndarray, target: Target) -> np.ndarray:
    """Per-stat distance of simulated features from the style center, in style σ."""
    return (split(sim, target)[0] - target.mean) / target.sd


def showdown_z(sim: np.ndarray, target: Target) -> np.ndarray:
    """Per line group: (sim − real mean shown equity) / sqrt(tol² + sim standard error²).
    Groups without real data or simulated shown hands give 0."""
    if target.showdown is None:
        return np.zeros(0)
    total, sq, n = split(sim, target)[1]
    mean = total / np.maximum(n, 1)
    var = np.maximum(sq / np.maximum(n, 1) - mean**2, 0.0)
    se2 = np.where(n > 1, var / np.maximum(n, 1), 0.25)
    ok = (n > 0) & np.isfinite(target.showdown)
    z = (mean - np.nan_to_num(target.showdown)) / np.sqrt(SHOWDOWN_TOL**2 + se2)
    return np.where(ok, z, 0.0)


def loss(sim: np.ndarray, target: Target) -> float:
    return float(np.mean(np.concatenate([feature_z(sim, target), showdown_z(sim, target)]) ** 2))


@dataclass
class StyleResult:
    index: int
    x: np.ndarray
    loss: float
    history: list[dict] = field(default_factory=list)  # per generation
    samples_x: list[list[float]] = field(default_factory=list)
    samples_z: list[list[float]] = field(default_factory=list)


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
    layout: Layout = LAYOUT["sh"],
    hands_q: np.ndarray | None = None,
) -> StyleResult:
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {"popsize": popsize, "seed": seed + 1, "maxiter": iters, "verbose": -9}
    )
    result = StyleResult(target.index, x0, float("inf"))
    gen = 0
    while not es.stop():
        xs = es.ask()
        gen_seed = seed * 10_000 + gen  # common random numbers within a generation
        sims = [
            simulate_features(
                to_params(x), target, pop, gen_seed, hands, focal_noise, layout, hands_q
            )
            for x in xs
        ]
        losses = [loss(s, target) for s in sims]
        es.tell(xs, losses)
        result.samples_x += [list(map(float, x)) for x in xs]
        result.samples_z += [s.tolist() for s in sims]
        result.history.append(
            {"gen": gen, "best": min(losses), "mean": float(np.mean(losses)), "sigma": es.sigma}
        )
        gen += 1
    result.x = np.asarray(es.result.xfavorite)  # distribution mean: robust to evaluation noise
    final = final_features(result.x, target, pop, seed, hands, focal_noise, layout, hands_q)
    result.loss = loss(final, target)
    return result


def final_features(
    x: np.ndarray,
    target: Target,
    pop: Population,
    seed: int,
    hands: int,
    focal_noise: np.ndarray | None,
    layout: Layout,
    hands_q: np.ndarray | None = None,
    repeats: int = 4,
) -> np.ndarray:
    """Features averaged over several fresh tables (same hand-count rule: shrinkage depends on
    it, so more hands per bot would not be the same measurement)."""
    return np.mean(
        [
            simulate_features(
                to_params(x),
                target,
                pop,
                seed * 10_000 + 9_000 + r,
                hands,
                focal_noise,
                layout,
                hands_q,
            )
            for r in range(repeats)
        ],
        axis=0,
    )


def _run(args) -> StyleResult:
    return calibrate_style(*args)


def spread(result: StyleResult, target: Target, ridge: float = 1e-2) -> np.ndarray:
    """Parameter covariance (in x space) that reproduces the real within-style spread.

    Fits sim features ≈ A x + b on the samples near the optimum, then maps the real spread
    back through the pseudo-inverse: Σ_x = A⁺ Σ_real A⁺ᵀ (Σ_real diagonal, standardized)."""
    X = np.array(result.samples_x)
    Y = np.array(result.samples_z)[:, : len(target.features)]
    d = np.linalg.norm(X - result.x, axis=1)
    near = d <= np.quantile(d, 0.5)
    Xc = np.column_stack([X[near] - result.x, np.ones(near.sum())])
    coef, *_ = np.linalg.lstsq(Xc, Y[near], rcond=None)
    A = coef[:-1].T  # (features, params)
    pinv = np.linalg.pinv(A, rcond=1e-2)
    cov = pinv @ np.diag(target.sd**2) @ pinv.T
    return cov + ridge * np.eye(len(CALIBRATED))


def warm_style(style: dict, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(x_mean, x_cov) of an earlier bots file's style in the current parameter space.
    Parameters it lacks start at their defaults with an independent spread of NEW_PARAM_SD."""
    params = {**{k: PARAMS[k][2] for k in CALIBRATED}, **style["params"]}
    x = to_x(params)
    cov = np.diag(np.full(len(CALIBRATED), NEW_PARAM_SD**2))
    idx = [CALIBRATED.index(k) for k in names]
    cov[np.ix_(idx, idx)] = np.array(style["x_cov"])
    return x, cov


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
    warm_names: tuple[str, ...] = CALIBRATED,
    layout: Layout = LAYOUT["sh"],
    hands_q: np.ndarray | None = None,
) -> tuple[list[StyleResult], Population]:
    """``warm``: styles from an earlier bots file (params, x_cov, x_scale; ``warm_names`` its
    calibrated parameters) to start from; their spread is used from round 0. Without it, round 0
    calibrates centers with no spread."""
    depth = [t.depth for t in targets]
    share = np.array([t.share for t in targets])
    if warm:
        starts = [warm_style(w, warm_names) for w in warm]
        params = [to_params(x) for x, _ in starts]
        scales = [w.get("x_scale") or DEFAULT_SCALE for w in warm]
        noise = [_noise(cov, sc) for (_, cov), sc in zip(starts, scales, strict=True)]
        sigma0 = 0.3
    else:
        params = [init_params(dict(zip(t.features, t.rate.tolist(), strict=True))) for t in targets]
        scales = [DEFAULT_SCALE] * len(targets)
        noise = [None] * len(targets)
        sigma0 = 0.6
    pop = Population(params, depth, share, noise, scales)
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
                layout,
                hands_q,
            )
            for i, t in enumerate(targets)
        ]
        with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
            results = list(pool.map(_run, jobs))
        noise = [
            _noise(spread(res, t), scales[i])
            for i, (res, t) in enumerate(zip(results, targets, strict=True))
        ]
        pop = Population([to_params(res.x) for res in results], depth, share, noise, scales)
        if log:
            log(r, results)
    return results, pop


def _noise(cov: np.ndarray, scale: float) -> np.ndarray:
    return scale * np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))


def write_bots(path: Path, version: str, run_id: str, formats: dict[str, dict]) -> None:
    """Write a bots file. If ``path`` already holds the same version and structure (formats
    calibrated by separate runs), the new formats are merged in; each format records its run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"version": version, "structure": STRUCTURE, "formats": {}}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("version") == version and old.get("structure") == STRUCTURE:
            doc = old
    doc["formats"].update({f: {**d, "run_id": run_id} for f, d in formats.items()})
    doc["run_id"] = {f: d["run_id"] for f, d in doc["formats"].items()}
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
