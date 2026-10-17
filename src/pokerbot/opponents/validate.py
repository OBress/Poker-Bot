"""Phase 3 exit check: do sampled bots look like the real players of their style?

For each style, sample many independent bots (spread = 1), seat them at mixed tables, measure each
bot's stats over a realistic sample of hands, and compare per stat:
  center:  (mean simulated logit − real style logit) / real σ      -> |z| should be < 1
  spread:  std of simulated logits / real σ                         -> ratio should be ~0.5–2
Real σ is the within-style spread of real players (MATH.md §3), so "within the real style's
spread" is literally |center z| < 1.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pokerbot.archetypes.profiles import load_profiles
from pokerbot.opponents.calibrate import logit
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.simulate import simulate_stats


def _table(args) -> list[tuple[int, dict]]:
    bots_path, fmt, focus, seats, hands, seed, spread = args
    rng = np.random.default_rng(seed)
    sampler = StyleSampler(bots_path, fmt, spread=spread)
    draws = [sampler.draw(rng, style=focus)] + sampler.table(rng, seats - 1)
    labels = [f"s{i}" for i in range(seats)]
    counts = simulate_stats([d.bot for d in draws], labels, hands, int(rng.integers(2**31)))
    return [(d.style, counts[labels[i]]) for i, d in enumerate(draws)]


def validate(
    bots_path: Path,
    profiles_path: Path,
    fmt: str = "sh",
    tables_per_style: int = 12,
    seats: int = 6,
    hands: int = 1500,
    workers: int = 8,
    seed: int = 7,
    spread: float = 1.0,
) -> dict:
    prof = load_profiles(profiles_path)["formats"][fmt]
    features = prof["transform"].features
    std = prof["transform"].std
    k = len(prof["styles"])
    jobs = [
        (bots_path, fmt, s, seats, hands, seed * 100_000 + s * 1000 + t, spread)
        for s in range(k)
        for t in range(tables_per_style)
    ]
    per_style: dict[int, list[np.ndarray]] = {s: [] for s in range(k)}
    with ProcessPoolExecutor(workers) as pool:
        for table in pool.map(_table, jobs):
            for style, counts in table:
                r = []
                for f in features:
                    n, x = counts.get(f, (0, 0))
                    r.append((x + 0.5) / (n + 1.0))
                per_style[style].append(logit(np.array(r)))
    report = {}
    for s in range(k):
        st = prof["styles"][s]
        sims = np.array(per_style[s])
        target = logit(np.array([st["pooled_rate"][f] for f in features]))
        sigma = np.sqrt(np.diag(st["cov"])) * std
        center = (sims.mean(axis=0) - target) / sigma
        ratio = sims.std(axis=0) / sigma
        report[st["name"]] = {
            "bots": len(sims),
            "stats": {
                f: {"center_z": float(center[j]), "spread_ratio": float(ratio[j])}
                for j, f in enumerate(features)
            },
            "max_abs_center_z": float(np.abs(center).max()),
            "median_spread_ratio": float(np.median(ratio)),
            "within_spread": int((np.abs(center) < 1).sum()),
            "features": len(features),
        }
    return report


def tune_scales(
    bots_path: Path,
    profiles_path: Path,
    fmt: str = "sh",
    scales: tuple[float, ...] = (0.15, 0.3, 0.5, 0.75),
    workers: int = 8,
    log=None,
) -> dict[str, float]:
    """Per style, the covariance scale at which sampled bots' median spread ratio is ~1.

    The linear-response estimate of the parameter covariance overshoots (stats respond
    nonlinearly, and insensitive parameter directions get huge variance), so the scale is measured
    by simulation. Ratios are interpolated in log space between the tested scales."""
    curves: dict[str, list[tuple[float, float]]] = {}
    for scale in scales:
        report = validate(
            bots_path, profiles_path, fmt, tables_per_style=8, hands=1200,
            workers=workers, seed=11, spread=scale,
        )  # fmt: skip
        for name, r in report.items():
            curves.setdefault(name, []).append((scale, r["median_spread_ratio"]))
            if log:
                log(scale, name, r)
    chosen = {}
    for name, pts in curves.items():
        s = np.log([p[0] for p in pts])
        ratio = np.log([max(p[1], 1e-3) for p in pts])
        order = np.argsort(ratio)
        chosen[name] = float(np.exp(np.interp(0.0, ratio[order], s[order])))
    return chosen
