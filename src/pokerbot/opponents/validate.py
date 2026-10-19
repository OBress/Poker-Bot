"""Phase 3 exit check: do sampled bots look like the real players of their style?

For each style, sample many independent bots (spread = 1), seat them at mixed tables of the
format's size, measure each bot's stats over a hand count drawn from the real profiled players'
hand counts (``calibrate.draw_hands``), and compare per stat in the style mixture's own feature
space: each bot's counts go through the profile's ``FeatureTransform`` (Beta-prior shrinkage,
logit, standardization), exactly as a real player's do, so both sides carry the same shrinkage
and similar sampling noise.
  center:  (mean bot feature − style mean μ_kj) / sqrt(Σ_kjj)   -> |z| should be < 1
  spread:  std of bot features / sqrt(Σ_kjj)                    -> ratio should be ~0.5–2
Σ_k is the spread of real members of the style (MATH.md §3.2), so "within the real style's
spread" is literally |center z| < 1.

``legacy_center_z`` is the bots-v2 yardstick, kept for comparison: mean of per-bot logits of
(x + 0.5) / (n + 1) against the logit of the style's pooled rate. It mixes a pooled center with
unshrunk per-bot rates, which biases rare stats toward 0.5 (a bot sees ~10 turn c-bets in 1,500
hands), so it overstates misses on low-count stats.

The same simulation also yields each style's shown-hand strength by betting line, compared with
real showdowns when a real table file (``bots showdown-real``) is given.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pokerbot.archetypes.profiles import load_profiles
from pokerbot.opponents.calibrate import draw_hands, logit
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.showdown import RangeTable, compare, load_real
from pokerbot.opponents.simulate import simulate_stats

SEATS = {"hu": 2, "sh": 6, "fr": 9}  # players dealt per table format (sh: 3-6, fr: 7-10)
HANDS = {"hu": 1000, "sh": 1200, "fr": 1200}  # ~median hands of a real member (styles-v1 run)
HANDS_CAP = 4000  # hands simulated per table at most; longer real samples are scaled counts
BOTS_PER_STYLE = 72  # seats per style's tables (focal + random): 12 six-max tables


def _table(args) -> tuple[list[tuple[int, dict]], dict]:
    bots_path, fmt, focus, seats, hands, hands_q, seed, spread, showdowns = args
    rng = np.random.default_rng(seed)
    hands, scale = (hands, 1.0) if hands_q is None else draw_hands(hands_q, rng, hands)
    sampler = StyleSampler(bots_path, fmt, spread=spread)
    draws = [sampler.draw(rng, style=focus)] + sampler.table(rng, seats - 1)
    labels = [f"s{i}" for i in range(seats)]
    table = RangeTable(len(sampler.styles)) if showdowns else None
    styles = [d.style for d in draws]
    on_hand = (lambda events: table.add_events(events, styles)) if table else None
    counts = simulate_stats(
        [d.bot for d in draws], labels, hands, int(rng.integers(2**31)), on_hand=on_hand
    )
    per_bot = [(d.style, counts[labels[i]], scale) for i, d in enumerate(draws)]
    return per_bot, table.to_dict() if table else {}


def validate(
    bots_path: Path,
    profiles_path: Path,
    fmt: str = "sh",
    tables_per_style: int | None = None,
    seats: int | None = None,
    hands: int | None = None,
    workers: int = 8,
    seed: int = 7,
    spread: float = 1.0,
    real_showdowns: Path | None = None,
    hands_q: np.ndarray | None = None,
) -> dict:
    """``hands_q``: percentiles of real players' hand counts (``membership.hand_quantiles``);
    each table's hand count is drawn from it with ``hands`` (default ``HANDS_CAP``) as the cap.
    Without it every table plays ``hands`` (default: the format's median, ``HANDS``)."""
    prof = load_profiles(profiles_path)["formats"][fmt]
    transform = prof["transform"]
    features = transform.features
    k = len(prof["styles"])
    seats = seats or SEATS[fmt]
    hands = hands or (HANDS[fmt] if hands_q is None else HANDS_CAP)
    tables_per_style = tables_per_style or max(1, round(BOTS_PER_STYLE / seats))
    real = load_real(real_showdowns, fmt) if real_showdowns else None
    jobs = [
        (
            bots_path,
            fmt,
            s,
            seats,
            hands,
            hands_q,
            seed * 100_000 + s * 1000 + t,
            spread,
            real is not None,
        )  # fmt: skip
        for s in range(k)
        for t in range(tables_per_style)
    ]
    per_style: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {s: [] for s in range(k)}
    sim_table = RangeTable(k)
    with ProcessPoolExecutor(workers) as pool:
        for per_bot, table in pool.map(_table, jobs):
            for style, counts, scale in per_bot:
                n = scale * np.array([counts.get(f, (0, 0))[0] for f in features], dtype=float)
                x = scale * np.array([counts.get(f, (0, 0))[1] for f in features], dtype=float)
                per_style[style].append((n, x))
            if table:
                sim_table.merge(RangeTable.from_dict(table))
    report: dict = {
        "format": fmt, "seats": seats, "tables_per_style": tables_per_style,
        "hands": "real distribution" if hands_q is not None else hands,
        "hands_cap": hands if hands_q is not None else None, "styles": {},
    }  # fmt: skip
    for s in range(k):
        st = prof["styles"][s]
        n = np.array([b[0] for b in per_style[s]])
        x = np.array([b[1] for b in per_style[s]])
        z, _ = transform.transform(n, x)
        sd = np.sqrt(np.diag(st["cov"]))
        center = (z.mean(axis=0) - np.array(st["mean"])) / sd
        ratio = z.std(axis=0) / sd
        sigma = sd * transform.std  # logit units
        target = logit(np.array([st["pooled_rate"][f] for f in features]))
        legacy = (logit((x + 0.5) / (n + 1.0)).mean(axis=0) - target) / sigma
        report["styles"][st["name"]] = {
            "bots": len(z),
            "stats": {
                f: {
                    "center_z": float(center[j]),
                    "spread_ratio": float(ratio[j]),
                    "legacy_center_z": float(legacy[j]),
                }
                for j, f in enumerate(features)
            },
            "max_abs_center_z": float(np.abs(center).max()),
            "median_spread_ratio": float(np.median(ratio)),
            "within_spread": int((np.abs(center) < 1).sum()),
            "within_2": int((np.abs(center) < 2).sum()),
            "legacy_within_spread": int((np.abs(legacy) < 1).sum()),
            "features": len(features),
        }
    pairs = k * len(features)
    report["summary"] = {
        "pairs": pairs,
        "within_spread": sum(r["within_spread"] for r in report["styles"].values()),
        "within_2": sum(r["within_2"] for r in report["styles"].values()),
        "legacy_within_spread": sum(r["legacy_within_spread"] for r in report["styles"].values()),
    }
    if real is not None:
        real_table, real_names = real
        names = [st["name"] for st in prof["styles"]]
        if len(real_names) != len(names):
            raise ValueError("real showdown tables don't match the profile's styles")
        report["showdowns"] = compare(real_table, sim_table, names)
    return report


def tune_scales(
    bots_path: Path,
    profiles_path: Path,
    fmt: str = "sh",
    scales: tuple[float, ...] = (0.2, 0.4, 0.6, 0.85, 1.15, 1.5),
    workers: int = 8,
    log=None,
    hands_q: np.ndarray | None = None,
) -> dict[str, float]:
    """Per style, the covariance scale at which sampled bots' median spread ratio is ~1.

    The linear-response estimate of the parameter covariance overshoots (stats respond
    nonlinearly, and insensitive parameter directions get huge variance), so the scale is measured
    by simulation. Observed variance is measurement noise plus parameter spread, so the ratios
    are fitted as ratio² = a + b·scale² (least squares over the tested scales) and the scale with
    ratio 1 is solved from the fit; a floor a >= 1 (noise alone fills the real spread) gives the
    smallest tested scale."""
    curves: dict[str, list[tuple[float, float]]] = {}
    for scale in scales:
        report = validate(
            bots_path, profiles_path, fmt, tables_per_style=None,
            workers=workers, seed=11, spread=scale, hands_q=hands_q,
        )  # fmt: skip
        for name, r in report["styles"].items():
            curves.setdefault(name, []).append((scale, r["median_spread_ratio"]))
            if log:
                log(scale, name, r)
    return {name: solve_scale(pts, min(scales), max(scales)) for name, pts in curves.items()}


def solve_scale(points: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Scale with spread ratio 1 under ratio² = a + b·scale², clipped to [lo, hi]."""
    s2 = np.array([p[0] ** 2 for p in points])
    r2 = np.array([p[1] ** 2 for p in points])
    b, a = np.polyfit(s2, r2, 1)
    if b <= 0 or a >= 1:
        return lo
    return float(np.clip(np.sqrt((1 - a) / b), lo, hi))
