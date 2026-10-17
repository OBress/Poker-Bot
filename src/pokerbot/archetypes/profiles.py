"""Style profiles: the versioned output of style discovery, consumed by opponents and classifiers.

A profile file holds, per table format, the feature transform and K styles. Each style has its
mixture parameters (share, mean, covariance in standardized logit space), pooled and median stat
rates of its members, per-stat Beta parameters (for the Beta-Binomial classifier, MATH.md §4.2),
and a suggested name with a plain-language description.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from pokerbot.archetypes.features import FeatureTransform


@dataclass
class Style:
    index: int
    name: str
    description: str
    share: float
    members: int
    mean: np.ndarray
    cov: np.ndarray
    pooled_rate: dict[str, float]
    median_rate: dict[str, float]
    beta: dict[str, tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "description": self.description,
            "share": self.share,
            "members": self.members,
            "mean": self.mean.tolist(),
            "cov": self.cov.tolist(),
            "pooled_rate": self.pooled_rate,
            "median_rate": self.median_rate,
            "beta": {k: list(v) for k, v in self.beta.items()},
        }


def _weighted_beta(rates: np.ndarray, w: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Weighted method of moments, removing binomial noise from the variance (as in features)."""
    w = w / w.sum()
    m = float(np.sum(w * rates))
    var = float(np.sum(w * (rates - m) ** 2)) - float(np.sum(w * m * (1 - m) / n))
    var = max(var, 1e-4)
    common = max(m * (1 - m) / var - 1, 2.0)
    return m * common, (1 - m) * common


def suggest_name(r: dict[str, float]) -> tuple[str, str]:
    """Heuristic label from pooled rates. A starting point; names are reviewed by a human."""
    vpip, pfr = r["vpip"], r["pfr"]
    ratio = pfr / max(vpip, 1e-9)
    agg = (r["agg_flop"] + r["agg_turn"] + r["agg_river"]) / 3
    looseness = "loose" if vpip >= 0.32 else "tight" if vpip <= 0.20 else "medium"
    style = "aggressive" if ratio >= 0.65 else "passive" if ratio <= 0.4 else "balanced"
    early = f"opens {r['rfi_EP']:.0%} early vs " if "rfi_EP" in r else "opens "
    desc = (
        f"{looseness}, {style} preflop (VPIP {vpip:.0%}, PFR {pfr:.0%}); "
        f"{early}{r['rfi_BTN']:.0%} button; limps {r['limp']:.0%}; "
        f"3-bets {r['3bet']:.0%}; postflop aggression {agg:.0%}; "
        f"folds to c-bet {r['fold_to_cbet_flop']:.0%}; WTSD {r['wtsd']:.0%}"
    )
    if vpip <= 0.15:
        name = "Nit"
    elif vpip >= 0.40 and ratio <= 0.35:
        name = "Calling station" if r["fold_to_cbet_flop"] <= 0.40 else "Loose-passive"
    elif vpip >= 0.35 and ratio >= 0.6:
        name = "Maniac" if agg >= 0.45 else "LAG"
    elif ratio >= 0.65:
        name = "LAG" if vpip >= 0.27 else "TAG"
    elif ratio <= 0.45:
        name = "Weak-tight" if vpip <= 0.25 else "Loose-passive"
    else:
        name = "Fit-or-fold" if r["fold_to_cbet_flop"] >= 0.55 else "Semi-loose reg"
    return name, desc


def build_styles(
    model: GaussianMixture,
    z: np.ndarray,
    n: np.ndarray,
    x: np.ndarray,
    features: tuple[str, ...],
) -> list[Style]:
    q = model.predict_proba(z)
    labels = q.argmax(axis=1)
    styles = []
    for k in range(model.n_components):
        members = labels == k
        w = q[:, k]
        pooled = (w[:, None] * x).sum(0) / np.maximum((w[:, None] * n).sum(0), 1e-9)
        rates = np.where(n > 0, x / np.maximum(n, 1), np.nan)
        median = np.nanmedian(rates[members], axis=0) if members.any() else pooled
        beta = {}
        for j, f in enumerate(features):
            ok = n[:, j] >= 20
            beta[f] = _weighted_beta(rates[ok, j], w[ok] + 1e-12, n[ok, j])
        pooled_d = {f: float(pooled[j]) for j, f in enumerate(features)}
        name, desc = suggest_name(pooled_d)
        styles.append(
            Style(
                index=k,
                name=name,
                description=desc,
                share=float(model.weights_[k]),
                members=int(members.sum()),
                mean=model.means_[k],
                cov=model.covariances_[k],
                pooled_rate=pooled_d,
                median_rate={f: float(median[j]) for j, f in enumerate(features)},
                beta=beta,
            )
        )
    styles.sort(key=lambda s: -s.pooled_rate["vpip"])  # loosest first, stable ordering
    for i, s in enumerate(styles):
        s.index = i
    return styles


def write_profiles(path: Path, version: str, run_id: str, formats: dict[str, dict]) -> None:
    """formats: {fmt: {"transform": FeatureTransform, "styles": [Style], "selection": [...],
    "players": int}}"""
    doc = {
        "version": version,
        "run_id": run_id,
        "source": "HandHQ 2009 (data/SOURCES.md)",
        "formats": {
            fmt: {
                "players": f["players"],
                "transform": f["transform"].to_dict(),
                "selection": f["selection"],
                "styles": [s.to_dict() for s in f["styles"]],
            }
            for fmt, f in formats.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_profiles(path: Path) -> dict:
    """Load a profile file, applying reviewed names from ``<stem>.names.json`` if present (the
    generated ``name`` is kept as ``suggested_name``)."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    names_path = path.with_name(path.stem + ".names.json")
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}
    for fmt, f in doc["formats"].items():
        for s in f["styles"]:
            s["suggested_name"] = s["name"]
            if str(s["index"]) in names.get(fmt, {}):
                s["name"], s["description"] = names[fmt][str(s["index"])]
    for f in doc["formats"].values():
        f["transform"] = FeatureTransform.from_dict(f["transform"])
        for s in f["styles"]:
            s["mean"] = np.array(s["mean"])
            s["cov"] = np.array(s["cov"])
    return doc
