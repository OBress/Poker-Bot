"""How typical a player is of the style they were read as (demo only).

The classifier always names the most likely of its known styles, even for a player unlike all of
them. The fit measure says how far the player sits from that style, under the same Gaussian the
profiles describe (MATH.md §4.3): over the stats seen so far O, with measurement noise S_O,

    d² = (z_O − μ_k,O)ᵀ (Σ_k,OO + S_O)⁻¹ (z_O − μ_k,O)  ~  χ²(|O|)  for a real member of style k.

``p`` is the χ² tail probability: the share of real players of that style who sit at least this
far from its center. Per-stat z-scores (on the diagonal) name the stats that stand out. Noise
shrinks with hands, so early reads fit easily and an unusual player only stops fitting once the
evidence is in. Reads only; the frozen profiles and classifier are untouched.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from pokerbot.evolve.api import StyleProfiles

WELL, LOOSE = 0.05, 1e-4  # p thresholds: fits well / loosely / unlike any style
EXTREME_Z = 2.5  # a stat this many sds from the style stands out
MANIAC_Z = 2.5  # VPIP and preflop raising this far above *every* style
# How each stat reads after "you" when far above / below the style; the third-person form adds
# an "s" to each verb (see ``_third``)
PHRASES = {
    "vpip": ("play far more hands", "play far fewer hands"),
    "pfr": ("raise before the flop far more", "raise before the flop far less"),
    "limp": ("limp far more", "limp far less"),
    "3bet": ("3-bet far more", "3-bet far less"),
    "fold_to_cbet_flop": ("fold to flop bets far more", "fold to flop bets far less"),
    "agg_flop": ("bet and raise on the flop far more", "bet and raise on the flop far less"),
    "wtsd": ("go to showdown far more", "go to showdown far less"),
}
VERBS = {"play", "raise", "limp", "3-bet", "fold", "bet", "go"}


def _third(phrase: str) -> str:
    """'go to showdown far more' -> 'goes to showdown far more'."""
    return " ".join(
        (w + "es" if w == "go" else w + "s") if w in VERBS else w for w in phrase.split()
    )


def _observed(prof: StyleProfiles, counts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardized features, their noise variances, and which stats have been seen."""
    n = np.array([counts.get(f, (0, 0))[0] for f in prof.features], dtype=float)
    x = np.array([counts.get(f, (0, 0))[1] for f in prof.features], dtype=float)
    z, s2 = prof.transform(n, x)
    return z, s2, n > 0


def style_fit(prof: StyleProfiles, counts, style: str) -> dict | None:
    """How well observed counts (``stat -> (n, x)``) fit ``style``; None before any stat is seen."""
    z, s2, seen = _observed(prof, counts)
    if not seen.any():
        return None
    k = prof.names.index(style)
    o = np.flatnonzero(seen)
    diff = z[o] - prof.means[k, o]
    cov = prof.covs[k][np.ix_(o, o)] + np.diag(s2[o])
    d2 = float(diff @ np.linalg.solve(cov, diff))
    p = float(chi2.sf(d2, len(o)))
    zs = diff / np.sqrt(np.diag(cov))
    extremes = sorted(
        (
            {
                "key": prof.features[j],
                "z": float(zj),
                "you": (phrase := PHRASES[prof.features[j]][int(zj < 0)]),
                "they": _third(phrase),
            }
            for j, zj in zip(o, zs, strict=True)
            if prof.features[j] in PHRASES and abs(zj) >= EXTREME_Z
        ),
        key=lambda e: -abs(e["z"]),
    )
    level = "well" if p >= WELL else "loose" if p >= LOOSE else "none"
    return {
        "level": level,
        "label": {
            "well": "fits well",
            "loose": "fits loosely",
            "none": "plays unlike any style",
        }[level],
        "distance": float(np.sqrt(d2)),
        "p": p,
        "stats": len(o),
        "extremes": extremes[:3],
        "alias": "Loose-aggressive / Maniac" if _maniac(prof, z, s2, seen) else None,
    }


def _maniac(prof: StyleProfiles, z: np.ndarray, s2: np.ndarray, seen: np.ndarray) -> bool:
    """VPIP and preflop raising both far above every style (a demo-only label)."""
    for key in ("vpip", "pfr"):
        if key not in prof.features:
            return False
        j = prof.features.index(key)
        if not seen[j]:
            return False
        sd = np.sqrt(prof.covs[:, j, j] + s2[j])
        if ((z[j] - prof.means[:, j]) / sd).min() < MANIAC_Z:
            return False
    return True
