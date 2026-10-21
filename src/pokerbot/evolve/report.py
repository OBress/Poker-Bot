"""Failure reports for the classifier prompt (EVOLUTION.md §5): where the parent goes wrong."""

from __future__ import annotations

import numpy as np

from pokerbot.evolve import scoring
from pokerbot.evolve.api import StyleProfiles


def classifier_report(posteriors, labels, profiles: StyleProfiles) -> str:
    names = profiles.names
    k = profiles.k
    lines = []
    for ci, t in ((1, 25), (3, 100)):
        conf = scoring.confusion(posteriors, labels, ci, k)
        acc = np.diag(conf)
        off = conf - np.diag(np.diag(conf))
        pairs = sorted(((off[i, j], i, j) for i in range(k) for j in range(k) if i != j),
                       reverse=True)[:3]  # fmt: skip
        lines.append(f"After {t} hands, mean probability on the true style, per style: "
                     + ", ".join(f"{names[i]} {acc[i]:.2f}" for i in range(k)))  # fmt: skip
        lines.append(f"After {t} hands, most mass on a wrong style: " + "; ".join(
            f"true {names[i]} -> {names[j]} {m:.2f}" for m, i, j in pairs))  # fmt: skip
    shares = np.bincount(np.concatenate([np.asarray(y) for y in labels]), minlength=k)
    lines.append("Seats per true style in the validation set: "
                 + ", ".join(f"{names[i]} {shares[i]}" for i in range(k)))  # fmt: skip
    return "\n".join(lines)
