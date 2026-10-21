"""Classifier candidate. Frozen template classifier_v1 (EVOLUTION.md §2).

Only the code between "# EVOLVE-BLOCK <name>" and "# END-EVOLVE-BLOCK" may change. The harness
creates one Classifier per session, feeds it every public event in order (observe) and asks for
a seat's style probabilities with posterior(seat, feats), which must return K non-negative finite
numbers (the harness normalizes them). PARAMS values are tuned within [low, high].
"""

import math

import numpy as np

from pokerbot.evolve.api import (
    BUCKETS,
    LINES,
    Event,
    HandEnd,
    HandStart,
    PlayerAction,
    SeatFeatures,
    Showdown,
    StyleProfiles,
    param,
)

# EVOLVE-BLOCK params
PARAMS = {
    "tau": param(1.0, 0.3, 4.0),  # softmax temperature
    "noise": param(1.0, 0.3, 3.0),  # multiplier on each stat's measurement variance
    "showdown": param(1.0, 0.0, 2.0),  # weight on shown-hand evidence
}
# END-EVOLVE-BLOCK

# EVOLVE-BLOCK helpers
def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()# END-EVOLVE-BLOCK


class Classifier:
    def __init__(self, profiles: StyleProfiles, params: dict[str, float]) -> None:
        self.profiles = profiles
        self.p = params
        self.reset()

    def reset(self) -> None:
        # EVOLVE-BLOCK state
        pass
        # END-EVOLVE-BLOCK

    def observe(self, event: Event) -> None:
        # EVOLVE-BLOCK observe
        pass
        # END-EVOLVE-BLOCK

    def posterior(self, seat: int, feats: SeatFeatures) -> np.ndarray:
        # EVOLVE-BLOCK posterior
        # Seed B (MATH.md §4.3): correlated Gaussian in standardized logit space. Observed stats
        # O (n > 0) give z_O with noise S_O; L_k = N(z_O | mu_kO, Sigma_kOO + S_O). Plus §4.5.
        prof = self.profiles
        n = np.array([feats.n(f) for f in prof.features], dtype=float)
        x = np.array([feats.x(f) for f in prof.features], dtype=float)
        logits = np.log(prof.shares).copy()
        seen = n > 0
        if seen.any():
            z, s2 = prof.transform(n, x)
            z, s2 = z[seen], s2[seen] * self.p["noise"]
            for k in range(prof.k):
                cov = prof.covs[k][np.ix_(seen, seen)] + np.diag(s2)
                d = z - prof.means[k][seen]
                sign, logdet = np.linalg.slogdet(cov)
                logits[k] += -0.5 * (logdet + d @ np.linalg.solve(cov, d))
        logits += (np.log(prof.showdown) * feats.shown).sum(axis=(1, 2)) * self.p["showdown"]
        return _softmax(logits / self.p["tau"])
        # END-EVOLVE-BLOCK


_UNUSED = (math, BUCKETS, LINES, HandEnd, HandStart, PlayerAction, Showdown, param)
