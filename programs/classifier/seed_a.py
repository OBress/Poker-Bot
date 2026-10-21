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
    "weight": param(0.5, 0.05, 1.0),  # per-stat weight w_j (corrects for correlated stats)
    "tau": param(1.0, 0.3, 4.0),  # softmax temperature
    "showdown": param(1.0, 0.0, 2.0),  # weight on shown-hand evidence
}
# END-EVOLVE-BLOCK

# EVOLVE-BLOCK helpers
def _log_beta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lg = np.vectorize(math.lgamma)
    return lg(a) + lg(b) - lg(a + b)


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
        # Seed A (MATH.md §4.2): Beta-Binomial naive Bayes over the profile's stats, plus the
        # showdown term (§4.5): sum over shown hands of log P(bucket | line, style).
        prof = self.profiles
        n = np.array([feats.n(f) for f in prof.features], dtype=float)
        x = np.array([feats.x(f) for f in prof.features], dtype=float)
        a, b = prof.beta_a, prof.beta_b
        ll = (_log_beta(x + a, n - x + b) - _log_beta(a, b)).sum(axis=1) * self.p["weight"]
        sd = (np.log(prof.showdown) * feats.shown).sum(axis=(1, 2)) * self.p["showdown"]
        return _softmax((np.log(prof.shares) + ll + sd) / self.p["tau"])
        # END-EVOLVE-BLOCK


_UNUSED = (math, BUCKETS, LINES, HandEnd, HandStart, PlayerAction, Showdown, param)
