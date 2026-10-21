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
PARAMS = {}
# END-EVOLVE-BLOCK

# EVOLVE-BLOCK helpers
# END-EVOLVE-BLOCK


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
        return np.array(self.profiles.shares)
        # END-EVOLVE-BLOCK


_UNUSED = (math, BUCKETS, LINES, HandEnd, HandStart, PlayerAction, Showdown, param)
