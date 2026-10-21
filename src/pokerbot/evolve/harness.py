"""The frozen harness around classifier programs (EVOLUTION.md §1, §3, §4).

- ``redact``: drops private events (other players' hole cards); what's left is public.
- ``FeatureTracker``: the per-seat counts behind ``SeatFeatures``, updated once per finished hand
  with the same stat code used on real data (``stats.hand_counts``) and the shown-hand table
  (``stats.showdown``).
- ``run_classifier``: feeds a program the public events of each session in order and asks for
  every seat's posterior at the checkpoints; outputs are validated (finite, non-negative, right
  shape), clipped and renormalized; invalid outputs are replaced by the uniform distribution and
  counted. Per-call time is measured. True labels never reach the program.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from pokerbot.engine.events import Event, HandStart, HoleCards
from pokerbot.evolve.api import BUCKETS, LINES, SeatFeatures, StyleProfiles
from pokerbot.stats.hand_stats import hand_counts
from pokerbot.stats.showdown import shown_hands

CHECKPOINTS = (10, 25, 50, 100, 200)


def redact(events: Sequence[Event]) -> tuple[Event, ...]:
    return tuple(e for e in events if not isinstance(e, HoleCards))


class FeatureTracker:
    def __init__(self) -> None:
        self.counts: dict[int, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        self.shown: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((len(LINES), BUCKETS)))
        self.hands: dict[int, int] = defaultdict(int)

    def add_hand(self, events: Sequence[Event]) -> None:
        start = next(e for e in events if isinstance(e, HandStart))
        for seat in range(len(start.stacks)):
            self.hands[seat] += 1
        for seat, c in hand_counts(events).items():
            acc = self.counts[seat]
            for stat, (n, x) in c.items():
                acc[stat][0] += n
                acc[stat][1] += x
        for sh in shown_hands(events):
            self.shown[sh.seat][LINES.index(sh.line), sh.bucket] += 1

    def features(self, seat: int) -> SeatFeatures:
        c = {k: (v[0], v[1]) for k, v in self.counts[seat].items()}
        return SeatFeatures.make(self.hands[seat], c, self.shown[seat])

    def snapshot(self, seat: int) -> tuple[int, dict[str, tuple[int, int]], np.ndarray]:
        f = self.features(seat)
        return f.hands, dict(f.counts), np.array(f.shown)


def validate_output(p, k: int) -> tuple[np.ndarray, bool]:
    """(normalized posterior, valid?) — invalid outputs become uniform."""
    try:
        a = np.asarray(p, dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001 - any garbage output is just invalid
        return np.full(k, 1.0 / k), False
    if a.shape != (k,) or not np.all(np.isfinite(a)) or np.any(a < 0) or a.sum() <= 0:
        return np.full(k, 1.0 / k), False
    return a / a.sum(), True


def run_classifier(make, profiles: StyleProfiles, sessions: Sequence[dict],
                   checkpoints: Sequence[int] = CHECKPOINTS) -> dict:  # fmt: skip
    """``make()`` builds a fresh classifier per session. Each session dict holds ``hands`` (tuple
    of redacted event tuples), ``feats`` ({checkpoint: [(hands, counts, shown) per scored seat]})
    and ``seats`` (the scored seat indices). Returns posteriors shaped
    (sessions, scored seats, checkpoints, K), invalid-output count and mean µs per call."""
    k = profiles.k
    out = []
    invalid = calls = 0
    seconds = 0.0
    for s in sessions:
        clf = make()
        seats = s["seats"]
        post = np.zeros((len(seats), len(checkpoints), k))
        t_index = {t: i for i, t in enumerate(checkpoints)}
        for h, events in enumerate(s["hands"], 1):
            t0 = time.perf_counter()
            for e in events:
                clf.observe(e)
            seconds += time.perf_counter() - t0
            if h in t_index:
                feats = s["feats"][h]
                for j, seat in enumerate(seats):
                    f = SeatFeatures.make(*feats[j])
                    t0 = time.perf_counter()
                    p = clf.posterior(seat, f)
                    seconds += time.perf_counter() - t0
                    calls += 1
                    post[j, t_index[h]], ok = validate_output(p, k)
                    invalid += not ok
        out.append(post)
    return {"posteriors": out, "invalid": invalid, "calls": calls,
            "us_per_call": 1e6 * seconds / max(calls, 1)}  # fmt: skip
