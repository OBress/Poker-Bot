"""Simulate tables of bots and measure their stats with the same code used on real hands."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from pokerbot.engine.table import Policy, play_hand
from pokerbot.stats.hand_stats import hand_counts

BIG_BLIND = 100  # chips; fine-grained enough for realistic bet sizes
SMALL_BLIND = 50


def stack_of(policy: Policy) -> int:
    return int(round(getattr(policy, "buyin_bb", 100.0) * BIG_BLIND))


def simulate_stats(
    seats: Sequence[Policy],
    labels: Sequence[str],
    hands: int,
    seed: int,
) -> dict[str, dict[str, list[int]]]:
    """Play ``hands`` cash-game hands (stacks reset to each bot's buy-in, button rotates) and
    return summed stat counts per label: {label: {stat: [n, x]}}. Seats sharing a label pool."""
    n = len(seats)
    deck_seeds = np.random.default_rng(seed).integers(0, 2**63 - 1, size=hands)
    stacks = [stack_of(p) for p in seats]
    totals: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for h in range(hands):
        result = play_hand(
            seats, stacks, button=h % n, deck_seed=int(deck_seeds[h]),
            small_blind=SMALL_BLIND, big_blind=BIG_BLIND, hand_id=h,
        )  # fmt: skip
        for seat, counts in hand_counts(result.events).items():
            t = totals[labels[seat]]
            for stat, (cn, cx) in counts.items():
                t[stat][0] += cn
                t[stat][1] += cx
    return {k: dict(v) for k, v in totals.items()}


def rates(counts: dict[str, list[int]]) -> dict[str, float]:
    return {k: x / n for k, (n, x) in counts.items() if n > 0}
