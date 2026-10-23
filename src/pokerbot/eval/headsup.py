"""Head-to-head duplicate heads-up matches between two named policies (blueprint sanity checks).

Every deck is played twice with the seats swapped, so both players get both hands of every deal;
all-in runouts are replaced by their expectation. The result is player A's bb/100 with a 95%
bootstrap CI over blocks of hands (a block = both orders of ``block`` consecutive deals).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pokerbot.engine.table import play_hand
from pokerbot.eval.allin import allin_adjusted
from pokerbot.eval.duplicate import BIG_BLIND, SMALL_BLIND, bootstrap_ci, make_policy

STACK_BB = 100


def _block(args) -> float:
    a_spec, b_spec, seed, deals = args
    rng = np.random.default_rng(seed)
    decks = rng.integers(0, 2**63 - 1, size=deals)
    total = 0.0
    for swap in (False, True):
        a, b = make_policy(a_spec, seed), make_policy(b_spec, seed + 1)
        seats = [b, a] if swap else [a, b]
        hero = 1 if swap else 0
        for h, d in enumerate(decks):
            r = play_hand(seats, [STACK_BB * BIG_BLIND] * 2, button=h % 2, deck_seed=int(d),
                          small_blind=SMALL_BLIND, big_blind=BIG_BLIND, hand_id=h)  # fmt: skip
            adj = allin_adjusted(r.events, seed=int(d) & 0xFFFFFFFF)
            total += (adj[hero] if adj is not None else r.payoffs[hero]) / BIG_BLIND
    return total


def heads_up(
    a: str, b: str, blocks: int, deals_per_block: int, seed: int = 0, workers: int = 8
) -> dict:
    jobs = [(a, b, seed * 1_000_003 + i, deals_per_block) for i in range(blocks)]
    with ProcessPoolExecutor(workers) as pool:
        totals = np.array(list(pool.map(_block, jobs)))
    mean, lo, hi = bootstrap_ci(totals, 2 * deals_per_block)
    return {"a": a, "b": b, "hands": 2 * blocks * deals_per_block, "bb100": mean,
            "ci95": [lo, hi]}  # fmt: skip
