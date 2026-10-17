"""Hand strength for rule-based bots.

- Preflop: each of the 169 starting-hand classes has an all-in equity vs one random hand
  (precomputed, `preflop_equity.json`). ``preflop_percentile`` maps a hand to the fraction of all
  1,326 combos that are stronger, so "plays the top 20%" means ``percentile < 0.20``.
- Postflop: Monte Carlo equity vs one random hand with a fixed number of samples. Sampling is
  seeded from the cards themselves, so the result is a pure function (reproducible and cacheable).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from phevaluator import evaluate_cards

from pokerbot.engine.cards import RANKS, rank_of, suit_of

_TABLE = Path(__file__).with_name("preflop_equity.json")
POSTFLOP_SAMPLES = 96  # standard error ~0.05 equity, ~350 us per call


def hand_class(hole: tuple[int, int]) -> str:
    """'AA', 'AKs', 'T9o', ... (higher rank first)."""
    a, b = sorted(hole, key=rank_of, reverse=True)
    ra, rb = RANKS[rank_of(a)], RANKS[rank_of(b)]
    if ra == rb:
        return ra + rb
    return ra + rb + ("s" if suit_of(a) == suit_of(b) else "o")


def all_classes() -> list[str]:
    out = []
    for i in range(12, -1, -1):
        for j in range(i, -1, -1):
            if i == j:
                out.append(RANKS[i] * 2)
            else:
                out += [RANKS[i] + RANKS[j] + "s", RANKS[i] + RANKS[j] + "o"]
    return out


def combos(cls: str) -> int:
    return 6 if len(cls) == 2 else 4 if cls[2] == "s" else 12


def compute_preflop_equity(samples: int = 20_000, seed: int = 0) -> dict[str, float]:
    """All-in equity of each class vs a uniformly random hand (ties count half)."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in all_classes():
        r1, r2 = RANKS.index(cls[0]), RANKS.index(cls[1])
        if len(cls) == 2:
            hole = (r1 * 4, r1 * 4 + 1)
        elif cls[2] == "s":
            hole = (r1 * 4, r2 * 4)
        else:
            hole = (r1 * 4, r2 * 4 + 1)
        deck = np.array([c for c in range(52) if c not in hole])
        score = 0.0
        for d in _draws(rng, deck, samples, 7):
            me = evaluate_cards(*hole, *d[2:])
            opp = evaluate_cards(*d[:7])
            score += 1.0 if me < opp else 0.5 if me == opp else 0.0
        out[cls] = score / samples
    return out


def _draws(rng: np.random.Generator, deck: np.ndarray, samples: int, k: int) -> list[list[int]]:
    """``samples`` draws of ``k`` distinct cards from ``deck`` (vectorized), as Python ints."""
    idx = rng.random((samples, len(deck))).argpartition(k, axis=1)[:, :k]
    return deck[idx].tolist()


@lru_cache(maxsize=1)
def _percentiles() -> dict[str, float]:
    eq = json.loads(_TABLE.read_text(encoding="utf-8"))
    order = sorted(eq, key=lambda c: -eq[c])
    total = 1326
    out = {}
    stronger = 0
    for cls in order:
        out[cls] = (stronger + combos(cls) / 2) / total  # midpoint of the class's combo block
        stronger += combos(cls)
    return out


@lru_cache(maxsize=1)
def preflop_equities() -> dict[str, float]:
    return json.loads(_TABLE.read_text(encoding="utf-8"))


def preflop_percentile(hole: tuple[int, int]) -> float:
    """0 = best hand (AA), 1 = worst. Fraction of combos ranked above this hand."""
    return _percentiles()[hand_class(hole)]


@lru_cache(maxsize=200_000)
def postflop_equity(hole: tuple[int, int], board: tuple[int, ...]) -> float:
    """Equity vs one random hand, completing the board at random. Deterministic per input."""
    seed = (hash((tuple(sorted(hole)), tuple(sorted(board)))) & 0xFFFFFFFF) or 1
    rng = np.random.default_rng(seed)
    used = set(hole) | set(board)
    deck = np.array([c for c in range(52) if c not in used])
    need = 5 - len(board)
    score = 0.0
    for d in _draws(rng, deck, POSTFLOP_SAMPLES, 2 + need):
        full = (*board, *d[2:])
        me = evaluate_cards(*hole, *full)
        opp = evaluate_cards(d[0], d[1], *full)
        score += 1.0 if me < opp else 0.5 if me == opp else 0.0
    return score / POSTFLOP_SAMPLES
