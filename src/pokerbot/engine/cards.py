"""Card encoding, decks and hand evaluation.

A card is an int in [0, 52): ``rank * 4 + suit`` with ranks 2..A -> 0..12 and suits
c, d, h, s -> 0..3. This matches phevaluator's ids, so cards go to the evaluator unchanged.
"""

from __future__ import annotations

import numpy as np
from phevaluator import evaluate_cards

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK_SIZE = 52


def card(text: str) -> int:
    """Parse ``"As"`` -> 51."""
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def cards(text: str) -> tuple[int, ...]:
    """Parse ``"AsKd"`` or ``"As Kd"`` into card ids."""
    text = text.replace(" ", "")
    return tuple(card(text[i : i + 2]) for i in range(0, len(text), 2))


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def cards_str(cs: tuple[int, ...] | list[int]) -> str:
    return "".join(card_str(c) for c in cs)


def rank_of(c: int) -> int:
    return c // 4


def suit_of(c: int) -> int:
    return c % 4


def shuffled_deck(seed: int | np.random.Generator) -> list[int]:
    """A deterministic shuffled deck. Same seed -> same deck, on every platform."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.permutation(DECK_SIZE).tolist()


def hand_rank(hole: tuple[int, ...], board: tuple[int, ...]) -> int:
    """Strength of the best 5-card hand. Lower is better (1 = royal flush)."""
    return evaluate_cards(*hole, *board)
