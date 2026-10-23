"""Exploitable opponent styles for the small games (Track A).

There is no real-player data for Kuhn or Leduc, so the styles are defined by hand to mirror the
Hold'em archetypes: a nit, a calling station, a maniac, a fit-or-fold player and a bluffer. A style
is nine probabilities indexed by hand strength s ∈ {weak, medium, strong}:

  bet[s]    bet when not facing a bet
  call[s]   continue (call or raise) when facing a bet
  raise[s]  raise, given it continues and a raise is allowed

Strength: before the public card J/Q/K = weak/medium/strong. After it, a pair is strong, a card
above the public card medium, otherwise weak. Kuhn has one round, so only the first rule applies.

Per seat, parameters are drawn around the style's center in logit space (``sample_params``), so an
opponent of a style is a distribution of players, as in Track C.
"""

from __future__ import annotations

import numpy as np

from pokerbot.baselines.cfr import Strategy, infosets
from pokerbot.games.limit import CALL, FOLD, RAISE, LimitGame

WEAK, MEDIUM, STRONG = 0, 1, 2

# style -> (bet, call, raise), each (weak, medium, strong)
STYLES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "nit": ((0.02, 0.10, 0.80), (0.05, 0.40, 1.00), (0.00, 0.10, 0.70)),
    "calling station": ((0.05, 0.15, 0.40), (0.80, 0.95, 1.00), (0.00, 0.05, 0.20)),
    "maniac": ((0.70, 0.80, 0.95), (0.60, 0.80, 1.00), (0.50, 0.60, 0.80)),
    "fit-or-fold": ((0.05, 0.20, 0.50), (0.10, 0.60, 0.95), (0.00, 0.02, 0.20)),
    "bluffer": ((0.60, 0.30, 0.90), (0.20, 0.50, 1.00), (0.30, 0.20, 0.70)),
}
SPREAD = 0.4  # per-seat sd of each probability in logit units


def strength(game: LimitGame, card: int, public: int | None) -> int:
    if public is None:
        return min(STRONG, card * 3 // game.ranks)
    if card == public:
        return STRONG
    return MEDIUM if card > public else WEAK


def style_strategy(game: LimitGame, params: np.ndarray) -> Strategy:
    """Tabulate a style (array shape (3, 3): bet, call, raise × strength) as a strategy."""
    bet, call, raise_ = params
    out: Strategy = {}
    for info, acts in infosets(game).items():
        card, pub, _ = info.split("|")
        s = strength(game, int(card), None if pub == "-" else int(pub))
        facing = FOLD in acts
        probs = {}
        if facing:
            go = call[s]
            r = raise_[s] if RAISE in acts else 0.0
            probs = {FOLD: 1 - go, CALL: go * (1 - r), RAISE: go * r}
        else:
            b = bet[s] if RAISE in acts else 0.0
            probs = {CALL: 1 - b, RAISE: b}
        out[info] = np.array([probs[a] for a in acts])
    return out


def center(style: str) -> np.ndarray:
    return np.array(STYLES[style], dtype=float)


def sample_params(style: str, rng: np.random.Generator, spread: float = SPREAD) -> np.ndarray:
    """One player of a style: each probability perturbed in logit space."""
    p = np.clip(center(style), 1e-3, 1 - 1e-3)
    z = np.log(p / (1 - p)) + spread * rng.standard_normal(p.shape)
    return 1 / (1 + np.exp(-z))
