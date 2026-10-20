"""Exact solving and evaluation for the small limit games (Track A).

- ``CFRPlus``: CFR+ (regret matching+, alternating updates, linearly weighted average strategy).
  Its average strategy converges to a Nash equilibrium; this is the Track A baseline.
- ``best_response_value``: value of the best response to a fixed strategy, computed exactly by
  walking the tree once per own card with a weight per opponent card (an information set of the
  responder is its own card + public card + history, so all opponent cards share one decision).
- ``exploitability``: mean of both players' best-response gains; 0 exactly at a Nash equilibrium.
- ``expected_value``: exact value of one strategy profile, used for matchup tables.

A strategy maps an information set (``LimitGame.infoset``) to probabilities over
``game.legal(round history)`` in that order. One dict holds both players' decisions, since the
history determines who acts. Values are chips per hand for the stated player.
"""

from __future__ import annotations

import numpy as np

from pokerbot.games.limit import FOLD, LimitGame

Strategy = dict[str, np.ndarray]


def _terminal(game: LimitGame, rounds: list[str]) -> bool:
    h = rounds[-1]
    return game.round_over(h) and (FOLD in h or len(rounds) == game.rounds)


def infosets(game: LimitGame) -> dict[str, str]:
    """Every information set of the game -> its legal actions."""
    out: dict[str, str] = {}

    def walk(card: int, public: int | None, rounds: list[str], needs_public: bool) -> None:
        h = rounds[-1]
        if game.round_over(h):
            if FOLD in h or len(rounds) == game.rounds:
                return
            for c in range(game.ranks):
                walk(card, c, [*rounds, ""], needs_public)
            return
        out[game.infoset(card, public, rounds)] = game.legal(h)
        for a in game.legal(h):
            walk(card, public, [*rounds[:-1], h + a], needs_public)

    for card in range(game.ranks):
        walk(card, None, [""], game.rounds > 1)
    return out


def uniform(game: LimitGame) -> Strategy:
    return {i: np.full(len(a), 1 / len(a)) for i, a in infosets(game).items()}


class CFRPlus:
    def __init__(self, game: LimitGame) -> None:
        self.game = game
        self.legal = infosets(game)
        self.regret = {i: np.zeros(len(a)) for i, a in self.legal.items()}
        self.avg = {i: np.zeros(len(a)) for i, a in self.legal.items()}
        self.t = 0
        self._sigma: Strategy = {}
        self._touched: set[str] = set()

    def current(self, info: str) -> np.ndarray:
        r = self.regret[info]
        s = r.sum()
        return r / s if s > 0 else np.full(len(r), 1 / len(r))

    def iterate(self, n: int = 1) -> None:
        for _ in range(n):
            self.t += 1
            for p in (0, 1):
                # One strategy per iteration: an information set is visited once per opponent
                # card, so regrets are summed during the walk and clipped (CFR+) afterwards.
                self._sigma = {i: self.current(i) for i in self.legal}
                for cards, prob in self.game.deals:
                    self._walk(p, cards, None, [""], 1.0, prob)
                for info in self._touched:
                    self.regret[info] = np.maximum(self.regret[info], 0.0)
                self._touched.clear()

    def _walk(self, p: int, cards, public, rounds: list[str], pi_p: float, pi_o: float) -> float:
        g = self.game
        h = rounds[-1]
        if g.round_over(h):
            if FOLD in h or len(rounds) == g.rounds:
                u = g.payoff(rounds, cards, public)
                return u if p == 0 else -u
            return sum(
                q * self._walk(p, cards, c, [*rounds, ""], pi_p, pi_o * q)
                for c, q in g.public_cards(*cards)
            )
        actor = g.to_act(h)
        info = g.infoset(cards[actor], public, rounds)
        sigma = self._sigma[info]
        acts = self.legal[info]
        if actor != p:
            return sum(
                sigma[i]
                * self._walk(p, cards, public, [*rounds[:-1], h + a], pi_p, pi_o * sigma[i])
                for i, a in enumerate(acts)
                if sigma[i] > 0
            )
        vals = np.array(
            [
                self._walk(p, cards, public, [*rounds[:-1], h + a], pi_p * sigma[i], pi_o)
                for i, a in enumerate(acts)
            ]
        )
        v = float(sigma @ vals)
        self.regret[info] += pi_o * (vals - v)
        self._touched.add(info)
        self.avg[info] += self.t * pi_p * sigma  # linear averaging
        return v

    def average(self) -> Strategy:
        out = {}
        for info, s in self.avg.items():
            total = s.sum()
            out[info] = s / total if total > 0 else np.full(len(s), 1 / len(s))
        return out


def _value(game: LimitGame, strategy: Strategy, cards, public, rounds, p: int) -> float:
    h = rounds[-1]
    if game.round_over(h):
        if FOLD in h or len(rounds) == game.rounds:
            u = game.payoff(rounds, cards, public)
            return u if p == 0 else -u
        return sum(
            q * _value(game, strategy, cards, c, [*rounds, ""], p)
            for c, q in game.public_cards(*cards)
        )
    sigma = strategy[game.infoset(cards[game.to_act(h)], public, rounds)]
    return sum(
        sigma[i] * _value(game, strategy, cards, public, [*rounds[:-1], h + a], p)
        for i, a in enumerate(game.legal(h))
        if sigma[i] > 0
    )


def expected_value(game: LimitGame, seat0: Strategy, seat1: Strategy) -> float:
    """Chips per hand won by the player in seat 0 (who uses ``seat0``) against ``seat1``."""
    merged = _merge(game, seat0, seat1)
    return sum(prob * _value(game, merged, cards, None, [""], 0) for cards, prob in game.deals)


def matchup(game: LimitGame, hero: Strategy, villain: Strategy) -> float:
    """Hero's chips per hand against villain, averaged over both seats."""
    return 0.5 * (expected_value(game, hero, villain) - expected_value(game, villain, hero))


def _merge(game: LimitGame, seat0: Strategy, seat1: Strategy) -> Strategy:
    """Seat 0's decisions from ``seat0`` and seat 1's from ``seat1``."""
    out = {}
    for info in infosets(game):
        rounds = info.split("|")[2].split("/")
        out[info] = (seat0 if game.to_act(rounds[-1]) == 0 else seat1)[info]
    return out


def best_response_value(game: LimitGame, strategy: Strategy, p: int) -> float:
    """Value for player ``p`` of the best response to ``strategy`` (used for the other seat)."""
    total = 0.0
    for x in range(game.ranks):
        w = {}
        for (c0, c1), prob in game.deals:
            if (c0 if p == 0 else c1) == x:
                w[c1 if p == 0 else c0] = prob
        total += _br(game, strategy, p, x, w, None, [""])
    return total


def best_response(game: LimitGame, strategy: Strategy) -> Strategy:
    """A pure best response for both seats against ``strategy`` (each seat vs the other)."""
    out: Strategy = {}
    for p in (0, 1):
        for x in range(game.ranks):
            w = {}
            for (c0, c1), prob in game.deals:
                if (c0 if p == 0 else c1) == x:
                    w[c1 if p == 0 else c0] = prob
            _br(game, strategy, p, x, w, None, [""], out)
    for info, acts in infosets(game).items():
        out.setdefault(info, np.full(len(acts), 1 / len(acts)))  # unreachable for the responder
    return out


def _br(game, strategy, p, x, w: dict[int, float], public, rounds, record=None) -> float:
    """Sum over opponent cards y of w[y] × value for p, with p best-responding."""
    h = rounds[-1]
    if not w:
        return 0.0
    if game.round_over(h):
        if FOLD in h or len(rounds) == game.rounds:
            total = 0.0
            for y, wy in w.items():
                cards = (x, y) if p == 0 else (y, x)
                u = game.payoff(rounds, cards, public)
                total += wy * (u if p == 0 else -u)
            return total
        total = 0.0
        for c in range(game.ranks):
            nw = {}
            for y, wy in w.items():
                q = dict(game.public_cards(x, y)).get(c, 0.0)
                if q > 0:
                    nw[y] = wy * q
            total += _br(game, strategy, p, x, nw, c, [*rounds, ""], record)
        return total
    acts = game.legal(h)
    if game.to_act(h) == p:
        vals = [_br(game, strategy, p, x, w, public, [*rounds[:-1], h + a], record) for a in acts]
        best = int(np.argmax(vals))
        if record is not None:
            onehot = np.zeros(len(acts))
            onehot[best] = 1.0
            record[game.infoset(x, public, rounds)] = onehot
        return vals[best]
    total = 0.0
    for i, a in enumerate(acts):
        nw = {}
        for y, wy in w.items():
            s = strategy[game.infoset(y, public, rounds)][i]
            if s > 0:
                nw[y] = wy * s
        total += _br(game, strategy, p, x, nw, public, [*rounds[:-1], h + a], record)
    return total


def exploitability(game: LimitGame, strategy: Strategy) -> float:
    """(best-response value for seat 0 + for seat 1) / 2, chips per hand; 0 at Nash."""
    return 0.5 * (best_response_value(game, strategy, 0) + best_response_value(game, strategy, 1))
