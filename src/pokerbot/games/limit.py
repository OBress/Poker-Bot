"""Two-player limit poker with a tiny deck: Kuhn and Leduc (Track A, ARCHITECTURE §3).

One game class covers both. Each player antes, gets one private card, and there are one or two
betting rounds; with two rounds a public card is dealt between them. Bets are a fixed size per
round and at most ``raise_cap`` bets/raises are made per round. Player 0 acts first every round.
At showdown a private card that pairs the public card wins, otherwise the higher rank; equal ranks
split the pot.

Cards are ranks 0..ranks-1 (suits only matter through how many copies the deck holds).
A betting history is one string per round from ``c`` (check/call), ``r`` (bet/raise), ``f`` (fold),
joined by ``/``; an information set is ``"<own card>|<public card or ->|<history>"``.

  Kuhn:  ranks J,Q,K once each, ante 1, one round, bet 1, cap 1. Nash value for player 0: −1/18.
  Leduc: ranks J,Q,K twice each, ante 1, two rounds, bets 2 then 4, cap 2 per round.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

FOLD, CALL, RAISE = "f", "c", "r"


@dataclass(frozen=True)
class LimitGame:
    name: str
    ranks: int
    suits: int  # copies of each rank in the deck
    rounds: int
    bet_sizes: tuple[int, ...]  # per round
    raise_cap: int
    ante: int = 1

    # ------------------------------------------------------------------ chance

    @cached_property
    def deals(self) -> list[tuple[tuple[int, int], float]]:
        """Private deals (card of player 0, card of player 1) with their probabilities."""
        n = self.ranks * self.suits
        out = []
        for a in range(self.ranks):
            for b in range(self.ranks):
                same = a == b
                if same and self.suits < 2:
                    continue
                p = self.suits / n * ((self.suits - 1) if same else self.suits) / (n - 1)
                out.append(((a, b), p))
        return out

    def public_cards(self, a: int, b: int) -> list[tuple[int, float]]:
        """Public card ranks and probabilities given both private cards."""
        left = self.ranks * self.suits - 2
        out = []
        for c in range(self.ranks):
            k = self.suits - (a == c) - (b == c)
            if k > 0:
                out.append((c, k / left))
        return out

    # ------------------------------------------------------------------ betting

    @staticmethod
    def round_over(h: str) -> bool:
        return h in ("cc",) or (len(h) >= 2 and h[-1] == CALL and RAISE in h) or FOLD in h

    @staticmethod
    def to_act(h: str) -> int:
        """Player to act within a round (player 0 opens every round)."""
        return len(h) % 2

    def legal(self, h: str) -> str:
        """Legal actions in a round whose history so far is ``h`` (not over)."""
        raises = h.count(RAISE)
        facing = raises > 0 and h[-1] == RAISE
        acts = (FOLD + CALL) if facing else CALL
        if raises < self.raise_cap:
            acts += RAISE
        return acts

    def contributions(self, rounds: list[str]) -> tuple[int, int]:
        """Chips each player has put in, ante included."""
        put = [self.ante, self.ante]
        for r, h in enumerate(rounds):
            level = 0
            street = [0, 0]
            for i, a in enumerate(h):
                p = i % 2
                if a == RAISE:
                    level += self.bet_sizes[r]
                    street[p] = level
                elif a == CALL:
                    street[p] = level
            put[0] += street[0]
            put[1] += street[1]
        return put[0], put[1]

    def payoff(self, rounds: list[str], cards: tuple[int, int], public: int | None) -> float:
        """Net chips won by player 0 at a terminal history."""
        put = self.contributions(rounds)
        last = rounds[-1]
        if FOLD in last:
            folder = (len(last) - 1) % 2
            return -put[0] if folder == 0 else put[1]
        s0, s1 = self.strength(cards[0], public), self.strength(cards[1], public)
        if s0 == s1:
            return 0.0
        return put[1] if s0 > s1 else -put[0]

    def strength(self, card: int, public: int | None) -> int:
        return card + (self.ranks if card == public else 0)

    @staticmethod
    def infoset(card: int, public: int | None, rounds: list[str]) -> str:
        return f"{card}|{'-' if public is None else public}|{'/'.join(rounds)}"


KUHN = LimitGame("kuhn", ranks=3, suits=1, rounds=1, bet_sizes=(1,), raise_cap=1)
LEDUC = LimitGame("leduc", ranks=3, suits=2, rounds=2, bet_sizes=(2, 4), raise_cap=2)
GAMES = {"kuhn": KUHN, "leduc": LEDUC}
