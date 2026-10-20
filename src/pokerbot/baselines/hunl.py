"""Abstract heads-up no-limit hold'em for the Track B blueprint (ARCHITECTURE §3, §4.6).

Game: two players, 100 bb stacks, blinds 1/2 chips (1 bb = 2 chips). Player 0 is the button: it
posts the small blind, acts first preflop and last on later streets.

Bet abstraction: fold ``f``, check/call ``c``, bet to the current bet plus 33%, 75% or 125% of
the pot after calling (``1``, ``2``, ``3``), all-in ``a``. Facing a bet the only raise sizes are
75% and all-in. Sizes that would put the player (nearly) all-in collapse into ``a``. At most
``RAISE_CAP`` sized bets/raises per street (preflop: the big blind, open, 3-bet, 4-bet; later
streets: bet and one raise); all-in stays available after the cap. This keeps the tree at ~21k
public decision nodes (~0.26M information sets) so MCCFR can cover it; the exploiter's menu
(MATH.md §5.2) is the same set of sizes.

Card abstraction: preflop, the 169 hand classes; flop, turn and river, ``BUCKETS`` equal-mass
buckets of the hand's equity against one random hand (``strength.postflop_equity``), with bucket
edges fitted per street on random deals (``fit_edges``).

An information set key is ``"<street>:<bucket>:<history>"`` with streets in the history
separated by ``/`` (the acting player follows from the history).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pokerbot.opponents.strength import all_classes, hand_class, postflop_equity

STACK = 200  # chips (100 bb)
SB, BB = 1, 2
SIZES = {"1": 0.33, "2": 0.75, "3": 1.25}
RAISE_SIZES = "2"  # sizes allowed when facing a bet
RAISE_CAP = {0: 4, 1: 2, 2: 2, 3: 2}  # per street; preflop counts the big blind
BUCKETS = 12
CLASSES = {c: i for i, c in enumerate(all_classes())}


@dataclass
class State:
    """Abstract betting state. ``put`` counts all chips each player has put in this hand."""

    street: int = 0
    put: list[int] = field(default_factory=lambda: [SB, BB])
    street_put: list[int] = field(default_factory=lambda: [SB, BB])
    to_act: int = 0
    hist: str = ""
    raises: int = 1  # the big blind counts as the first bet preflop
    acted: int = 0  # decisions this street
    last_raise: int = BB
    folded: int = -1

    def copy(self) -> State:
        return State(self.street, list(self.put), list(self.street_put), self.to_act, self.hist,
                     self.raises, self.acted, self.last_raise, self.folded)  # fmt: skip

    @property
    def pot(self) -> int:
        return self.put[0] + self.put[1]

    def to_call(self) -> int:
        return self.street_put[1 - self.to_act] - self.street_put[self.to_act]

    def stack(self, p: int) -> int:
        return STACK - self.put[p]

    def terminal(self) -> bool:
        return self.folded >= 0 or self.street == 4

    def all_in(self) -> bool:
        return self.stack(0) == 0 or self.stack(1) == 0


def raise_to(s: State, frac: float) -> int:
    """Street-total bet the actor raises to with a pot-fraction size (before clipping)."""
    call = s.to_call()
    target = s.street_put[1 - s.to_act] + frac * (s.pot + call)
    return int(round(max(target, s.street_put[1 - s.to_act] + s.last_raise)))


def legal(s: State) -> str:
    p = s.to_act
    call = s.to_call()
    acts = "fc" if call > 0 else "c"
    max_to = s.street_put[p] + s.stack(p)
    if s.stack(p) > call and s.stack(1 - p) > 0:
        if s.raises < RAISE_CAP[s.street]:
            for a, frac in SIZES.items():
                if call > 0 and a not in RAISE_SIZES:
                    continue
                if raise_to(s, frac) < 0.9 * max_to:
                    acts += a
        acts += "a"  # all-in stays available after the cap
    return acts


def apply(s: State, a: str) -> State:
    s = s.copy()
    p = s.to_act
    s.hist += a
    s.acted += 1
    if a == "f":
        s.folded = p
        return s
    if a == "c":
        amount = min(s.to_call(), s.stack(p))
    else:
        target = s.street_put[p] + s.stack(p) if a == "a" else raise_to(s, SIZES[a])
        target = min(target, s.street_put[p] + s.stack(p))
        s.last_raise = max(s.last_raise, target - s.street_put[1 - p])
        amount = target - s.street_put[p]
        s.raises += 1
    s.put[p] += amount
    s.street_put[p] += amount
    s.to_act = 1 - p
    closed = s.street_put[0] == s.street_put[1] and s.acted >= 2
    if a == "c" and s.street == 0 and s.acted == 1:
        closed = False  # the small blind completed: the big blind still has the option
    if closed or (a == "c" and s.all_in()):
        _next_street(s)
    return s


def _next_street(s: State) -> None:
    if s.all_in():
        s.street = 4  # no more decisions: run the board out
        return
    s.street += 1
    if s.street == 4:
        return
    s.hist += "/"
    s.street_put = [0, 0]
    s.to_act = 1  # the big blind acts first after the flop
    s.raises = 0
    s.acted = 0
    s.last_raise = BB


def infoset(s: State, bucket: int) -> str:
    return f"{s.street}:{bucket}:{s.hist}"


# ------------------------------------------------------------------------------- card buckets

_BOARD = {1: 3, 2: 4, 3: 5}


def fit_edges(samples: int = 20_000, seed: int = 0) -> list[list[float]]:
    """Equal-mass equity bucket edges for flop, turn and river from random deals."""
    rng = np.random.default_rng(seed)
    edges = []
    for street in (1, 2, 3):
        eqs = []
        for _ in range(samples):
            cards = rng.choice(52, size=2 + _BOARD[street], replace=False).tolist()
            eqs.append(postflop_equity((cards[0], cards[1]), tuple(cards[2:])))
        qs = np.quantile(eqs, np.linspace(0, 1, BUCKETS + 1)[1:-1])
        edges.append(qs.tolist())
    return edges


def bucket(hole: tuple[int, int], board: tuple[int, ...], street: int, edges) -> int:
    if street == 0:
        return CLASSES[hand_class(hole)]
    eq = postflop_equity(hole, board[: _BOARD[street]])
    return int(np.searchsorted(edges[street - 1], eq, side="right"))


def showdown_payoff(s: State, holes, board, rank) -> float:
    """Chips won by player 0 at a terminal state (folds or showdown on the full board)."""
    if s.folded >= 0:
        return -s.put[0] if s.folded == 0 else s.put[1]
    r0, r1 = rank(holes[0], board), rank(holes[1], board)
    if r0 == r1:
        return 0.0
    return s.put[1] if r0 < r1 else -s.put[0]
