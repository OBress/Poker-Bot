"""No-limit hold'em hand state machine.

One ``HandState`` plays one hand: blinds, four betting rounds, side pots and showdown.
It is driven step by step: read ``actor`` and ``legal()``, then ``apply(action)`` until
``finished``. Chip amounts are integers.

Rules implemented:
- Heads-up: the button posts the small blind, acts first preflop and last postflop.
- 3+ players: SB and BB sit left of the button; UTG opens preflop; postflop starts left
  of the button.
- Minimum raise is the size of the last full bet/raise (initially the big blind).
- Raising is only allowed if another live player could call more than the current bet.
- An all-in raise smaller than a full raise does not reopen betting for players who already acted.
- Side pots are built from total contributions; uncalled chips return to their owner.
- Showdown order and mucking follow real-table rules (see ``_reveal``); hands that can't
  win any pot are dead. Layers with the same contenders form one pot, and a split pot's odd
  chips all go to the winner closest to the left of the button (matches pokerkit).
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerbot.engine.actions import Action, ActionType, Street
from pokerbot.engine.cards import hand_rank
from pokerbot.engine.events import (
    BlindPosted,
    BoardDealt,
    Event,
    HandEnd,
    HandStart,
    HoleCards,
    Muck,
    PlayerAction,
    Showdown,
)

_BOARD_SIZE = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


@dataclass(frozen=True, slots=True)
class LegalActions:
    to_call: int  # chips needed to call (already capped at the stack)
    can_raise: bool
    min_raise_to: int
    max_raise_to: int

    @property
    def can_check(self) -> bool:
        return self.to_call == 0


@dataclass(frozen=True, slots=True)
class DecisionState:
    """What a seat sees when it must act."""

    seat: int
    hole: tuple[int, int]
    board: tuple[int, ...]
    street: Street
    pot: int  # all chips committed this hand, including current-street bets
    legal: LegalActions
    stacks: tuple[int, ...]  # chips behind, per seat
    bets: tuple[int, ...]  # current-street bets, per seat
    folded: tuple[bool, ...]
    button: int
    big_blind: int
    actions: tuple[PlayerAction, ...]  # every action so far this hand

    @property
    def n_players(self) -> int:
        return len(self.stacks)


class HandState:
    def __init__(
        self,
        stacks: list[int] | tuple[int, ...],
        button: int,
        deck: list[int],
        small_blind: int = 1,
        big_blind: int = 2,
        hand_id: int = 0,
    ) -> None:
        n = len(stacks)
        if n < 2:
            raise ValueError("need at least 2 players")
        if any(s <= 0 for s in stacks):
            raise ValueError("every seat must have chips")
        if len(deck) < 2 * n + 5:
            raise ValueError("deck too short")

        self.n = n
        self.button = button % n
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.hand_id = hand_id
        self.start_stacks = tuple(stacks)
        self.stacks = list(stacks)
        self.bets = [0] * n
        self.contrib = [0] * n
        self.folded = [False] * n
        self.acted = [False] * n
        self.hole = [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]
        self._runout = tuple(deck[2 * n : 2 * n + 5])
        self.board: tuple[int, ...] = ()
        self.street = Street.PREFLOP
        self.current_bet = 0
        self.last_raise_size = big_blind
        self.actor: int | None = None
        self.finished = False
        self.showdown = False
        self.payoffs: tuple[int, ...] = ()
        self.events: list[Event] = []
        self.actions: list[PlayerAction] = []
        self.last_aggressor: int | None = None  # last bettor/raiser on the current street

        self._emit(HandStart(hand_id, self.button, small_blind, big_blind, self.start_stacks))
        for seat in range(n):
            self._emit(HoleCards(seat, self.hole[seat]))

        if n == 2:
            sb, bb = self.button, (self.button + 1) % n
        else:
            sb, bb = (self.button + 1) % n, (self.button + 2) % n
        self._post(sb, small_blind)
        self._post(bb, big_blind)
        self.current_bet = max(self.bets)
        self.actor = self._next_actor(bb)
        if self.actor is None:
            self._end_street()

    # ----------------------------------------------------------------- queries

    @property
    def pot(self) -> int:
        return sum(self.contrib)

    def legal(self) -> LegalActions:
        i = self._require_actor()
        to_call = min(self.current_bet - self.bets[i], self.stacks[i])
        max_to = self.bets[i] + self.stacks[i]
        can_raise = (
            self.stacks[i] > to_call and not self.acted[i] and self._opponent_can_call_more(i)
        )
        if self.current_bet == 0:
            min_to = self.big_blind
        else:
            min_to = self.current_bet + self.last_raise_size
        return LegalActions(to_call, can_raise, min(min_to, max_to), max_to)

    def decision_state(self) -> DecisionState:
        i = self._require_actor()
        return DecisionState(
            seat=i,
            hole=self.hole[i],
            board=self.board,
            street=self.street,
            pot=self.pot,
            legal=self.legal(),
            stacks=tuple(self.stacks),
            bets=tuple(self.bets),
            folded=tuple(self.folded),
            button=self.button,
            big_blind=self.big_blind,
            actions=tuple(self.actions),
        )

    # ----------------------------------------------------------------- actions

    def apply(self, action: Action) -> None:
        """Apply the actor's action.

        Lenient by design so simple bots can't crash a simulation: folding when a check is
        free becomes a check, a raise when raising is illegal becomes a call, and raise
        amounts are clamped to the legal range.
        """
        i = self._require_actor()
        legal = self.legal()
        kind = action.type
        if kind == ActionType.FOLD and legal.to_call == 0:
            kind = ActionType.CHECK_CALL
        if kind == ActionType.BET_RAISE and not legal.can_raise:
            kind = ActionType.CHECK_CALL

        pot_before = self.pot
        facing = legal.to_call
        if kind == ActionType.FOLD:
            self.folded[i] = True
            added = 0
        elif kind == ActionType.CHECK_CALL:
            added = legal.to_call
            self._commit(i, added)
        else:
            target = max(legal.min_raise_to, min(action.amount, legal.max_raise_to))
            added = target - self.bets[i]
            self._commit(i, added)
            raise_size = target - self.current_bet
            if raise_size >= self.last_raise_size:
                self.last_raise_size = raise_size
                for j in range(self.n):
                    if j != i:
                        self.acted[j] = False
            self.current_bet = target
            self.last_aggressor = i
        self.acted[i] = True

        event = PlayerAction(
            seat=i,
            street=self.street,
            type=kind,
            amount=added,
            to=self.bets[i],
            pot_before=pot_before,
            to_call=facing,
            is_all_in=self.stacks[i] == 0 and kind != ActionType.FOLD,
        )
        self.actions.append(event)
        self._emit(event)

        if sum(not f for f in self.folded) == 1:
            self._finish()
            return
        self.actor = self._next_actor(i)
        if self.actor is None:
            self._end_street()

    # ----------------------------------------------------------------- internals

    def _emit(self, event: Event) -> None:
        self.events.append(event)

    def _require_actor(self) -> int:
        if self.finished or self.actor is None:
            raise RuntimeError("hand is finished")
        return self.actor

    def _commit(self, seat: int, amount: int) -> None:
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.contrib[seat] += amount

    def _post(self, seat: int, blind: int) -> None:
        amount = min(blind, self.stacks[seat])
        self._commit(seat, amount)
        self._emit(BlindPosted(seat, amount))

    def _needs_to_act(self, i: int) -> bool:
        if self.folded[i] or self.stacks[i] == 0:
            return False
        if self.bets[i] < self.current_bet:
            return True
        if self.acted[i]:
            return False
        # Unacted with nothing to call: only worth acting if a bet could be called.
        return self._opponent_can_call_more(i)

    def _opponent_can_call_more(self, i: int) -> bool:
        """Whether any other live player could put in more than the current bet. If not,
        raising is pointless (the excess would come straight back) and isn't allowed."""
        return any(
            j != i and not self.folded[j] and self.bets[j] + self.stacks[j] > self.current_bet
            for j in range(self.n)
        )

    def _next_actor(self, after: int) -> int | None:
        for k in range(1, self.n + 1):
            j = (after + k) % self.n
            if self._needs_to_act(j):
                return j
        return None

    def _end_street(self) -> None:
        while True:
            self.bets = [0] * self.n
            self.acted = [False] * self.n
            self.current_bet = 0
            self.last_raise_size = self.big_blind
            if self.street == Street.RIVER:
                self._finish()
                return
            self.street = Street(self.street + 1)
            self.board = self._runout[: _BOARD_SIZE[self.street]]
            self.last_aggressor = None
            self._emit(BoardDealt(self.street, self.board))
            self.actor = self._next_actor(self.button)
            if self.actor is not None:
                return

    def _finish(self) -> None:
        self.actor = None
        self.finished = True
        live = [i for i in range(self.n) if not self.folded[i]]
        won = [0] * self.n
        if len(live) == 1:
            won[live[0]] = self.pot
        else:
            self.showdown = True
            ranks = {i: hand_rank(self.hole[i], self.board) for i in live}
            self._reveal(live, ranks)
            # Hands that can't win any pot are dead; pots are rebuilt over the remaining
            # contenders so layers with the same contenders merge (matters for odd chips).
            contenders = {
                w for _, eligible in self._pots(live) for w in self._winners(eligible, ranks)
            }
            for amount, eligible in self._pots(sorted(contenders)):
                winners = self._winners(eligible, ranks)
                share, odd = divmod(amount, len(winners))
                for w in winners:
                    won[w] += share
                won[winners[0]] += odd
        final = [self.stacks[i] + won[i] for i in range(self.n)]
        self.stacks = final
        self.payoffs = tuple(final[i] - self.start_stacks[i] for i in range(self.n))
        self._emit(HandEnd(self.hand_id, self.payoffs, self.showdown))

    def _reveal(self, live: list[int], ranks: dict[int, int]) -> None:
        """Emit showdown reveals in the order and with the mucks a real table would show.

        - If any live player is all-in, every live hand is tabled (TDA all-in rule).
        - Otherwise the last river bettor/raiser shows first; if the river checked through,
          the first live player left of the button does. Play goes clockwise, and each later
          player shows only a hand that beats or ties the best shown so far; others muck.
        Without an all-in there is a single pot, so "can win" means "beats or ties".
        """
        if any(self.stacks[i] == 0 for i in live):
            for i in self._from_button_left(live):
                self._emit(Showdown(i, self.hole[i], ranks[i]))
            return
        first = self.last_aggressor if self.last_aggressor in live else None
        order = self._from_button_left(live)
        if first is not None:
            k = order.index(first)
            order = order[k:] + order[:k]
        best = None
        for i in order:
            if best is None or ranks[i] <= best:
                best = ranks[i]
                self._emit(Showdown(i, self.hole[i], ranks[i]))
            else:
                self._emit(Muck(i))

    def _winners(self, eligible: list[int], ranks: dict[int, int]) -> list[int]:
        best = min(ranks[i] for i in eligible)
        return self._from_button_left([i for i in eligible if ranks[i] == best])

    def _pots(self, contenders: list[int]) -> list[tuple[int, list[int]]]:
        """Main pot and side pots as (amount, eligible seats), built from total
        contributions. Consecutive layers with the same eligible seats form one pot."""
        allowed = set(contenders)
        pots: list[tuple[int, list[int]]] = []
        carry = 0
        prev = 0
        for level in sorted({c for c in self.contrib if c > 0}):
            amount = sum(min(c, level) - min(c, prev) for c in self.contrib) + carry
            eligible = [i for i in range(self.n) if i in allowed and self.contrib[i] >= level]
            if eligible and pots and pots[-1][1] == eligible:
                pots[-1] = (pots[-1][0] + amount, eligible)
                carry = 0
            elif eligible:
                pots.append((amount, eligible))
                carry = 0
            else:
                carry = amount
            prev = level
        if carry and pots:
            last_amount, last_eligible = pots[-1]
            pots[-1] = (last_amount + carry, last_eligible)
        return pots

    def _from_button_left(self, seats: list[int]) -> list[int]:
        return sorted(seats, key=lambda s: (s - self.button - 1) % self.n)
